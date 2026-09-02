import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2

from app.core.config import settings


LOGGER = logging.getLogger(__name__)


def _find_ffmpeg() -> str | None:
    candidates = [
        "ffmpeg",
        str(Path(__file__).resolve().parent.parent.parent / "ffmpeg" / "ffmpeg"),
        str(Path.home() / "ffmpeg" / "ffmpeg"),
        str(Path.home() / "jersey-active-learning" / "ffmpeg" / "ffmpeg"),
        "/usr/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
    ]
    for candidate in candidates:
        try:
            result = subprocess.run([candidate, "-version"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return candidate
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return None


_FFMPEG = _find_ffmpeg()

#: 本次进程走的是哪条抽帧路径。写进 video_assets.decode_path，因为这两条路
#: 数的**不是同一个流**：转码路数的是转码产物，直读路数的是原视频。帧号语义
#: 因此取决于这台机器装没装 ffmpeg —— 本机和 GPU 服务器不一致的话，同一份
#: 代码会产出两套编号。不落库就只能靠猜。
DECODE_PATH_TRANSCODE = "ffmpeg_transcode"
DECODE_PATH_DIRECT = "direct"


def decode_path() -> str:
    """本次抽帧会走哪条路。上传时落库，事后能查。"""
    return DECODE_PATH_TRANSCODE if _FFMPEG is not None else DECODE_PATH_DIRECT


LOGGER.info(
    "frame extraction backend = %s (ffmpeg: %s)",
    decode_path(), _FFMPEG or "not found",
)


class FrameCountMismatchError(RuntimeError):
    """转码前后帧数不一致 —— 帧号已经不可信，不要继续。"""


def _probe_frame_count(path: Path) -> int:
    """视频有多少帧。拿不到返回 -1（不同于 0 帧，0 帧是真的空）。"""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return -1
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        return count if count > 0 else -1
    finally:
        cap.release()


def ensure_data_dirs() -> None:
    for name in ["videos", "frames", "crops", "annotations", "exports"]:
        (settings.data_dir / name).mkdir(parents=True, exist_ok=True)


def video_output_path(filename: str) -> Path:
    ensure_data_dirs()
    return settings.data_dir / "videos" / filename


def frame_output_dir(video_id: int) -> Path:
    ensure_data_dirs()
    path = settings.data_dir / "frames" / str(video_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def extract_frames(video_path: Path, output_dir: Path, frame_interval: int) -> list[tuple[int, Path, int, int]]:
    """抽帧。返回 ``(真实帧号, 路径, 宽, 高)``。

    帧号是**解码序号**，不是抽样序号：计数器每读一帧就 +1，只在
    ``% frame_interval == 0`` 时落盘。所以文件是 000000/000030/000060…，
    和原视频帧号一一对应，下游 mine 不需要换算。
    """
    LOGGER.info(
        "extract_frames video=%s interval=%s decode_path=%s",
        video_path.name, frame_interval, decode_path(),
    )
    if _FFMPEG is not None:
        return _extract_frames_ffmpeg_cv2(video_path, output_dir, frame_interval)
    return _extract_frames_cv2(video_path, output_dir, frame_interval)


def probe_fps(video_path: Path) -> float:
    """视频帧率。读不到返回 0.0，调用方自己决定怎么记。"""
    cap = cv2.VideoCapture(str(video_path))
    try:
        return float(cap.get(cv2.CAP_PROP_FPS) or 0.0) if cap.isOpened() else 0.0
    finally:
        cap.release()


def _transcode_to_h264(video_path: Path) -> Path:
    """转成 H.264 供 OpenCV 稳定解码，**逐帧一一对应**，不补不丢。

    为什么必须保帧
    --------------
    落盘的帧号（frame_%06d.jpg）是在**转码产物**上数出来的，而下游挖训练
    数据的 mine 是在**原视频**上数的。两个编号相等，靠的就是转码 1:1 保帧。

    原来的命令没有 -fps_mode / -vsync。ffmpeg 默认会把 VFR 源转成 CFR，
    也就是补帧或丢帧。CFR 源（现有的 30fps 素材）碰巧 1:1，所以一直没暴露；
    换成手机直录或录屏这类 VFR 素材，帧号会整体错位，而且**不报错** ——
    标注框会落到别的帧上，事后极难查。

    ``-fps_mode passthrough`` 让每一个输入帧原样出一帧。老版 ffmpeg 不认这个
    参数，退回等价的 ``-vsync 0``。
    """
    def _run(mode_args: list[str]) -> tuple[Path, subprocess.CompletedProcess]:
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        path = Path(tmp.name)
        tmp.close()
        completed = subprocess.run(
            [_FFMPEG, "-y", "-i", str(video_path.resolve()),
             *mode_args,
             "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-preset", "fast", "-crf", "23",
             str(path.resolve())],
            capture_output=True, text=True, timeout=120,
        )
        return path, completed

    tmp_path, result = _run(["-fps_mode", "passthrough"])
    if result.returncode != 0:
        tmp_path.unlink(missing_ok=True)
        LOGGER.info("ffmpeg rejected -fps_mode, retrying with legacy -vsync 0")
        tmp_path, result = _run(["-vsync", "0"])
    if result.returncode != 0:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg transcode failed: {result.stderr}")

    source_frames = _probe_frame_count(video_path)
    transcoded_frames = _probe_frame_count(tmp_path)
    LOGGER.info(
        "transcode frame count: source=%s transcoded=%s (%s)",
        source_frames, transcoded_frames, video_path.name,
    )
    if source_frames > 0 and transcoded_frames > 0 and source_frames != transcoded_frames:
        tmp_path.unlink(missing_ok=True)
        raise FrameCountMismatchError(
            f"转码前后帧数不一致：原视频 {source_frames} 帧，转码后 {transcoded_frames} 帧"
            f"（{video_path.name}）。\n"
            "  这几乎肯定是 VFR（可变帧率）素材 —— 手机直录、录屏、某些剪辑软件的导出。\n"
            "  继续跑下去的后果：落盘的 frame_%06d.jpg 编号是转码流的编号，\n"
            "  和原视频的真实帧号对不上，而 mine 挖训练数据数的是原视频。\n"
            "  标注框会静默落到错误的帧上，导出的数据集从此不可信。\n"
            "\n"
            "  处理办法：先用 ffmpeg 把素材离线转成 CFR 再上传，例如\n"
            f"      ffmpeg -i \"{video_path.name}\" -fps_mode cfr -r 30 -c:v libx264 out.mp4\n"
            "  然后传 out.mp4。转出来的帧号从此就是这个文件自己的帧号。"
        )
    if source_frames <= 0 or transcoded_frames <= 0:
        LOGGER.warning(
            "无法读出帧数用于校验（source=%s transcoded=%s）。帧号一致性这次没验上。",
            source_frames, transcoded_frames,
        )
    return tmp_path


def _extract_frames_ffmpeg_cv2(video_path: Path, output_dir: Path, frame_interval: int) -> list[tuple[int, Path, int, int]]:
    interval = max(1, frame_interval)
    transcode_path = None
    try:
        transcode_path = _transcode_to_h264(video_path)
        cap = cv2.VideoCapture(str(transcode_path.resolve()))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open transcoded video: {transcode_path}")
        saved: list[tuple[int, Path, int, int]] = []
        frame_index = 0
        try:
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_index % interval == 0:
                    frame_name = f"frame_{frame_index:06d}.jpg"
                    frame_path = output_dir / frame_name
                    if not cv2.imwrite(str(frame_path), frame):
                        raise RuntimeError(f"Failed to write frame: {frame_path}")
                    h, w = frame.shape[:2]
                    saved.append((frame_index, frame_path, int(w), int(h)))
                if frame_index > 0 and frame_index % 30 == 0:
                    LOGGER.info("frame extraction progress video=%s processed=%s saved=%s", video_path.name, frame_index, len(saved))
                frame_index += 1
        finally:
            cap.release()
    finally:
        if transcode_path is not None:
            try:
                transcode_path.unlink(missing_ok=True)
            except Exception:
                pass
    if not saved:
        raise RuntimeError(f"No frames extracted from {video_path}")
    return saved


def _extract_frames_cv2(video_path: Path, output_dir: Path, frame_interval: int) -> list[tuple[int, Path, int, int]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    saved: list[tuple[int, Path, int, int]] = []
    frame_index = 0
    interval = max(1, frame_interval)
    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % interval == 0:
                frame_name = f"frame_{frame_index:06d}.jpg"
                frame_path = output_dir / frame_name
                if not cv2.imwrite(str(frame_path), frame):
                    raise RuntimeError(f"Failed to write frame: {frame_path}")
                h, w = frame.shape[:2]
                saved.append((frame_index, frame_path, int(w), int(h)))
            if frame_index > 0 and frame_index % 30 == 0:
                LOGGER.info("frame extraction progress video=%s processed=%s saved=%s", video_path.name, frame_index, len(saved))
            frame_index += 1
    finally:
        cap.release()
    if not saved:
        raise RuntimeError(f"No frames extracted from {video_path}")
    return saved