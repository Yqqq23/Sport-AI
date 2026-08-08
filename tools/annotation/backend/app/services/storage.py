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
    if _FFMPEG is not None:
        return _extract_frames_ffmpeg_cv2(video_path, output_dir, frame_interval)
    return _extract_frames_cv2(video_path, output_dir, frame_interval)


def _transcode_to_h264(video_path: Path) -> Path:
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    result = subprocess.run(
        [_FFMPEG, "-y", "-i", str(video_path.resolve()),
         "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-preset", "fast", "-crf", "23",
         str(tmp_path.resolve())],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg transcode failed: {result.stderr}")
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