"""把轨迹画到俯视球场图上。

直接复用 ``calibration.topdown.CourtRenderer`` —— 球场画法只有一份，
不在这里重画一遍球场，否则两边尺寸一改就对不上。

产出
----
    trajectory_<label>.png      轨迹图（按速度着色）
    heatmap_<label>.png         热区图（0.5 m 网格）
    trajectory_<label>.mp4      轨迹动画（可选）
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import cv2
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from calibration.io_utils import imwrite_unicode  # noqa: E402
from calibration.topdown import CourtRenderer  # noqa: E402

from .paths import Track, occupancy_grid  # noqa: E402

LOGGER = logging.getLogger(__name__)

#: 速度着色的上限（m/s）。人类冲刺 ~9-10 m/s，封顶到 8 让常见速度铺满色阶。
SPEED_MAX_MPS = 8.0


def _speed_colors(track: Track) -> np.ndarray:
    """每一段的 BGR 颜色，按瞬时速度取 TURBO 色图。"""
    sp = track.speed_mps
    if sp.size == 0:
        return np.zeros((0, 3), np.uint8)
    v = np.clip(np.nan_to_num(sp, nan=0.0) / SPEED_MAX_MPS, 0.0, 1.0)
    lut = cv2.applyColorMap((v * 255).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_TURBO)
    return lut.reshape(-1, 3)


def _legend(img, rend: CourtRenderer, label: str, track: Track) -> None:
    h, w = img.shape[:2]
    bar_w, bar_h = 220, 14
    x0, y0 = w - bar_w - 24, h - 54
    grad = cv2.applyColorMap(
        np.tile(np.linspace(0, 255, bar_w, dtype=np.uint8), (bar_h, 1)), cv2.COLORMAP_TURBO)
    img[y0:y0 + bar_h, x0:x0 + bar_w] = grad
    cv2.rectangle(img, (x0 - 1, y0 - 1), (x0 + bar_w, y0 + bar_h), (40, 40, 40), 1)
    cv2.putText(img, "0", (x0 - 4, y0 + bar_h + 16), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.putText(img, f"{SPEED_MAX_MPS:g} m/s", (x0 + bar_w - 52, y0 + bar_h + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1, cv2.LINE_AA)

    lines = [
        f"player {label}",
        f"{track.distance_m:.0f} m over {track.duration_sec:.0f} s",
        f"{len(track)} samples ({int(track.interpolated.sum())} filled)",
    ]
    for i, s in enumerate(lines):
        cv2.putText(img, s, (18, 30 + i * 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6 if i == 0 else 0.5, (25, 25, 25), 2 if i == 0 else 1, cv2.LINE_AA)


def draw_trajectory(track: Track, out_path: str | Path, *, ppm: float = 60.0,
                    show_raw: bool = False, thickness: int = 3) -> Path:
    """画一张轨迹图：球场底图 + 按速度着色的路径。"""
    rend = CourtRenderer(ppm=ppm)
    img = rend.draw_court(grid=True)
    px = rend.to_px_i(track.xy)

    if show_raw:
        raw = rend.to_px_i(track.xy_raw)
        for i in range(len(raw) - 1):
            cv2.line(img, tuple(raw[i]), tuple(raw[i + 1]), (185, 185, 185), 1, cv2.LINE_AA)

    cols = _speed_colors(track)
    for i in range(len(px) - 1):
        c = tuple(int(v) for v in cols[i]) if i < len(cols) else (60, 60, 60)
        cv2.line(img, tuple(px[i]), tuple(px[i + 1]), c, thickness, cv2.LINE_AA)

    # 起点绿、终点红，方向一目了然
    if len(px):
        cv2.circle(img, tuple(px[0]), 9, (60, 190, 60), -1, cv2.LINE_AA)
        cv2.circle(img, tuple(px[0]), 9, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(img, "start", (px[0][0] + 12, px[0][1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (25, 25, 25), 1, cv2.LINE_AA)
        cv2.circle(img, tuple(px[-1]), 9, (60, 60, 220), -1, cv2.LINE_AA)
        cv2.circle(img, tuple(px[-1]), 9, (20, 20, 20), 2, cv2.LINE_AA)
        # 闭环轨迹的起点和终点会重合，标签叠在一起就都看不清了：挤在一起就把
        # end 的标签放到下面去。
        dy = -8 if np.hypot(*(px[-1] - px[0])) > 40 else 22
        cv2.putText(img, "end", (px[-1][0] + 12, px[-1][1] + dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (25, 25, 25), 1, cv2.LINE_AA)

    _legend(img, rend, track.label, track)
    out_path = Path(out_path)
    imwrite_unicode(str(out_path), img)
    LOGGER.info("已写出 -> %s", out_path)
    return out_path


def draw_heatmap(track: Track, out_path: str | Path, *, ppm: float = 60.0,
                 cell_m: float = 0.5, alpha: float = 0.72) -> Path:
    """画热区图：球员在球场各处停留的时间分布。"""
    rend = CourtRenderer(ppm=ppm)
    base = rend.draw_court(grid=False)
    grid, _ = occupancy_grid(track, cell_m=cell_m)
    if grid.max() == 0:
        LOGGER.warning("%s 没有落在场内的点，热区图是空的", track.label)

    # 归一化后放大到画布尺寸；用平方根压一下长尾，否则一个久站点会吃掉整张图
    norm = np.sqrt(grid.astype(np.float32) / max(grid.max(), 1))
    big = cv2.resize(norm, (int(rend.w_m * rend.ppm), int(rend.l_m * rend.ppm)),
                     interpolation=cv2.INTER_LINEAR)
    big = cv2.GaussianBlur(big, (0, 0), sigmaX=rend.ppm * 0.35)
    if big.max() > 0:
        big = big / big.max()
    heat = cv2.applyColorMap((big * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)

    x0, y0 = rend.to_px_i([[0.0, rend.l_m]])[0]
    h, w = heat.shape[:2]
    roi = base[y0:y0 + h, x0:x0 + w]
    mask = (big > 0.04)[..., None]
    roi[:] = np.where(mask, (alpha * heat + (1 - alpha) * roi).astype(np.uint8), roi)

    cv2.putText(base, f"player {track.label}  occupancy  ({cell_m:g} m cells)",
                (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (25, 25, 25), 2, cv2.LINE_AA)
    out_path = Path(out_path)
    imwrite_unicode(str(out_path), base)
    LOGGER.info("已写出 -> %s", out_path)
    return out_path


def render_video(track: Track, out_path: str | Path, *, ppm: float = 60.0,
                 fps: float = 30.0, trail_sec: float = 4.0, fourcc: str = "mp4v") -> Path:
    """轨迹动画：一个点在跑，后面拖一条渐隐的尾巴。"""
    rend = CourtRenderer(ppm=ppm)
    base = rend.draw_court(grid=True)
    px = rend.to_px_i(track.xy)
    cols = _speed_colors(track)
    trail_n = max(2, int(trail_sec * fps))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*fourcc),
                             fps, (rend.width, rend.height))
    if not writer.isOpened():
        raise RuntimeError(
            f"打不开视频写出器: {out_path}（fourcc={fourcc}）。"
            " 服务器上常见原因是 OpenCV 没带对应编码器，换 --fourcc MJPG 并用 .avi 后缀。"
        )
    try:
        for k in range(len(px)):
            frame = base.copy()
            lo = max(0, k - trail_n)
            for i in range(lo, k):
                a = (i - lo + 1) / max(k - lo, 1)          # 越旧越淡
                c = tuple(int(v) for v in cols[i]) if i < len(cols) else (60, 60, 60)
                c = tuple(int(255 - a * (255 - v)) for v in c)
                cv2.line(frame, tuple(px[i]), tuple(px[i + 1]), c, 3, cv2.LINE_AA)
            cv2.circle(frame, tuple(px[k]), 10, (40, 40, 240), -1, cv2.LINE_AA)
            cv2.circle(frame, tuple(px[k]), 10, (20, 20, 20), 2, cv2.LINE_AA)
            cv2.putText(frame, f"{track.label}   t={track.t[k]:6.1f}s   "
                               f"({track.xy[k,0]:5.2f}, {track.xy[k,1]:5.2f}) m",
                        (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (25, 25, 25), 2, cv2.LINE_AA)
            writer.write(frame)
    finally:
        writer.release()
    LOGGER.info("已写出 -> %s", out_path)
    return out_path
