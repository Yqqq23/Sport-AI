"""俯视图（bird's-eye view）构建。

    python -m calibration.topdown --background court_plate.png
    python -m calibration.topdown --background court_plate.png --extent -6 21 -6 20

产出：

1. ``topdown_court.png``     纯球场示意图，画法照 standardone.png（竖版，底线在下）
2. ``topdown_warped.png``    把真实画面按单应**整幅拉平**成俯视图
3. ``topdown_measure.png``   拉平画面 + 带刻度的 1 米网格，可直接把地板线读成米
4. ``topdown_overlay.png``   示意图 + 拉平画面叠加，用来肉眼验矩阵

视野范围
--------
默认只渲染球场本身（0–15 m × 0–14 m）。地板上的线常常伸到球场外，
用 ``--extent XMIN XMAX YMIN YMAX``（单位米）把视野放大，
负数就是球场外侧。例如 ``--extent -6 21 -6 20``。

⚠ 投影球员位置时取检测框**底边中点**（脚），不是框中心。
   原因见 homography.py 顶部 —— 单应只对地平面成立，胸口不在地面上。
   直接用 ``ViewTransformer.transform_boxes_to_court()`` 就是对的。
"""

from __future__ import annotations

import argparse
import os

import cv2
import numpy as np

from .calibrate import OUTPUT_JSON, load_calibration, transformer_from_calibration
from .court_config import (BASKET_DIST_MM, CONFIG, COURT_LENGTH_MM, COURT_WIDTH_MM,
                           FREE_THROW_DIST_MM, NUM_POINTS, PAINT_WIDTH_MM,
                           THREE_INSET_MM, THREE_RADIUS_MM)
from .io_utils import asset, calib, imread_unicode, imwrite_unicode, read_video_frame

COURT_PNG = calib("topdown_court.png")
WARPED_PNG = calib("topdown_warped.png")
MEASURE_PNG = calib("topdown_measure.png")
OVERLAY_PNG = calib("topdown_overlay.png")

VIDEO_NAME = "3 v 3 Basketball Game Side A 24022026.mp4"
DEFAULT_FRAME = 6000

# 篮下限制区半圆半径（FIBA no-charge semicircle），米
NO_CHARGE_R_M = 1.25

# 配色（BGR），照 standardone.png 的木地板 + 白线风格
C_FLOOR = (150, 196, 222)
C_LINE = (255, 255, 255)
C_KEY = (232, 244, 250)
C_EDGE = (54, 54, 190)
C_GRID = (120, 168, 198)
C_TEXT = (40, 40, 40)


class CourtRenderer:
    """球场米 ↔ 俯视图像素。

    俯视图坐标系（和 standardone.png 一致）：
        底线画在**下方**，中线在上方，X 向右。
        球场米的 +Y 是「远离底线」，图像 y 向下，所以 Y 要翻转。

    ``x_range`` / ``y_range`` 以米给出渲染视野；默认正好是球场本身。
    要看球场外的线就把范围放大（可以给负数）。
    """

    def __init__(self, ppm: float = 60.0, margin: int = 60,
                 x_range: tuple[float, float] | None = None,
                 y_range: tuple[float, float] | None = None):
        self.ppm = float(ppm)
        self.margin = int(margin)
        self.w_m = COURT_WIDTH_MM / 1000.0
        self.l_m = COURT_LENGTH_MM / 1000.0
        self.x0, self.x1 = x_range if x_range else (0.0, self.w_m)
        self.y0, self.y1 = y_range if y_range else (0.0, self.l_m)
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError(f"视野范围非法: x {self.x0}..{self.x1}, y {self.y0}..{self.y1}")
        self.width = int(round((self.x1 - self.x0) * self.ppm)) + 2 * self.margin
        self.height = int(round((self.y1 - self.y0) * self.ppm)) + 2 * self.margin

    # -- 坐标 --------------------------------------------------------------
    def to_px(self, pts_m) -> np.ndarray:
        """(N,2) 球场米 → (N,2) 俯视图像素。"""
        p = np.asarray(pts_m, dtype=np.float64).reshape(-1, 2)
        x = (p[:, 0] - self.x0) * self.ppm + self.margin
        y = self.height - self.margin - (p[:, 1] - self.y0) * self.ppm
        return np.stack([x, y], axis=1)

    def to_px_i(self, pts_m) -> np.ndarray:
        return np.round(self.to_px(pts_m)).astype(np.int32)

    def to_metres(self, pts_px) -> np.ndarray:
        """俯视图像素 → 球场米（读图用的逆运算）。"""
        p = np.asarray(pts_px, dtype=np.float64).reshape(-1, 2)
        x = (p[:, 0] - self.margin) / self.ppm + self.x0
        y = (self.height - self.margin - p[:, 1]) / self.ppm + self.y0
        return np.stack([x, y], axis=1)

    @property
    def court_to_topdown(self) -> np.ndarray:
        """3×3 齐次矩阵：球场米 → 俯视图像素。用来和单应串联。"""
        return np.array(
            [[self.ppm, 0.0, self.margin - self.x0 * self.ppm],
             [0.0, -self.ppm, self.height - self.margin + self.y0 * self.ppm],
             [0.0, 0.0, 1.0]], dtype=np.float64)

    # -- 画底图 ------------------------------------------------------------
    @staticmethod
    def _arc(cx, cy, r, a0, a1, n=240):
        t = np.linspace(a0, a1, n)
        return np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], axis=1)

    def _grid_ticks(self):
        gx = range(int(np.floor(self.x0)), int(np.ceil(self.x1)) + 1)
        gy = range(int(np.floor(self.y0)), int(np.ceil(self.y1)) + 1)
        return gx, gy

    def draw_court(self, grid: bool = True) -> np.ndarray:
        """画一张空球场俯视图。"""
        img = np.full((self.height, self.width, 3), C_FLOOR, dtype=np.uint8)
        W, L = self.w_m, self.l_m
        ft = FREE_THROW_DIST_MM / 1000.0
        bk = BASKET_DIST_MM / 1000.0
        r3 = THREE_RADIUS_MM / 1000.0
        ins = THREE_INSET_MM / 1000.0
        ph = PAINT_WIDTH_MM / 2000.0
        cx = W / 2.0

        if grid:
            gxs, gys = self._grid_ticks()
            for gx in gxs:
                cv2.line(img, *map(tuple, self.to_px_i([[gx, self.y0], [gx, self.y1]])),
                         C_GRID, 1, cv2.LINE_AA)
            for gy in gys:
                cv2.line(img, *map(tuple, self.to_px_i([[self.x0, gy], [self.x1, gy]])),
                         C_GRID, 1, cv2.LINE_AA)

        cv2.fillPoly(img, [self.to_px_i([[cx - ph, 0], [cx + ph, 0],
                                         [cx + ph, ft], [cx - ph, ft]])], C_KEY)

        brk_y = bk + np.sqrt(max(r3 ** 2 - (cx - ins) ** 2, 0.0))
        phi0 = np.arctan2(brk_y - bk, cx - ins)
        cv2.polylines(img, [self.to_px_i([[ins, 0], [ins, brk_y]])], False, C_LINE, 3, cv2.LINE_AA)
        cv2.polylines(img, [self.to_px_i([[W - ins, 0], [W - ins, brk_y]])], False, C_LINE, 3, cv2.LINE_AA)
        cv2.polylines(img, [self.to_px_i(self._arc(cx, bk, r3, phi0, np.pi - phi0))],
                      False, C_LINE, 3, cv2.LINE_AA)

        # 限制区边框（罚球线在远端那条边）
        cv2.polylines(img, [self.to_px_i([[cx - ph, 0], [cx - ph, ft],
                                          [cx + ph, ft], [cx + ph, 0]])], False, C_LINE, 3, cv2.LINE_AA)

        # 罚球圈：限制区外那半实线，圈内那半虚线（FIBA 画法；
        # sample.jpg 只画了实线那半，这里保留虚线半圈，信息更全）
        cv2.polylines(img, [self.to_px_i(self._arc(cx, ft, 1.8, 0, np.pi))],
                      False, C_LINE, 3, cv2.LINE_AA)
        dash = self._arc(cx, ft, 1.8, np.pi, 2 * np.pi, 120)
        for k in range(0, len(dash) - 6, 12):
            cv2.polylines(img, [self.to_px_i(dash[k:k + 6])], False, C_LINE, 2, cv2.LINE_AA)

        # 篮下限制区半圆（no-charge semicircle），半径 1.25 m，圆心 = 篮筐点，
        # 开口朝底线。sample.jpg 里限制区内靠底线那道弧就是它 —— 之前漏画了。
        cv2.polylines(img, [self.to_px_i(self._arc(cx, bk, NO_CHARGE_R_M, 0, np.pi))],
                      False, C_LINE, 3, cv2.LINE_AA)
        # 半圆两端连到底线的短竖线
        for sx in (cx - NO_CHARGE_R_M, cx + NO_CHARGE_R_M):
            cv2.polylines(img, [self.to_px_i([[sx, bk], [sx, 1.2]])], False, C_LINE, 2, cv2.LINE_AA)

        # 篮板（距底线 1.20 m，宽 1.80 m）+ 篮筐（内径 0.45 m）
        cv2.polylines(img, [self.to_px_i([[cx - 0.9, 1.2], [cx + 0.9, 1.2]])],
                      False, C_TEXT, 4, cv2.LINE_AA)
        bp = self.to_px_i([[cx, bk]])[0]
        cv2.circle(img, tuple(bp), max(int(0.225 * self.ppm), 2), C_EDGE, 3, cv2.LINE_AA)

        cv2.polylines(img, [self.to_px_i(self._arc(cx, L, 1.8, np.pi, 2 * np.pi))],
                      False, C_LINE, 3, cv2.LINE_AA)
        cv2.polylines(img, [self.to_px_i([[0, 0], [W, 0], [W, L], [0, L]])],
                      True, C_LINE, 4, cv2.LINE_AA)
        return img

    # -- 画点 --------------------------------------------------------------
    def draw_points(self, img, pts_m, labels=None, color=(60, 200, 60), radius=7):
        """把球场坐标画到俯视图上。越界的点画成空心，提示它出界了。"""
        pts_m = np.asarray(pts_m, dtype=np.float64).reshape(-1, 2)
        for i, (mx, my) in enumerate(pts_m):
            if not np.isfinite([mx, my]).all():
                continue
            inside = (0 <= mx <= self.w_m) and (0 <= my <= self.l_m)
            p = tuple(self.to_px_i([[mx, my]])[0])
            cv2.circle(img, p, radius, color, -1 if inside else 2, cv2.LINE_AA)
            if labels is not None:
                t = str(labels[i])
                cv2.putText(img, t, (p[0] + radius + 3, p[1] - radius),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 4, cv2.LINE_AA)
                cv2.putText(img, t, (p[0] + radius + 3, p[1] - radius),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        return img

    def draw_measure_grid(self, img):
        """带刻度的 1 米网格，每 5 米一条粗线并标数字。"""
        gxs, gys = self._grid_ticks()
        for gx in gxs:
            a, b = self.to_px_i([[gx, self.y0], [gx, self.y1]])
            major = gx % 5 == 0
            cv2.line(img, tuple(a), tuple(b), (0, 140, 255) if major else (95, 95, 95),
                     2 if major else 1, cv2.LINE_AA)
            cv2.putText(img, str(gx), (a[0] + 3, self.height - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 140, 255), 1, cv2.LINE_AA)
        for gy in gys:
            a, b = self.to_px_i([[self.x0, gy], [self.x1, gy]])
            major = gy % 5 == 0
            cv2.line(img, tuple(a), tuple(b), (0, 140, 255) if major else (95, 95, 95),
                     2 if major else 1, cv2.LINE_AA)
            cv2.putText(img, str(gy), (5, a[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 140, 255), 1, cv2.LINE_AA)
        return img


def warp_to_topdown(frame: np.ndarray, vt, renderer: CourtRenderer) -> np.ndarray:
    """把一帧画面整幅拉平成俯视图。

    串联两个变换：画面像素 --H--> 球场米 --renderer--> 俯视图像素。
    """
    M = renderer.court_to_topdown @ vt.m
    return cv2.warpPerspective(frame, M, (renderer.width, renderer.height),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(30, 30, 30))


def _banner(im, text):
    out = im.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (20, 20, 20), -1)
    cv2.putText(out, text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def build(calibration_path: str = OUTPUT_JSON, frame_index: int = DEFAULT_FRAME,
          ppm: float = 60.0, background: str | None = None,
          extent: tuple[float, float, float, float] | None = None) -> None:
    data = load_calibration(calibration_path)
    vt = transformer_from_calibration(data)
    xr = (extent[0], extent[1]) if extent else None
    yr = (extent[2], extent[3]) if extent else None
    rend = CourtRenderer(ppm=ppm, x_range=xr, y_range=yr)
    print(f"视野 : X {rend.x0:g}..{rend.x1:g} m   Y {rend.y0:g}..{rend.y1:g} m   "
          f"-> {rend.width}x{rend.height} px @ {ppm:g} px/m")

    # 1) 纯示意图 + 已确认的标定点
    court = rend.draw_court()
    defined = CONFIG.defined_mask
    used_ids = set(data["point_ids_used"])
    for i in range(NUM_POINTS):
        if not defined[i]:
            continue
        col = (60, 170, 60) if (i + 1) in used_ids else (150, 150, 150)
        rend.draw_points(court, CONFIG.vertices[i:i + 1], [i + 1], col)
    imwrite_unicode(COURT_PNG, court)
    print(f"已写出 -> {COURT_PNG}")

    # 2) 真实画面拉平
    if background:
        src_path = background if os.path.isabs(background) else asset(background)
        frame = imread_unicode(src_path)
        label = os.path.basename(src_path)
    else:
        video = asset(VIDEO_NAME)
        if os.path.isfile(video):
            frame = read_video_frame(video, frame_index)
            label = f"video frame {frame_index}"
        else:
            frame = imread_unicode(asset("court_plate.png"))
            label = "court_plate.png"

    h, w = frame.shape[:2]
    want = tuple(data["source_resolution"])
    if (w, h) != want:
        raise ValueError(
            f"背景帧 {w}x{h} 与标定分辨率 {want[0]}x{want[1]} 不一致。"
            " 单应矩阵和像素系绑死，拉平结果会整体错位。"
        )

    warped = warp_to_topdown(frame, vt, rend)
    imwrite_unicode(WARPED_PNG, _banner(warped, f"WARPED TO TOP-DOWN: {label}"))
    print(f"已写出 -> {WARPED_PNG}")

    # 3) 带刻度网格，用来把地板线读成米
    meas = rend.draw_measure_grid(warped.copy())
    imwrite_unicode(MEASURE_PNG, _banner(
        meas, "MEASURE: warped floor + 1 m grid   X across court, Y from the anchored baseline"))
    print(f"已写出 -> {MEASURE_PNG}")

    # 4) 叠加：拉平的真实地板 + 模型线条。线对齐了就说明矩阵对。
    lines = rend.draw_court(grid=False)
    mask = (np.abs(lines.astype(np.int16) - np.array(C_FLOOR, np.int16)).sum(axis=2) > 40)
    overlay = warped.copy()
    overlay[mask] = (0.25 * overlay[mask] + 0.75 * lines[mask]).astype(np.uint8)
    imwrite_unicode(OVERLAY_PNG, _banner(
        overlay, "OVERLAY: warped floor + court model  (model lines must sit on the real floor lines)"))
    print(f"已写出 -> {OVERLAY_PNG}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="构建球场俯视图")
    ap.add_argument("--calibration", default=OUTPUT_JSON)
    ap.add_argument("--frame", type=int, default=DEFAULT_FRAME, help="用哪一帧拉平")
    ap.add_argument("--ppm", type=float, default=60.0, help="俯视图每米多少像素")
    ap.add_argument("--background", default=None,
                    help="用指定图片代替视频帧（必须是标定分辨率）")
    ap.add_argument("--extent", type=float, nargs=4, metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
                    default=None, help="渲染视野，单位米，可给负数看球场外")
    args = ap.parse_args(argv)
    build(args.calibration, args.frame, args.ppm, args.background,
          tuple(args.extent) if args.extent else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
