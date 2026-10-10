"""用地板上**画出来的球场线**精修单应矩阵。

    python -m calibration.refine
    python -m calibration.refine --dry-run        # 只报告，不写回

为什么需要这一步
----------------
手点的特征点有像素级误差，而且远端一两个点点歪就会把整个远场拽偏。
地板上画出来的线不一样：它是**成千上万个像素**，位置由油漆决定，不会手抖。
拿它去拟合，等于把矩阵按实际画线对齐。

不预先认线
----------
``court_line_distance`` 算的是「到**最近的任意一条**球场线的距离」，
不区分那条线是限制区边线、三分线还是中线。这很关键 —— 这块多用途地板上
叠了好几套球场线，事先判断「这条蓝线是哪条」正是之前反复出错的地方。
只要一个线像素落在某条球场线附近就算对，识别问题就消失了。

做法（ICP 风格）
----------------
    1. 从地板上抠出球场线的像素（这块场地的篮球线是深蓝色的）
    2. 用当前矩阵投影到球场坐标，留下靠近模型任意一条线的那些
    3. 最小化它们到最近模型线的**米制距离**，同时保留已确认的锚点约束
    4. 重新筛选，迭代几轮

模型全部由 court_config.py 的常量推出，改场地改那里即可。
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import cv2
import numpy as np

from .calibrate import OUTPUT_JSON, load_calibration, transformer_from_calibration
from .court_config import (BASKET_DIST_MM, CIRCLE_R_MM, COURT_LENGTH_MM,
                           COURT_WIDTH_MM, FREE_THROW_DIST_MM, PAINT_WIDTH_MM,
                           POINT_NAMES, THREE_INSET_MM, THREE_RADIUS_MM)
from .io_utils import asset, imread_unicode

# 球场线在画面里的颜色范围（HSV）。这块地板的篮球线是深蓝色的。
DEFAULT_HSV_LO = (95, 55, 45)
DEFAULT_HSV_HI = (132, 255, 255)
# 地板以上是墙和天花板，一律不要
DEFAULT_FLOOR_TOP = 600

W_M = COURT_WIDTH_MM / 1000.0
L_M = COURT_LENGTH_MM / 1000.0
BK_M = BASKET_DIST_MM / 1000.0
R_M = THREE_RADIUS_MM / 1000.0
INS_M = THREE_INSET_MM / 1000.0
FT_M = FREE_THROW_DIST_MM / 1000.0
PAINT_HALF_M = PAINT_WIDTH_MM / 2000.0
CIRCLE_R_M = CIRCLE_R_MM / 1000.0
NO_CHARGE_R_M = 1.25
BRK_M = BK_M + float(np.sqrt(R_M ** 2 - (W_M / 2 - INS_M) ** 2))


class RefinementError(RuntimeError):
    """精修失败 —— 线像素太少或拟合发散。"""


# ---------------------------------------------------------------------------
def apply_h(h: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = np.concatenate([pts, np.ones((len(pts), 1))], axis=1) @ h.T
    return p[:, :2] / p[:, 2:3]


def _vseg_dist(x, y, sx, y0, y1):
    """点到竖直线段 (sx, y0)-(sx, y1) 的距离。"""
    return np.hypot(x - sx, y - np.clip(y, y0, y1))


def _hseg_dist(x, y, sy, x0, x1):
    """点到水平线段 (x0, sy)-(x1, sy) 的距离。"""
    return np.hypot(x - np.clip(x, x0, x1), y - sy)


def _circle_dist(x, y, cx, cy, r):
    return np.abs(np.hypot(x - cx, y - cy) - r)


def court_line_distance(court_pts: np.ndarray) -> np.ndarray:
    """球场坐标（米）到**最近的任意一条球场线**的距离（米）。"""
    x, y = court_pts[:, 0], court_pts[:, 1]
    cx = W_M / 2.0
    ph = PAINT_HALF_M
    d = []

    # 外框：底线、中线、两条边线
    d.append(_hseg_dist(x, y, 0.0, 0.0, W_M))
    d.append(_hseg_dist(x, y, L_M, 0.0, W_M))
    d.append(_vseg_dist(x, y, 0.0, 0.0, L_M))
    d.append(_vseg_dist(x, y, W_M, 0.0, L_M))

    # 限制区：两条边线 + 罚球线
    d.append(_vseg_dist(x, y, cx - ph, 0.0, FT_M))
    d.append(_vseg_dist(x, y, cx + ph, 0.0, FT_M))
    d.append(_hseg_dist(x, y, FT_M, cx - ph, cx + ph))

    # 罚球圈；中圈只有场内那半
    d.append(_circle_dist(x, y, cx, FT_M, CIRCLE_R_M))
    d.append(np.where(y <= L_M, _circle_dist(x, y, cx, L_M, CIRCLE_R_M), np.inf))

    # 篮下限制区半圆
    d.append(np.where(y >= BK_M, _circle_dist(x, y, cx, BK_M, NO_CHARGE_R_M), np.inf))

    # 三分线：两条直段 + 圆弧
    d.append(_vseg_dist(x, y, INS_M, 0.0, BRK_M))
    d.append(_vseg_dist(x, y, W_M - INS_M, 0.0, BRK_M))
    ang = np.arctan2(y - BK_M, x - cx)
    a0 = np.arctan2(BRK_M - BK_M, cx - INS_M)
    d.append(np.where((ang >= a0) & (ang <= np.pi - a0),
                      _circle_dist(x, y, cx, BK_M, R_M), np.inf))

    return np.min(np.stack(d, axis=0), axis=0)


def extract_line_pixels(img: np.ndarray, hsv_lo=DEFAULT_HSV_LO, hsv_hi=DEFAULT_HSV_HI,
                        floor_top: int = DEFAULT_FLOOR_TOP) -> np.ndarray:
    """从画面里抠出球场线的像素，返回 (N, 2) 的 xy。"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_lo, hsv_hi)
    mask[:max(floor_top, 0)] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    ys, xs = np.nonzero(mask)
    return np.stack([xs, ys], axis=1).astype(np.float64)


def anchor_correspondences(data: dict):
    """从标定结果里取出参与解算的对应点，作为精修时的软约束。

    只用 RANSAC 判为 inlier 的那些 —— outlier 本来就是标歪的，
    再把它们当约束会把精修带偏。
    """
    src, dst, ids = [], [], []
    outliers = set(data.get("outlier_point_ids", []))
    for p in data["points"]:
        if p["used"] and p["image_px_video"] and p["court_m"] and p["id"] not in outliers:
            src.append(p["image_px_video"])
            dst.append(p["court_m"])
            ids.append(p["id"])
    return np.array(src, np.float64), np.array(dst, np.float64), ids


def refine(h0: np.ndarray, pixels: np.ndarray, anchor_src: np.ndarray,
           anchor_dst: np.ndarray, iterations: int = 8,
           gate_first: float = 2.0, gate: float = 0.4,
           anchor_weight: float = 2.0):
    """ICP 精修。返回 (H, 统计信息)。"""
    from scipy.optimize import least_squares

    def pack(h):
        return (h / h[2, 2]).ravel()[:8]

    def unpack(v):
        return np.array([[v[0], v[1], v[2]], [v[3], v[4], v[5]], [v[6], v[7], 1.0]])

    h = h0 / h0[2, 2]
    history = []
    for it in range(iterations):
        dist = court_line_distance(apply_h(h, pixels))
        sel = pixels[dist < (gate_first if it == 0 else gate)]
        if len(sel) < 100:
            raise RefinementError(
                f"第 {it} 轮只剩 {len(sel)} 个线像素（<100），无法精修。\n"
                "  多半是颜色阈值没抠到球场线，或初始矩阵偏得太远。\n"
                "  先把 calibrate 的结果修到物理上成立，再跑这一步。"
            )

        def residual(v):
            hh = unpack(v)
            r_line = court_line_distance(apply_h(hh, sel)) / np.sqrt(len(sel))
            r_anchor = (apply_h(hh, anchor_src) - anchor_dst).ravel()
            return np.concatenate([r_line, anchor_weight * r_anchor / np.sqrt(len(anchor_src))])

        out = least_squares(residual, pack(h), method="lm", max_nfev=20000)
        h = unpack(out.x)
        dd = court_line_distance(apply_h(h, sel))
        history.append({
            "iter": it,
            "pixels": int(len(sel)),
            "median_m": round(float(np.median(dd)), 4),
            "p90_m": round(float(np.percentile(dd, 90)), 4),
        })

    if not np.isfinite(h).all() or abs(np.linalg.det(h)) < 1e-12:
        raise RefinementError("精修后的矩阵是奇异的或含 NaN，拟合发散了。")
    return h, history


def horizon_crosses_court(h: np.ndarray) -> bool:
    """球场区域是否跨越地平线（物理上不可能）。"""
    inv = np.linalg.inv(h)
    corners = np.array([[0.0, 0.0], [W_M, 0.0], [W_M, L_M], [0.0, L_M],
                        [W_M / 2, L_M / 2]])
    w = corners @ inv[2, :2] + inv[2, 2]
    return bool(w.min() * w.max() <= 0)


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="用地板上画出来的球场线精修单应矩阵")
    ap.add_argument("--calibration", default=OUTPUT_JSON)
    ap.add_argument("--image", default="court_plate.png",
                    help="抠线用的底图，必须是标定分辨率（默认 court_plate.png）")
    ap.add_argument("--iterations", type=int, default=8)
    ap.add_argument("--floor-top", type=int, default=DEFAULT_FLOOR_TOP,
                    help="这一行以上全部忽略（墙和天花板）")
    ap.add_argument("--dry-run", action="store_true", help="只报告，不写回 json")
    args = ap.parse_args(argv)

    data = load_calibration(args.calibration)
    vt0 = transformer_from_calibration(data)

    path = args.image if os.path.isabs(args.image) else asset(args.image)
    img = imread_unicode(path)
    h_img, w_img = img.shape[:2]
    want = tuple(data["source_resolution"])
    if (w_img, h_img) != want:
        raise ValueError(
            f"底图 {w_img}x{h_img} 与标定分辨率 {want[0]}x{want[1]} 不一致，"
            "抠出来的线像素坐标系对不上。"
        )

    pixels = extract_line_pixels(img, floor_top=args.floor_top)
    a_src, a_dst, a_ids = anchor_correspondences(data)
    if len(a_src) < 4:
        raise RefinementError(
            f"只有 {len(a_src)} 个 inlier 锚点（<4），精修没有足够的约束。"
            " 先把取点修好。"
        )

    print("=" * 78)
    print(f"底图     : {os.path.basename(path)} ({w_img}x{h_img})")
    print(f"线像素   : {len(pixels)}")
    print(f"锚点     : {a_ids}  （只用 RANSAC inlier，outlier 不参与）")
    print("-" * 78)

    before = court_line_distance(apply_h(vt0.m, pixels))
    n_b = int((before < 0.4).sum())
    print(f"精修前   : {n_b} 个像素落在球场线 0.4 m 内"
          + (f"，中位偏差 {np.median(before[before < 0.4]):.4f} m" if n_b else ""))

    h, history = refine(vt0.m, pixels, a_src, a_dst, iterations=args.iterations)
    for rec in history:
        print(f"  iter {rec['iter']}: {rec['pixels']:6d} px   "
              f"中位 {rec['median_m']:.4f} m   p90 {rec['p90_m']:.4f} m")

    after = court_line_distance(apply_h(h, pixels))
    n_a = int((after < 0.4).sum())
    print("-" * 78)
    print(f"精修后   : {n_a} 个像素落在 0.4 m 内，"
          f"中位偏差 {np.median(after[after < 0.4]):.4f} m")

    crosses = horizon_crosses_court(h)
    print(f"物理检查 : 球场跨越地平线 = {crosses}"
          + ("   ⚠ 矩阵不成立" if crosses else "   ✅"))

    anchor_err = np.linalg.norm(apply_h(h, a_src) - a_dst, axis=1)
    print("锚点残差 :")
    for pid, e in zip(a_ids, anchor_err):
        print(f"    #{pid:<3d} {POINT_NAMES[pid - 1]:<20} {e:.4f} m")

    ip_path = os.path.join(os.path.dirname(os.path.abspath(args.calibration)),
                           "image_points.json")
    if os.path.isfile(ip_path):
        ip = json.load(open(ip_path, encoding="utf-8"))
        print("13 个点在精修后矩阵下的球场坐标：")
        for p in ip["points"]:
            if not p["labeled"]:
                print(f"    #{p['id']:<3d} {POINT_NAMES[p['id'] - 1]:<20} (未标注)")
                continue
            q = apply_h(h, np.float64([[p["x"], p["y"]]]))[0]
            print(f"    #{p['id']:<3d} {POINT_NAMES[p['id'] - 1]:<20} "
                  f"({q[0]:8.3f}, {q[1]:8.3f}) m")
    print("=" * 78)

    if args.dry_run:
        print("--dry-run：未写回。")
        return 0

    inv = np.linalg.inv(h)
    data["homography"] = (h / h[2, 2]).tolist()
    data["homography_inverse"] = (inv / inv[2, 2]).tolist()
    data["horizon_crosses_court"] = crosses
    data["refined"] = {
        "method": "ICP against every painted court line (nearest-line distance)",
        "image": os.path.basename(path),
        "line_pixels_used": history[-1]["pixels"],
        "median_dev_m": history[-1]["median_m"],
        "p90_dev_m": history[-1]["p90_m"],
        "anchor_ids": a_ids,
        "anchor_residual_m": [round(float(e), 4) for e in anchor_err],
        "history": history,
        "refined_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(args.calibration, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"已写回 -> {args.calibration}")
    print("下一步 : python -m calibration.topdown --background court_plate.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
