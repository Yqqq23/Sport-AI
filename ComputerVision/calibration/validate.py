"""标定验收。这是判断矩阵能不能用的**唯一**依据。

    python -m calibration.validate

输出
----
1. 每个点的误差（米）、平均误差、最大误差
2. RANSAC 判为 outlier 的点编号（这些通常就是点歪了的）
3. 判定：平均 < 0.2 m 可用；> 1 m 提示「疑似编号错位」
4. calibration/validation_projection.png
       左：标注图 + 13 个点   右：俯视球场 + 投影结果 vs 标准坐标
5. calibration/validation_reverse.png
       把俯视图上的球场边线和 1 米网格用**逆矩阵**投回画面，叠在球场上。
       这个比看数字直观得多 —— 网格线跟地面实际线条对不齐，矩阵就是错的。
"""

from __future__ import annotations

import argparse
import os

import cv2
import numpy as np

from . import pixel_frames as pf
from .calibrate import (OUTPUT_JSON, load_calibration, transformer_from_calibration)
from .court_config import CONFIG, NUM_POINTS, POINT_NAMES
from .io_utils import asset, calib, imread_unicode, imwrite_unicode, read_video_frame

PROJECTION_PNG = calib("validation_projection.png")
REVERSE_PNG = calib("validation_reverse.png")

VIDEO_NAME = "3 v 3 Basketball Game Side A 24022026.mp4"
DEFAULT_FRAME = 6000

# 判定阈值（米）
GOOD_MEAN_M = 0.20
BAD_MEAN_M = 1.00

PPM = 80                 # 俯视图每米多少像素
MARGIN = 80

C_STD = (200, 200, 200)
C_PROJ = (60, 220, 60)
C_OUT = (60, 60, 255)
C_ERR = (0, 165, 255)
C_LINE = (80, 255, 255)


# ---------------------------------------------------------------------------
# 误差
# ---------------------------------------------------------------------------
def compute_errors(data: dict):
    """返回 (ids, names, projected_m, standard_m, errors_m, inlier_flags)。"""
    vt = transformer_from_calibration(data)
    used = [p for p in data["points"] if p["used"]]
    src = np.array([p["image_px_video"] for p in used], dtype=np.float32)
    std = np.array([p["court_m"] for p in used], dtype=np.float32)
    proj = vt.transform_points(src)
    err = np.linalg.norm(proj - std, axis=1)
    ids = [p["id"] for p in used]
    names = [p["name"] for p in used]
    inl = np.array(data["inlier_mask"], dtype=bool)
    return ids, names, proj, std, err, inl


def orientation_report(data: dict) -> list[str]:
    """报告球场四角落在画面的哪一侧。

    ⚠ 为什么不是「自动检测镜像」
    ---------------------------
    单应变换本身包含镜像和旋转。如果你把 court_config.py 里的坐标整体做了
    左右镜像 / 前后颠倒 / 旋转 180°，解出来的仍然是一个**完美吻合**的矩阵，
    平均误差照样是 0.0001 m。这一点已被冒烟测试实测确认。

    也就是说：**整体镜像在数值上是不可检测的**，残差永远看不出来。
    能发现它的只有两样东西 ——
      1. 下面这份「哪个角在画面哪一侧」的对照，你自己判断合不合理
      2. validation_reverse.png 那张反向叠加图，用肉眼看

    残差能抓到的是「编号错位」（比如整体错开一位），因为那种情况下
    对应关系互相矛盾，不存在一致的模型。镜像不矛盾，所以抓不到。
    """
    vt = transformer_from_calibration(data)
    w_m = CONFIG.width_mm / 1000.0
    l_m = CONFIG.length_mm / 1000.0
    corners = {
        "#12 corner_bl (底线左角)": (0.0, 0.0),
        "#13 corner_br (底线右角)": (w_m, 0.0),
        "#1  corner_tl (中线左端)": (0.0, l_m),
        "#2  corner_tr (中线右端)": (w_m, l_m),
    }
    img_w, img_h = data["source_resolution"]
    lines = []
    for label, xy in corners.items():
        p = vt.inverse_transform_points(np.array([xy], dtype=np.float32))[0]
        if not np.isfinite(p).all():
            lines.append(f"{label} -> 投影发散（在地平线外）")
            continue
        h = "画面左" if p[0] < img_w / 2 else "画面右"
        v = "上(远)" if p[1] < img_h / 2 else "下(近)"
        inside = "" if (0 <= p[0] <= img_w and 0 <= p[1] <= img_h) else "  [画面外]"
        lines.append(f"{label} -> ({p[0]:8.1f}, {p[1]:8.1f}) px  {h}/{v}{inside}")
    return lines


def report(data: dict) -> float:
    ids, names, proj, std, err, inl = compute_errors(data)
    mean_err, max_err = float(err.mean()), float(err.max())

    print("=" * 78)
    print("标定验收报告")
    print("=" * 78)
    print(f"矩阵像素系 : {data['pixel_frame']} "
          f"{data['source_resolution'][0]}x{data['source_resolution'][1]}")
    print(f"标注来源   : {data['source_image']} "
          f"({data['annotation_image_resolution'][0]}x"
          f"{data['annotation_image_resolution'][1]}, "
          f"frame='{data['annotation_pixel_frame']}')")
    print(f"生成时间   : {data['created_utc']}")
    print(f"球场标准   : {data['court']['standard']} "
          f"{data['court']['width_mm'] / 1000:g}m x {data['court']['length_mm'] / 1000:g}m")
    print("-" * 78)
    print(f"{'id':>3}  {'name':<20}{'projected (m)':>20}{'standard (m)':>20}"
          f"{'err (m)':>10}  RANSAC")
    for k in range(len(ids)):
        flag = "inlier " if inl[k] else "OUTLIER"
        print(f"{ids[k]:>3}  {names[k]:<20}"
              f"({proj[k, 0]:7.3f},{proj[k, 1]:7.3f}){' ':>2}"
              f"({std[k, 0]:7.3f},{std[k, 1]:7.3f}){' ':>2}"
              f"{err[k]:>10.4f}  {flag}")
    print("-" * 78)
    print(f"参与解算   : {len(ids)}/{NUM_POINTS} 个点")
    print(f"平均误差   : {mean_err:.4f} m")
    print(f"最大误差   : {max_err:.4f} m   (点号 #{ids[int(np.argmax(err))]} "
          f"{names[int(np.argmax(err))]})")
    print(f"中位误差   : {float(np.median(err)):.4f} m")

    out_ids = data.get("outlier_point_ids", [])
    if out_ids:
        print(f"outlier 点 : {out_ids}")
        for pid in out_ids:
            k = ids.index(pid)
            print(f"             #{pid} {names[k]}  误差 {err[k]:.4f} m  <- 这个点多半点歪了")
    else:
        print("outlier 点 : 无")

    if data.get("dropped_points"):
        print("未参与解算 :")
        for d in data["dropped_points"]:
            print(f"             #{d['id']} {d['name']}  ({d['reason']})")

    # 重投影误差（像素空间）—— RANSAC 的判据本身
    vt = transformer_from_calibration(data)
    px = vt.reprojection_errors_px()
    print(f"重投影误差 : 中位 {np.median(px):.2f} px / 最大 {px.max():.2f} px "
          f"(RANSAC 阈值 {vt.ransac_threshold_px:g} px)")

    ref = data.get("refined")
    if ref:
        print("-" * 78)
        print("线拟合   : 跑过 refine，**这才是主判据** —— 手点的点各有误差，"
              "地板上画出来的线没有")
        print(f"           {ref['line_pixels_used']} 个球场线像素落在模型线 0.4 m 内，"
              f"中位偏差 {ref['median_dev_m']:.4f} m / p90 {ref['p90_dev_m']:.4f} m")
        if ref["median_dev_m"] < 0.12:
            print("           ✅ 这个量级相当于一条画线的宽度，矩阵是对的")
        elif ref["median_dev_m"] < 0.30:
            print("           ⚠ 偏大，看 topdown_overlay.png 确认")
        else:
            print("           ❌ 模型线没压住地板线")

    if data.get("horizon_crosses_court"):
        print("⚠ 物理检查 : 球场区域跨越地平线 —— 矩阵在物理上不可能成立，"
              "不是精度问题，是彻底错了。")

    print("-" * 78)
    print("方位对照（整体镜像在数值上不可检测，只能靠这里和反向图肉眼判断）：")
    for line in orientation_report(data):
        print("       " + line)

    print("-" * 78)
    if mean_err < GOOD_MEAN_M:
        print(f"判定 : ✅ 可用（平均误差 {mean_err:.4f} m < {GOOD_MEAN_M} m）")
        print("       但仍然要看一眼 validation_reverse.png —— 数字好看"
              "不排除整体镜像/前后颠倒。")
    elif mean_err <= BAD_MEAN_M:
        print(f"判定 : ⚠ 偏大（{GOOD_MEAN_M} m ≤ {mean_err:.4f} m ≤ {BAD_MEAN_M} m）")
        print("       多半是有点点歪了。先看 outlier 点号，重点回去补点那几个。")
    else:
        line_ok = bool(ref) and ref["median_dev_m"] < 0.12
        if line_ok:
            print(f"判定 : ⚠ 少数点认错了特征（平均 {mean_err:.4f} m 被 outlier 拉高），"
                  "但矩阵本身是对的")
            print(f"       线拟合中位偏差只有 {ref['median_dev_m']:.4f} m —— "
                  "地板上的线压住了，说明矩阵没问题。")
            print("       平均误差高是因为有几个点**点在了别的线上**（不是点歪，是认错）。")
            print("       把它们在 image_points.json 里设成 labeled:false，"
                  "或者重点，误差就正常了。")
            print("       上面 OUTLIER 那几个就是。")
        else:
            print(f"判定 : ❌ 疑似编号错位（平均误差 {mean_err:.4f} m > {BAD_MEAN_M} m）")
            print("       这个量级的误差不是点歪，是对应关系整体错了。检查两件事：")
            print("       1. 取点顺序是否和 pick_guide.png 一致")
            print("       2. court_config.py 里的坐标有没有填串行（错开一位最常见）")
            print("       注意：单纯的左右镜像/前后颠倒**不会**产生这么大的误差，")
            print("             那种情况误差反而接近 0，只能靠反向叠加图发现。")
    print("=" * 78)
    return mean_err


# ---------------------------------------------------------------------------
# 可视化：正向投影对照
# ---------------------------------------------------------------------------
def _topdown_canvas():
    w = int(CONFIG.width_mm / 1000.0 * PPM) + 2 * MARGIN
    h = int(CONFIG.length_mm / 1000.0 * PPM) + 2 * MARGIN
    return np.full((h, w, 3), 35, dtype=np.uint8)


def _m_to_px(pts_m: np.ndarray, canvas_h: int) -> np.ndarray:
    """球场米 → 俯视图像素。Y 轴翻转，让底线画在下方，和参考图方向一致。"""
    pts = np.asarray(pts_m, dtype=np.float64).reshape(-1, 2)
    x = pts[:, 0] * PPM + MARGIN
    y = canvas_h - MARGIN - pts[:, 1] * PPM
    return np.stack([x, y], 1)


def draw_projection_figure(data: dict, out_path: str) -> None:
    ids, names, proj, std, err, inl = compute_errors(data)

    # ---- 右：俯视球场 ----
    top = _topdown_canvas()
    H, W = top.shape[:2]
    w_m, l_m = CONFIG.width_mm / 1000.0, CONFIG.length_mm / 1000.0

    for gx in range(int(w_m) + 1):                      # 1 米网格
        p = _m_to_px([[gx, 0], [gx, l_m]], H).astype(int)
        cv2.line(top, tuple(p[0]), tuple(p[1]), (60, 60, 60), 1)
    for gy in range(int(l_m) + 1):
        p = _m_to_px([[0, gy], [w_m, gy]], H).astype(int)
        cv2.line(top, tuple(p[0]), tuple(p[1]), (60, 60, 60), 1)
    outline = _m_to_px(CONFIG.outline_m(), H).astype(np.int32)
    cv2.polylines(top, [outline], True, (200, 200, 200), 2)
    cv2.putText(top, "BASELINE (basket side)", (MARGIN, H - MARGIN + 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_STD, 1, cv2.LINE_AA)
    cv2.putText(top, "HALF-COURT LINE", (MARGIN, MARGIN - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_STD, 1, cv2.LINE_AA)

    std_px = _m_to_px(std, H)
    proj_px = _m_to_px(proj, H)
    for k in range(len(ids)):
        s = tuple(std_px[k].astype(int))
        p = tuple(proj_px[k].astype(int))
        cv2.line(top, s, p, C_ERR, 1, cv2.LINE_AA)               # 误差向量
        cv2.circle(top, s, 8, C_STD, 1, cv2.LINE_AA)             # 标准坐标：空心
        cv2.circle(top, p, 5, C_OUT if not inl[k] else C_PROJ, -1, cv2.LINE_AA)
        cv2.putText(top, f"{ids[k]}", (p[0] + 9, p[1] - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    C_OUT if not inl[k] else C_PROJ, 2, cv2.LINE_AA)

    cv2.putText(top, "TOP-DOWN: hollow=standard  filled=projected  red=RANSAC outlier",
                (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

    # ---- 左：标注图 + 点 ----
    ann_path = asset(data["source_image"])
    left = imread_unicode(ann_path)
    ann_pts = {p["id"]: p["image_px_annotation"] for p in data["points"]}
    for pid in ids:
        xy = ann_pts[pid]
        if xy is None:
            continue
        k = ids.index(pid)
        c = C_OUT if not inl[k] else C_PROJ
        pt = (int(round(xy[0])), int(round(xy[1])))
        cv2.circle(left, pt, 16, c, 3, cv2.LINE_AA)
        cv2.drawMarker(left, pt, c, cv2.MARKER_CROSS, 26, 2)
        cv2.putText(left, str(pid), (pt[0] + 20, pt[1] - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(left, str(pid), (pt[0] + 20, pt[1] - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, c, 3, cv2.LINE_AA)
    cv2.putText(left, f"SOURCE: {data['source_image']}", (16, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)

    # ---- 拼接 ----
    # 标注图很宽（2880x752），俯视图接近方形。直接拉成等高会把俯视图压扁，
    # 所以先把标注图限宽，再用留白补齐高度、垂直居中，保持两边各自的比例。
    LEFT_MAX_W = 2200
    if left.shape[1] > LEFT_MAX_W:
        s = LEFT_MAX_W / left.shape[1]
        left = cv2.resize(left, (LEFT_MAX_W, int(round(left.shape[0] * s))),
                          interpolation=cv2.INTER_AREA)

    fig_h = max(left.shape[0], top.shape[0])

    def _pad(img):
        pad = fig_h - img.shape[0]
        if pad <= 0:
            return img
        t, b = pad // 2, pad - pad // 2
        return cv2.copyMakeBorder(img, t, b, 0, 0, cv2.BORDER_CONSTANT, value=(15, 15, 15))

    fig = cv2.hconcat([_pad(left), np.full((fig_h, 12, 3), 15, np.uint8), _pad(top)])

    banner = np.full((46, fig.shape[1], 3), 15, np.uint8)
    cv2.putText(banner, f"mean err {err.mean():.4f} m   max {err.max():.4f} m   "
                        f"inliers {int(inl.sum())}/{len(ids)}",
                (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    imwrite_unicode(out_path, cv2.vconcat([banner, fig]))
    print(f"已写出 -> {out_path}")


# ---------------------------------------------------------------------------
# 可视化：反向边线叠加
# ---------------------------------------------------------------------------
def _draw_projected_polyline(canvas, vt, pts_m, color, thickness=2, samples=80):
    """把球场上的一条线密集采样后逆投影画到画面上。

    必须分段画：地平线以外的点会被投到画面外的极端坐标甚至跑到画面另一侧，
    直接连一条直线会在图上拉出一道横贯全屏的假线。
    """
    pts_m = np.asarray(pts_m, dtype=np.float32)
    dense = []
    for a, b in zip(pts_m[:-1], pts_m[1:]):
        t = np.linspace(0, 1, samples)[:, None]
        dense.append(a[None, :] * (1 - t) + b[None, :] * t)
    dense = np.vstack(dense)
    px = vt.inverse_transform_points(dense)

    h, w = canvas.shape[:2]
    lim = 4 * max(h, w)
    ok = np.isfinite(px).all(axis=1) & (np.abs(px[:, 0]) < lim) & (np.abs(px[:, 1]) < lim)
    for i in range(len(px) - 1):
        if ok[i] and ok[i + 1]:
            cv2.line(canvas, tuple(px[i].astype(int)), tuple(px[i + 1].astype(int)),
                     color, thickness, cv2.LINE_AA)


def draw_reverse_figure(data: dict, out_path: str, frame_index: int) -> None:
    vt = transformer_from_calibration(data)

    # 背景优先用真实视频帧（矩阵就是在视频像素系里解的）
    video_path = asset(VIDEO_NAME)
    if os.path.isfile(video_path):
        bg = read_video_frame(video_path, frame_index)
        src_label = f"video frame {frame_index}"
    else:
        # 退路：把标注图按实测相似变换摆进视频像素系
        still = imread_unicode(asset(data["source_image"]))
        bg = cv2.warpAffine(still, pf.STILL_TO_VIDEO.astype(np.float32),
                            pf.VIDEO_RESOLUTION, borderValue=(30, 30, 30))
        src_label = f"{data['source_image']} warped into video pixel frame"

    h, w = bg.shape[:2]
    want = tuple(data["source_resolution"])
    if (w, h) != want:
        raise ValueError(
            f"背景图分辨率 {w}x{h} 与标定分辨率 {want[0]}x{want[1]} 不一致，"
            "叠加会错位。"
        )

    w_m, l_m = CONFIG.width_mm / 1000.0, CONFIG.length_mm / 1000.0

    # 1 米网格 —— 判断透视对不对最直观的东西
    for gx in range(int(w_m) + 1):
        _draw_projected_polyline(bg, vt, [[gx, 0], [gx, l_m]], (70, 70, 70), 1)
    for gy in range(int(l_m) + 1):
        _draw_projected_polyline(bg, vt, [[0, gy], [w_m, gy]], (70, 70, 70), 1)

    # 球场外框
    outline = np.vstack([CONFIG.outline_m(), CONFIG.outline_m()[:1]])
    _draw_projected_polyline(bg, vt, outline, C_LINE, 3)

    # 已定义的特征点逆投影回画面
    defined = CONFIG.defined_mask
    for i in range(NUM_POINTS):
        if not defined[i]:
            continue
        p = vt.inverse_transform_points(CONFIG.vertices[i:i + 1])[0]
        if not np.isfinite(p).all() or not (-w < p[0] < 2 * w and -h < p[1] < 2 * h):
            continue
        pt = (int(p[0]), int(p[1]))
        cv2.drawMarker(bg, pt, C_LINE, cv2.MARKER_TILTED_CROSS, 18, 2)
        cv2.putText(bg, str(i + 1), (pt[0] + 10, pt[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(bg, str(i + 1), (pt[0] + 10, pt[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_LINE, 2, cv2.LINE_AA)

    cv2.rectangle(bg, (0, 0), (w, 46), (15, 15, 15), -1)
    cv2.putText(bg, f"REVERSE CHECK  |  {src_label}  |  court outline + 1m grid via H^-1",
                (14, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    imwrite_unicode(out_path, bg)
    print(f"已写出 -> {out_path}")
    print("       用肉眼看：黄色外框和灰色 1 米网格有没有和地面上的实际线条对齐。"
          "网格歪了，数字再好看也没用。")


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="标定验收：误差报告 + 两张可视化图")
    ap.add_argument("--calibration", default=OUTPUT_JSON)
    ap.add_argument("--frame", type=int, default=DEFAULT_FRAME,
                    help="反向检查用哪一帧做背景（默认 6000）")
    ap.add_argument("--no-figures", action="store_true", help="只打报告，不出图")
    args = ap.parse_args(argv)

    data = load_calibration(args.calibration)
    mean_err = report(data)

    if not args.no_figures:
        draw_projection_figure(data, PROJECTION_PNG)
        draw_reverse_figure(data, REVERSE_PNG, args.frame)

    return 0 if mean_err < GOOD_MEAN_M else 2


if __name__ == "__main__":
    raise SystemExit(main())
