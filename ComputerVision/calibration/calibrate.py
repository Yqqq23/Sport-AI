"""解算单应矩阵，写出 calibration/court_3x3.json。

    python -m calibration.calibrate
    python -m calibration.calibrate --allow-todo    # 只用已填坐标的点先跑通流程

⚠ 双边同步过滤 —— 本文件最容易出 bug 的地方
------------------------------------------
13 个点里可能有几个在画面上被遮挡、没标出来（pick_points 里按 k 跳过），
也可能有几个在 court_config.py 里还是 TODO。这两种情况都要剔除，而且
**必须用同一个布尔数组同时索引两边**，保证「画面点」和「球场标准点」
永远一一对应：

    usable = labeled_mask & defined_mask     # 一个掩码
    source = image_points[usable]            # 索引画面点
    target = CONFIG.vertices[usable]         # 索引球场点 —— 同一个掩码

绝对不能写成「先把画面点里 None 的去掉，再另外处理球场点」。
那样两个数组长度也许仍然相等，但第 k 个画面点对上的已经不是第 k 个球场点了 ——
矩阵照样解得出来，只是彻底错的，而且 RANSAC 还会好心地把矛盾点标成 outlier
让你以为「只是有几个点标歪了」。

⚠ 像素系统一
-----------
标注图 real.png 是 2880×752，视频是 1920×1080。矩阵在**视频像素系**里解，
所以 json 里的 source_resolution 就是 1920×1080，下游对视频帧投影时
这个校验才是真的在校验。换算关系见 pixel_frames.py。
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np

from . import pixel_frames as pf
from .court_config import CONFIG, NUM_POINTS, POINT_NAMES
from .homography import RANSAC_REPROJ_THRESHOLD_PX, ViewTransformer
from .io_utils import calib

IMAGE_POINTS_JSON = calib("image_points.json")
OUTPUT_JSON = calib("court_3x3.json")


def load_image_points(path: str = IMAGE_POINTS_JSON) -> dict:
    """读 pick_points 存的 json，做完整性校验。"""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"找不到 {path}\n"
            "  请先跑取点工具：python -m calibration.pick_points"
        )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for key in ("source_image", "source_resolution", "pixel_frame", "points"):
        if key not in data:
            raise ValueError(f"{path} 缺少字段 '{key}'，文件可能是手改坏了")

    pts = data["points"]
    if len(pts) != NUM_POINTS:
        raise ValueError(
            f"{path} 里有 {len(pts)} 个点，参考图是 {NUM_POINTS} 个。"
            " 数量对不上就意味着编号已经错位，必须重新取点。"
        )
    for i, p in enumerate(pts):
        if p.get("id") != i + 1:
            raise ValueError(
                f"{path} 第 {i} 项的 id 是 {p.get('id')}，应为 {i + 1}。"
                " 点的顺序被打乱了，编号会错位。"
            )
        if p.get("name") != POINT_NAMES[i]:
            raise ValueError(
                f"{path} 第 {i + 1} 号点名字是 {p.get('name')!r}，"
                f"court_config 里是 {POINT_NAMES[i]!r}。两边定义不一致。"
            )
    return data


def build_arrays(data: dict):
    """从 json 构造 (13,2) 的画面点数组（视频像素系）和 labeled 掩码。

    未标注的点填 NaN，保持长度 13，这样掩码索引才对得上球场点。
    """
    raw = np.full((NUM_POINTS, 2), np.nan, dtype=np.float64)
    labeled = np.zeros(NUM_POINTS, dtype=bool)
    for i, p in enumerate(data["points"]):
        if p.get("labeled") and p.get("x") is not None and p.get("y") is not None:
            raw[i] = (p["x"], p["y"])
            labeled[i] = True

    # 换算到视频像素系。NaN 行会保持 NaN，不影响。
    frame = data["pixel_frame"]
    video_px = np.full_like(raw, np.nan)
    if labeled.any():
        video_px[labeled] = pf.to_video_pixels(raw[labeled], frame)
    return raw, video_px, labeled


def calibrate(image_points_path: str = IMAGE_POINTS_JSON,
              output_path: str = OUTPUT_JSON,
              allow_todo: bool = False,
              ransac_threshold_px: float = RANSAC_REPROJ_THRESHOLD_PX) -> dict:
    data = load_image_points(image_points_path)
    raw_px, video_px, labeled_mask = build_arrays(data)

    if not allow_todo:
        CONFIG.require_all_defined()

    defined_mask = CONFIG.defined_mask

    # ---- 双边同步过滤：一个掩码，同时索引两边 ----------------------------
    usable_mask = labeled_mask & defined_mask
    source = video_px[usable_mask]          # 画面点（视频像素）
    target = CONFIG.vertices[usable_mask]   # 球场标准点（米）
    # ---------------------------------------------------------------------

    used_ids = (np.where(usable_mask)[0] + 1).tolist()
    dropped = []
    for i in range(NUM_POINTS):
        if usable_mask[i]:
            continue
        why = []
        if not labeled_mask[i]:
            why.append("画面上未标注")
        if not defined_mask[i]:
            why.append("court_config 里还是 TODO")
        dropped.append((i + 1, POINT_NAMES[i], " / ".join(why)))

    print("=" * 78)
    print(f"画面点   : {data['source_image']} "
          f"({data['source_resolution'][0]}x{data['source_resolution'][1]}, "
          f"frame='{data['pixel_frame']}')")
    print(f"像素换算 : {pf.describe()}")
    print(f"球场配置 : {CONFIG!r}")
    print(f"可用点   : {len(used_ids)}/{NUM_POINTS} -> {used_ids}")
    if dropped:
        print("剔除的点 :")
        for pid, name, why in dropped:
            print(f"    #{pid:<3d} {name:<18} {why}")
    print("-" * 78)

    # ViewTransformer 内部做点数 / 退化检查，并强制走 RANSAC
    vt = ViewTransformer(source, target, ransac_threshold_px=ransac_threshold_px)

    px_err = vt.reprojection_errors_px()
    print(f"RANSAC   : 阈值 {ransac_threshold_px} px（真·像素，见 homography.py 顶部说明），"
          f"内点 {vt.num_inliers}/{len(source)}")
    print(f"重投影   : 像素误差 中位数 {np.median(px_err):.2f} px / "
          f"最大 {px_err.max():.2f} px")
    if vt.outlier_indices:
        out_ids = [used_ids[k] for k in vt.outlier_indices]
        print(f"outlier  : 点号 {out_ids}  <- 这几个多半标歪了")
    if vt.horizon_crosses(CONFIG.outline_m()):
        print("⚠ 警告   : 球场区域跨越了地平线，这在物理上不可能 —— 矩阵是错的，"
              "不只是精度差。")
    print("-" * 78)

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "court": {
            "standard": "FIBA 3x3",
            "width_mm": CONFIG.width_mm,
            "length_mm": CONFIG.length_mm,
            "units": "metres",
        },
        # 矩阵所在的像素系 —— 下游对视频帧投影时必须校验这个分辨率
        "source_image": data["source_image"],
        "source_resolution": list(pf.VIDEO_RESOLUTION),
        "pixel_frame": "video",
        "annotation_image_resolution": data["source_resolution"],
        "annotation_pixel_frame": data["pixel_frame"],
        "still_to_video": {
            "matrix_2x3": pf.STILL_TO_VIDEO.tolist(),
            "note": pf.describe(),
        },
        "ransac": {
            "method": "cv2.RANSAC",
            "reproj_threshold_px": ransac_threshold_px,
            "fit_direction": "court_m -> image_px (阈值单位才是像素)",
            "num_inliers": vt.num_inliers,
            "num_points": int(len(source)),
            "reproj_px_median": round(float(np.median(px_err)), 3),
            "reproj_px_max": round(float(px_err.max()), 3),
        },
        "horizon_crosses_court": vt.horizon_crosses(CONFIG.outline_m()),
        "homography": vt.m.tolist(),                 # 视频像素 -> 球场米
        "homography_inverse": vt.inverse_matrix.tolist(),
        "point_ids_used": used_ids,
        "inlier_mask": vt.inlier_mask.astype(int).tolist(),
        "outlier_point_ids": [used_ids[k] for k in vt.outlier_indices],
        "dropped_points": [
            {"id": pid, "name": name, "reason": why} for pid, name, why in dropped
        ],
        "points": [
            {
                "id": i + 1,
                "name": POINT_NAMES[i],
                "image_px_annotation": None if not labeled_mask[i] else
                    [round(float(raw_px[i, 0]), 2), round(float(raw_px[i, 1]), 2)],
                "image_px_video": None if not labeled_mask[i] else
                    [round(float(video_px[i, 0]), 2), round(float(video_px[i, 1]), 2)],
                "court_m": None if not defined_mask[i] else
                    [float(CONFIG.vertices[i, 0]), float(CONFIG.vertices[i, 1])],
                "used": bool(usable_mask[i]),
            }
            for i in range(NUM_POINTS)
        ],
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"已写出 -> {output_path}")
    print("下一步 : python -m calibration.validate")
    print("=" * 78)
    return payload


def load_calibration(path: str = OUTPUT_JSON) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"找不到 {path}\n  请先跑：python -m calibration.calibrate"
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def transformer_from_calibration(data: dict) -> ViewTransformer:
    """从 json 重建 ViewTransformer（跳过重新拟合，直接用存下来的矩阵）。"""
    vt = ViewTransformer.__new__(ViewTransformer)
    vt.m = np.array(data["homography"], dtype=np.float64)
    inv = np.array(data["homography_inverse"], dtype=np.float64)
    vt._inv = inv / inv[2, 2]
    vt.inlier_mask = np.array(data["inlier_mask"], dtype=bool)
    vt.source = np.array(
        [p["image_px_video"] for p in data["points"] if p["used"]], dtype=np.float32)
    vt.target = np.array(
        [p["court_m"] for p in data["points"] if p["used"]], dtype=np.float32)
    vt.ransac_threshold_px = float(data["ransac"]["reproj_threshold_px"])
    return vt


def assert_frame_resolution(data: dict, width: int, height: int) -> None:
    """对视频帧投影前的分辨率校验。不一致就炸，绝不静默算出错误结果。"""
    want = tuple(data["source_resolution"])
    if (width, height) != tuple(want):
        raise ValueError(
            f"分辨率不匹配：标定矩阵是在 {want[0]}x{want[1]} 上解的，"
            f"而当前帧是 {width}x{height}。\n"
            "  单应矩阵和像素系绑死，直接用会得到系统性偏移的错误坐标。\n"
            "  要么把帧缩放回标定分辨率，要么重新标定。"
        )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="解算 3x3 球场单应矩阵")
    ap.add_argument("--points", default=IMAGE_POINTS_JSON)
    ap.add_argument("--out", default=OUTPUT_JSON)
    ap.add_argument("--allow-todo", action="store_true",
                    help="允许 court_config.py 里还有 TODO，只用已填坐标的点解算")
    ap.add_argument("--ransac-threshold-px", type=float,
                    default=RANSAC_REPROJ_THRESHOLD_PX,
                    help="RANSAC 内外点阈值，单位像素（默认 5.0）")
    args = ap.parse_args(argv)
    calibrate(args.points, args.out, args.allow_todo, args.ransac_threshold_px)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
