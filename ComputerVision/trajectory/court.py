"""把画面像素投影成球场**米**坐标，用本项目 calibration/ 标好的那个矩阵。

为什么不用 basketball-v2 自带的标定
-----------------------------------
那边的 ``configs/court_calibration.json`` 把球场点表示成一张 500×940
俯视画布的**像素**，而且它的 image_points 顺序是乱的
（``corner_tl`` 在 (679,615)、``corner_br`` 在 (849,637)，两个角挨在一起）。
画布像素也没法直接拿来算跑动距离、速度、热区。

本模块用 ``calibration/court_3x3.json``：单位是**米**，经过
    - 8 个点参与解算，平均误差 0.165 m、中位 0.026 m
    - 11219 个球场线像素拟合，中位偏差 0.0635 m
    - 地平线不穿过球场（物理可实现）
的验收。

坐标系
------
    X: 0 → 15 m   沿底线向右
    Y: 0 → 14 m   从底线（篮筐那侧）指向中线

⚠ 球员位置一律取检测框**底边中点**（脚），不是框中心。
   实测同一个人差 2.07 米 —— 足够把人从篮下挪到三分线外。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# 本包和 calibration/ 是同级目录，把项目根挂上去就能 import
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from calibration.calibrate import (  # noqa: E402
    OUTPUT_JSON,
    assert_frame_resolution,
    load_calibration,
    transformer_from_calibration,
)
from calibration.court_config import COURT_LENGTH_MM, COURT_WIDTH_MM  # noqa: E402

COURT_W_M = COURT_WIDTH_MM / 1000.0
COURT_L_M = COURT_LENGTH_MM / 1000.0

#: 判定「投影结果离谱」的边界（米）。球员会跑到界外，所以留一圈余量，
#: 但离场地几十米那种一定是检测框错了或者投影发散。
SANE_MARGIN_M = 6.0


@dataclass
class Projection:
    """一次投影的结果。``in_bounds`` 只是标注，不做裁剪。"""

    x_m: float
    y_m: float
    in_court: bool      # 严格在 15×14 之内
    sane: bool          # 在场地 ± SANE_MARGIN_M 之内，超出说明这个检测不可信


class CourtProjector:
    """一次性加载标定，之后按需投影。矩阵和像素系绑死，所以要校验分辨率。"""

    def __init__(self, calibration_path: str | Path | None = None):
        self.path = Path(calibration_path) if calibration_path else Path(OUTPUT_JSON)
        self.data = load_calibration(self.path)
        self.vt = transformer_from_calibration(self.data)
        self.width, self.height = self.data["source_resolution"]
        if self.data.get("horizon_crosses_court"):
            raise ValueError(
                f"{self.path} 里的矩阵会让球场跨越地平线，物理上不成立，不能用来投影。"
                " 先把标定修好（见 calibration/README.md）。"
            )

    # -- 校验 --------------------------------------------------------------
    def check_frame(self, frame_width: int, frame_height: int) -> None:
        """帧分辨率必须和标定时一致，不一致直接炸，不静默算出错误坐标。"""
        assert_frame_resolution(self.data, frame_width, frame_height)

    # -- 投影 --------------------------------------------------------------
    def project_points(self, points_px) -> np.ndarray:
        """(N,2) 画面像素 → (N,2) 球场米。空数组原样返回。"""
        arr = np.asarray(points_px, dtype=np.float32)
        if arr.size == 0:
            return np.empty((0, 2), dtype=np.float32)
        return self.vt.transform_points(arr.reshape(-1, 2))

    def project_feet(self, boxes_xyxy) -> np.ndarray:
        """(N,4) xyxy 检测框 → (N,2) 球场米，取**底边中点**。"""
        return self.vt.transform_boxes_to_court(boxes_xyxy)

    def project_box(self, bbox_xyxy) -> Projection:
        """单个框 → Projection，带越界 / 离谱标记。"""
        q = self.project_feet(np.asarray([bbox_xyxy], dtype=np.float32))
        if q.shape[0] == 0 or not np.isfinite(q[0]).all():
            return Projection(float("nan"), float("nan"), False, False)
        x, y = float(q[0, 0]), float(q[0, 1])
        in_court = (0.0 <= x <= COURT_W_M) and (0.0 <= y <= COURT_L_M)
        sane = (-SANE_MARGIN_M <= x <= COURT_W_M + SANE_MARGIN_M) and (
            -SANE_MARGIN_M <= y <= COURT_L_M + SANE_MARGIN_M
        )
        return Projection(x, y, in_court, sane)

    # -- 元信息 ------------------------------------------------------------
    def describe(self) -> str:
        r = self.data.get("refined") or {}
        bits = [
            f"calibration = {self.path}",
            f"pixel frame = {self.width}x{self.height}",
            f"court       = {COURT_W_M:g} x {COURT_L_M:g} m",
        ]
        if r:
            bits.append(
                f"line fit    = {r.get('line_pixels_used')} px, "
                f"median {r.get('median_dev_m')} m"
            )
        return "\n".join("  " + b for b in bits)
