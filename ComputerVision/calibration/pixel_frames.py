"""像素坐标系之间的换算：标注图 (real.png) ↔ 视频帧。

背景
----
RA 标注用的 ``real.png`` 是 **2880 × 752**，而比赛视频是 **1920 × 1080**。
两者不是同一个像素系，直接把标注图上点出来的坐标拿去对视频帧投影，
结果会整体偏 1.5 倍 —— 这是这类项目最典型的「全盘作废」错误。

实测结论
--------
用 SIFT + RANSAC 把 real.png 和视频的第 600 / 6000 / 13000 / 24000 帧
分别配准（这四帧横跨整段 878 秒）：

    frame   600: scale=0.66525  rot=0.0744°  t=(2.74, 555.31)
    frame  6000: scale=0.66541  rot=0.0731°  t=(2.52, 555.22)
    frame 13000: scale=0.66519  rot=0.0745°  t=(2.79, 555.30)
    frame 24000: scale=0.66534  rot=0.0845°  t=(2.92, 554.96)

    合并 339 对内点 → scale=0.665324  rot=0.07892°  t=(2.757, 555.159)
    残差：中位数 0.573 px，p95 1.355 px，最大 1.778 px

两点结论：

1. 四帧参数一致到小数点后三位 ⇒ 整段视频相机确实没动，
   「只算一个单应矩阵」的前提成立。
2. 标注图就是这台相机 —— 它是视频画面 y∈[555, 1057] 那一条
   放大约 1.503 倍的截图，**纯相似变换，没有任何透视差异**。
   所以标注可以迁移，但必须先做下面这个换算。

亚像素级的残差（中位数 0.57 px）远小于手点的精度（通常 ±3~5 px），
所以这一步不是精度瓶颈。

（另外：同目录的 ``realcourt1.png`` **不是**这台相机 —— 与视频配准只有
20/254 内点、旋转差 -3.5°、残差中位数 74 px。它上面的任何标注都不能用。）
"""

from __future__ import annotations

import cv2
import numpy as np

# 标注图 real.png 的分辨率
STILL_RESOLUTION = (2880, 752)

# 比赛视频的分辨率
VIDEO_RESOLUTION = (1920, 1080)

# 实测的相似变换参数：real.png 像素 → 视频帧像素
_SCALE = 0.665324
_ROT_DEG = 0.07892
_TX = 2.757
_TY = 555.159

# 实测残差，供 validate 报告里引用
FIT_RESIDUAL_PX = {"median": 0.573, "p95": 1.355, "max": 1.778, "n_inliers": 329}


def _build_similarity() -> np.ndarray:
    """从 (scale, rot, t) 拼出 2×3 相似变换矩阵。"""
    th = np.deg2rad(_ROT_DEG)
    c, s = _SCALE * np.cos(th), _SCALE * np.sin(th)
    return np.array([[c, -s, _TX],
                     [s,  c, _TY]], dtype=np.float64)


# 2×3，real.png px → video px
STILL_TO_VIDEO = _build_similarity()

# 3×3 齐次形式，方便和单应矩阵相乘
STILL_TO_VIDEO_H = np.vstack([STILL_TO_VIDEO, [0.0, 0.0, 1.0]])
VIDEO_TO_STILL_H = np.linalg.inv(STILL_TO_VIDEO_H)


def still_to_video(points) -> np.ndarray:
    """(N, 2) real.png 像素 → (N, 2) 视频帧像素。"""
    arr = np.asarray(points, dtype=np.float32)
    if arr.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    arr = arr.reshape(-1, 2)
    out = cv2.transform(arr.reshape(-1, 1, 2), STILL_TO_VIDEO.astype(np.float32))
    return out.reshape(-1, 2).astype(np.float32)


def video_to_still(points) -> np.ndarray:
    """(N, 2) 视频帧像素 → (N, 2) real.png 像素。"""
    arr = np.asarray(points, dtype=np.float32)
    if arr.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    arr = arr.reshape(-1, 2)
    out = cv2.perspectiveTransform(arr.reshape(-1, 1, 2),
                                   VIDEO_TO_STILL_H.astype(np.float32))
    return out.reshape(-1, 2).astype(np.float32)


def to_video_pixels(points, pixel_frame: str) -> np.ndarray:
    """按 ``pixel_frame`` 把点统一换算到视频像素系。

    pixel_frame:
        ``"video"`` —— 点本来就在 1920×1080 上，原样返回
        ``"still"`` —— 点在 2880×752 的 real.png 上，做相似变换
    """
    if pixel_frame == "video":
        return np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if pixel_frame == "still":
        return still_to_video(points)
    raise ValueError(
        f"未知的 pixel_frame={pixel_frame!r}，只支持 'video' 或 'still'。"
    )


def describe() -> str:
    """一段人读的说明，写进 json / 打进日志。"""
    return (f"real.png({STILL_RESOLUTION[0]}x{STILL_RESOLUTION[1]}) -> "
            f"video({VIDEO_RESOLUTION[0]}x{VIDEO_RESOLUTION[1]}): "
            f"similarity scale={_SCALE:.6f} rot={_ROT_DEG:.5f}deg "
            f"t=({_TX:.3f}, {_TY:.3f}); "
            f"measured residual median={FIT_RESIDUAL_PX['median']:.3f}px "
            f"max={FIT_RESIDUAL_PX['max']:.3f}px")
