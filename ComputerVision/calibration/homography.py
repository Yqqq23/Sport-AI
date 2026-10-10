"""单应变换。

⚠ 原理约束一：单应变换只对**平面**有效
------------------------------------
球场地面是平的，所以「画面像素 ↔ 球场米」这个映射成立。但**球员是立体的**。
投影球员位置时，必须取检测框的 **底边中点**（脚与地面的接触点），
因为只有脚在地面这个平面上。

    ✅ foot  = ((x1 + x2) / 2, y2)      ← 底边中点
    ❌ centre = ((x1 + x2) / 2, (y1 + y2) / 2)   ← 框中心 ≈ 胸口，离地约 1 米

用框中心会怎样：胸口不在地平面上，单应把它当成地平面上的点来解，
会沿着「相机 → 胸口」这条视线一路延伸到地面，落点被推向远离相机的方向，
实测偏差通常是 **1~2 米**。在 15×11 米的场地上，这个误差足以让人跨越三分线。
球和篮筐同理（球在空中时根本不能投影）。

⚠ 原理约束二：至少 4 个点，且不能退化
----------------------------------
单应有 8 个自由度，一对点给 2 个方程，所以最少要 4 对点。
但「4 个点」不够，还必须**处于一般位置**：如果 4 个点里有 3 个共线，
方程组秩不足，矩阵是奇异的。本模块把「点数不足」和「点退化」区分成两种
不同的异常信息，不会含混地报一个 None。

注意：**球场上本来就有很多共线的点**（比如底线上的 12/10/8/9/11/13 六个点
全部共线），这是正常的。要求的是「存在至少一组 4 个点两两不共线」，
不是「任意三点都不共线」。

⚠ RANSAC 阈值的单位陷阱 —— 这里和最初的设计稿有一处**故意的偏离**
------------------------------------------------------------------
``cv2.findHomography(src, dst, cv2.RANSAC, t)`` 的阈值 ``t``，单位是
**dst 的单位**，不是像素。OpenCV 文档里的判据是：

    ||dst_i - H * src_i|| > t   ⇒   第 i 个点是 outlier

如果按「source=画面像素、target=球场米」直接调用并传 5.0，那 5.0 就是
**5 米**。在 15×11 米的场地上，5 米的容忍度等于完全不做剔除 ——
RANSAC 名义上开着，实际上退化成最小二乘。这一点已被本包的冒烟测试实测确认：
把一个点故意点歪 90 像素，用 5.0「米」阈值时它仍被判为 inlier。

所以这里把 RANSAC 放在 **球场米 → 画面像素** 方向做，阈值才真正是像素；
最终的「画面 → 球场」矩阵取其精确逆矩阵。

这样做除了让阈值有意义，统计上也更对：点击误差本来就是**像素空间**里
大致均匀的（手抖 ±3~5 px），而在米空间里做最小化，会让球场远端的点
（1 像素 ≈ 30 厘米）获得极大的权重，把矩阵往远端拽偏。
"""

from __future__ import annotations

from itertools import combinations

import cv2
import numpy as np

# RANSAC 重投影阈值，单位**像素**（见模块顶部关于单位陷阱的说明）。
# 手点一个球场角点的实际精度大约 ±3~5 px，所以 5.0 px 刚好卡在
# 「正常的手抖」和「真的点错地方」之间。
RANSAC_REPROJ_THRESHOLD_PX = 5.0

# 兼容旧名字
RANSAC_REPROJ_THRESHOLD = RANSAC_REPROJ_THRESHOLD_PX

# 判定三点共线的面积阈值。三角形面积小于它就算共线。
# 单位是「源坐标系单位的平方」，像素系下 1.0 相当于三点几乎压在一条线上。
COLLINEAR_AREA_EPS = 1.0

# 点数不超过这个数就用穷举最小样本搜索代替随机 RANSAC（确定性，且更准）
EXHAUSTIVE_MAX_POINTS = 20


class InsufficientPointsError(ValueError):
    """点数不足 4 对。"""


class DegenerateConfigurationError(ValueError):
    """点数够，但几何退化（共线），解不出单应。"""


class HomographyFitError(RuntimeError):
    """点数和几何都没问题，但 cv2.findHomography 仍然失败了。"""


def _as_2d(points, name: str) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[1] == 1 and arr.shape[2] == 2:
        arr = arr.reshape(-1, 2)          # 接受 (N,1,2) 这种 OpenCV 惯用形状
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{name} 应为 (N, 2) 的坐标数组，实际形状是 {arr.shape}")
    if not np.isfinite(arr).all():
        bad = np.where(~np.isfinite(arr).all(axis=1))[0].tolist()
        raise ValueError(
            f"{name} 里有 NaN/Inf，下标 {bad}。"
            " 多半是把「未标注」或「court_config 里还是 TODO」的点混进来了 ——"
            " 请在传进来之前用同一个布尔掩码把两边一起过滤掉。"
        )
    return arr


def _triangle_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return abs(float(np.cross(b - a, c - a))) / 2.0


def _exhaustive_ransac(src: np.ndarray, dst: np.ndarray, threshold: float):
    """穷举所有 4 点最小样本的 RANSAC。返回 (H, mask)，和 cv2 的接口一致。

    标定场景点数少（十几个），C(n,4) 全试一遍成本可以忽略，好处是
    **确定性**：同样的输入永远给同样的结果，不会像随机 RANSAC 那样
    时好时坏、还可能漏掉最优子集。

    评分先比内点数，再比内点的平均残差。最后在内点上做一次最小二乘重拟合。
    """
    n = len(src)
    best = None
    for quad in combinations(range(n), 4):
        s = src[list(quad)].astype(np.float32)
        d = dst[list(quad)].astype(np.float32)
        # 样本自身退化（有三点共线）就跳过
        if any(_triangle_area(s[i], s[j], s[k]) <= COLLINEAR_AREA_EPS
               for i, j, k in combinations(range(4), 3)):
            continue
        try:
            h = cv2.getPerspectiveTransform(s, d)
        except cv2.error:
            continue
        if not np.isfinite(h).all():
            continue
        proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2).astype(np.float32), h)
        if proj is None or not np.isfinite(proj).all():
            continue
        err = np.linalg.norm(proj.reshape(-1, 2) - dst, axis=1)
        inl = err < threshold
        n_in = int(inl.sum())
        if n_in < 4:
            continue
        score = (n_in, -float(err[inl].mean()))
        if best is None or score > best[0]:
            best = (score, inl.copy())

    if best is None:
        return None, None
    inl = best[1]
    # 只用内点重新做最小二乘拟合
    h, _ = cv2.findHomography(src[inl].astype(np.float32),
                              dst[inl].astype(np.float32), 0)
    return h, inl.astype(np.uint8).reshape(-1, 1)


def check_configuration(points: np.ndarray, name: str) -> None:
    """点数 + 退化检查。两种失败给两种不同的异常和信息。"""
    n = len(points)
    if n < 4:
        raise InsufficientPointsError(
            f"{name} 只有 {n} 个点，单应至少需要 4 个（8 个自由度 ÷ 每点 2 个方程）。"
            " 如果是因为有点被遮挡没标出来，那这次标定就得补标或换一帧。"
        )

    # 全部共线？
    a = points[0]
    rest = points[1:]
    if all(_triangle_area(a, rest[i], rest[j]) <= COLLINEAR_AREA_EPS
           for i, j in combinations(range(len(rest)), 2)):
        raise DegenerateConfigurationError(
            f"{name} 的 {n} 个点全部共线，构不成平面，单应无解。"
            " 检查是不是只点了底线上那一排点。"
        )

    # 存在一组 4 点两两不共线？
    for quad in combinations(range(n), 4):
        if all(_triangle_area(points[i], points[j], points[k]) > COLLINEAR_AREA_EPS
               for i, j, k in combinations(quad, 3)):
            return
    raise DegenerateConfigurationError(
        f"{name} 的 {n} 个点里找不到任何 4 个处于一般位置（每一组 4 点中都至少有 3 点共线），"
        " 几何退化，单应无解。需要补充不在同一直线上的点。"
    )


class ViewTransformer:
    """把源平面上的点映射到目标平面。

    典型用法：source = 画面像素，target = 球场米。

    Parameters
    ----------
    source : (N, 2)
        源平面坐标（画面像素）。
    target : (N, 2)
        目标平面坐标（球场米）。必须和 source 一一对应、顺序一致。

    Attributes
    ----------
    m : (3, 3) float64
        单应矩阵 source → target。
    inlier_mask : (N,) bool
        RANSAC 判定的内点掩码，供 validate.py 诊断哪几个点标歪了。
    outlier_indices : list[int]
        被判为 outlier 的点在输入数组里的下标。
    ransac_threshold_px : float
        RANSAC 内外点阈值，单位**像素**。
    """

    def __init__(self, source, target,
                 ransac_threshold_px: float = RANSAC_REPROJ_THRESHOLD_PX):
        src = _as_2d(source, "source")
        dst = _as_2d(target, "target")

        if len(src) != len(dst):
            raise ValueError(
                f"source 有 {len(src)} 个点、target 有 {len(dst)} 个点，对不上。"
                " 这几乎总是因为过滤时只索引了一边 —— 必须用**同一个**布尔掩码"
                " 同时索引 image_points 和 CONFIG.vertices。"
            )

        check_configuration(src, "source（画面点）")
        check_configuration(dst, "target（球场标准点）")

        # 必须用 RANSAC：一个点标歪就会把最小二乘解整体拽偏，
        # 而 RANSAC 会把它扔进 outlier，其余点照样得到正确的矩阵。
        #
        # 注意方向：拟合的是 target(米) -> source(像素)，这样 findHomography
        # 的阈值单位才真的是像素。理由见模块顶部「RANSAC 阈值的单位陷阱」。
        #
        # 点数少时改用**穷举**最小样本搜索：标定场景通常只有十几个点，
        # C(13,4)=715 个四点样本全试一遍毫无压力，而且是确定性的。
        # cv2 的随机 RANSAC 在这个规模下会漏掉最优子集 —— 实测在本项目的
        # 13 点数据上，它选中的子集只有 5 个内点，穷举则找到 7 个自洽点。
        if len(src) <= EXHAUSTIVE_MAX_POINTS:
            m_t2s, mask = _exhaustive_ransac(dst, src, ransac_threshold_px)
        else:
            m_t2s, mask = cv2.findHomography(dst, src, cv2.RANSAC, ransac_threshold_px)
        if m_t2s is None:
            raise HomographyFitError(
                "cv2.findHomography 返回了 None。点数和共线性都过了检查，"
                "所以最可能的原因是 source / target 的**编号顺序对不上**"
                "（比如点击顺序和参考图编号不一致），导致没有任何一致的模型。"
            )

        self.inlier_mask = mask.ravel().astype(bool)
        self.source = src
        self.target = dst
        self.ransac_threshold_px = float(ransac_threshold_px)

        if self.inlier_mask.sum() < 4:
            raise HomographyFitError(
                f"RANSAC 只找到 {int(self.inlier_mask.sum())} 个内点（< 4），矩阵不可信。"
                " 通常意味着点标得很乱，或者编号顺序错位。"
            )

        if abs(np.linalg.det(m_t2s)) < 1e-12:
            raise HomographyFitError(
                "拟合出的矩阵是奇异的（行列式≈0），无法求逆。几何配置退化。"
            )

        self._inv = (m_t2s / m_t2s[2, 2]).astype(np.float64)   # 球场米 -> 画面像素
        m = np.linalg.inv(self._inv)
        self.m = (m / m[2, 2]).astype(np.float64)              # 画面像素 -> 球场米

    # -- 属性 --------------------------------------------------------------
    @property
    def inverse_matrix(self) -> np.ndarray:
        """(3, 3) 逆矩阵 target → source，用于把球场边线投回画面做反向检查。

        这个方向才是 RANSAC 直接拟合出来的那一个，``m`` 是它的精确逆。
        """
        return self._inv

    def reprojection_errors_px(self) -> np.ndarray:
        """(N,) 每个点在**像素空间**的重投影误差 —— RANSAC 内外点判据本身。"""
        pred = self.inverse_transform_points(self.target)
        return np.linalg.norm(pred - self.source, axis=1)

    @property
    def outlier_indices(self) -> list[int]:
        return np.where(~self.inlier_mask)[0].tolist()

    @property
    def num_inliers(self) -> int:
        return int(self.inlier_mask.sum())

    # -- 变换 --------------------------------------------------------------
    def transform_points(self, points) -> np.ndarray:
        """source → target。返回 (N, 2) float32。

        空数组原样返回 (0, 2)，不报错 —— 某一帧没检出任何球员是正常情况。
        """
        arr = np.asarray(points, dtype=np.float32)
        if arr.size == 0:
            return np.empty((0, 2), dtype=np.float32)
        arr = _as_2d(arr, "points")
        out = cv2.perspectiveTransform(arr.reshape(-1, 1, 2), self.m)
        return out.reshape(-1, 2).astype(np.float32)

    def inverse_transform_points(self, points) -> np.ndarray:
        """target → source。用来把球场坐标投回画面像素。"""
        arr = np.asarray(points, dtype=np.float32)
        if arr.size == 0:
            return np.empty((0, 2), dtype=np.float32)
        arr = _as_2d(arr, "points")
        out = cv2.perspectiveTransform(arr.reshape(-1, 1, 2), self._inv)
        return out.reshape(-1, 2).astype(np.float32)

    def transform_boxes_to_court(self, boxes) -> np.ndarray:
        """把 (N, 4) 的 xyxy 检测框投影成球场坐标。

        **取底边中点**，不是框中心 —— 原因见模块顶部的原理约束一。
        """
        arr = np.asarray(boxes, dtype=np.float32)
        if arr.size == 0:
            return np.empty((0, 2), dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 4:
            raise ValueError(f"boxes 应为 (N, 4) 的 xyxy，实际形状是 {arr.shape}")
        x1, _y1, x2, y2 = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
        feet = np.stack([(x1 + x2) / 2.0, y2], axis=1)   # 底边中点 = 脚
        return self.transform_points(feet)

    def horizon_crosses(self, polygon_target) -> bool:
        """目标平面上的这块区域是否跨越了地平线。

        跨越 = 齐次坐标第三分量 w 变号 = 区域有一部分「在相机背后」，
        物理上不可能。出现这种情况说明矩阵是错的，而不只是精度差。
        """
        pts = np.asarray(polygon_target, dtype=np.float64).reshape(-1, 2)
        w = pts @ self._inv[2, :2] + self._inv[2, 2]
        return bool(w.min() * w.max() <= 0)

    def __repr__(self) -> str:
        return (f"<ViewTransformer {len(self.source)} 点, "
                f"inliers {self.num_inliers}/{len(self.source)}>")
