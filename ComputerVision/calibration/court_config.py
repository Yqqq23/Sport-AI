"""3x3 篮球场 13 个特征点在真实球场上的坐标。

⚠ 场地尺寸：15 m × **14 m**，不是 3x3 官方的 15 × 11
--------------------------------------------------
这块场地地上的**蓝线**是一个标准 FIBA 全场（28 × 15 m）的线，
比赛只用其中一半。所以实际几何是 **FIBA 半场 15 m 宽 × 14 m 深**。

FIBA 3x3 官方专用场是 15 × 11 —— 这里**不适用**，深度差 3 米。
限制区和三分弧两边通用，所以只有 #1 / #2（中线两端）受影响：
Y 坐标是 14000 而不是 11000。

下游任何按「球场深度」算的东西（热区网格、跑动距离归一化、
半场划分）都要用 14 m，不要照抄 3x3 的 11 m。

坐标系（俯视图，右手系）
------------------------
    原点 (0, 0)  = 底线左角  corner_bl  (#12)
    +X           = 沿底线向右，0 → 15.000 m
    +Y           = 从底线指向中线（远离篮筐），0 → 14.000 m

        Y=14 ┌───────────────────────┐   ← 中线 half-court line（全场中线）
             │ #1 corner_tl   #2 tr  │
             │                       │
             │        (球场)          │
             │                       │
        Y=0  └───────────────────────┘   ← 底线 baseline（篮筐在这一侧）
             #12 corner_bl      #13 corner_br
             X=0                      X=15

    「左 / 右」一律指**俯视图上的左右**（即上面这张图的左右），
    不是摄像机画面里的左右，也不是站在场上球员的左右。

单位约定
--------
内部一律用**毫米整数**存储，避免 0.1 + 0.2 这类浮点误差累积；
``.vertices`` 属性对外返回 **米** 的 numpy 数组供下游使用。

编号顺序
--------
严格照抄参考图 standardone.png 上的 1–13 号。点击取点时必须按同一顺序，
否则整个矩阵作废（validate.py 会以「疑似编号错位」的形式报出来）。

⚠ 参考图自身的标签冲突（填坐标前请先裁定）
------------------------------------------
参考图上 6 号点的引线同时挂了两个名字：``three_left_break``（3-POINT ARC
LEFT BREAK）和 ``10 three_left_base``（3-POINT LINE LEFT BASELINE END）；
而真正的 10 号点在底线上，名字也叫 ``three_left_base``。7 / 11 号同样如此。
本文件按「6/7 = 弧顶两侧折点，10/11 = 三分线底线端点」采用，
如果你的本意相反，改 POINT_NAMES 里 6/7 与 10/11 的名字并相应调整坐标。

FIBA 半场尺寸依据（单位 mm，下面所有坐标都是从这几个数推出来的）
--------------------------------------------------------------
    全场          28000 × 15000  →  半场 15000 宽 × 14000 深
    限制区(paint) 4900 宽 × 5800 深（现代 FIBA 是矩形，不是老式梯形）
    罚球线        距底线内沿 5800，居中
    篮筐中心      距底线内沿 1575，位于球场中轴 X = 7500
    三分弧        半径 6750，圆心 = 篮筐中心在地面的投影
    三分线直线段  距边线 900，从底线延伸到与圆弧相交处

    #6 / #7 折点的 Y 是算出来的，不是量出来的：
        直线 X = 900，圆心 (7500, 1575)，半径 6750
        dx = 7500 - 900 = 6600
        dy = √(6750² - 6600²) = √2002500 = 1415.1
        Y  = 1575 + 1415.1 = 2990.1  →  取 2990

⚠ 这些是**标准尺寸**。那是个多用途馆，地上叠着好几套线，如果实际画线
和标准有出入，validate.py 的误差会立刻暴露出来 —— 平均误差
超过 0.2 m 就回来改这里的数。
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# 球场外框尺寸（毫米整数）—— FIBA 半场 15m × 14m
# 注意不是 3x3 官方的 15×11：这块地板画的是标准全场，比赛用其中一半。
# ---------------------------------------------------------------------------
COURT_WIDTH_MM = 15_000   # 沿底线方向（X）
COURT_LENGTH_MM = 14_000  # 底线到中线（Y）= 全场 28000 的一半

# 下面这些是推导 13 个点坐标的中间量，改场地时改这里就够了
PAINT_WIDTH_MM = 4_900       # 限制区宽
FREE_THROW_DIST_MM = 5_800   # 罚球线距底线
BASKET_DIST_MM = 1_575       # 篮筐中心距底线
THREE_RADIUS_MM = 6_750      # 三分弧半径
THREE_INSET_MM = 900         # 三分线直线段距边线

_CX = COURT_WIDTH_MM // 2                    # 球场中轴 X = 7500
_PAINT_HALF = PAINT_WIDTH_MM // 2            # 2450
# 三分弧折点：直线 X=THREE_INSET 与圆的交点
_dx = _CX - THREE_INSET_MM                   # 6600
_THREE_BREAK_Y = int(round(
    BASKET_DIST_MM + (THREE_RADIUS_MM ** 2 - _dx ** 2) ** 0.5))   # 2990

# 点的数量。参考图上实际画了 13 个标记，不是 12 个。
NUM_POINTS = 13

# 编号 → 名字。索引 0 对应 1 号点，以此类推。
#
# ⚠ 名字已按**实地核对结果**修正，和 standardone.png 的标注不完全一致
# ---------------------------------------------------------------------
# 参考图上的名字是照着一张通用球场图排的，和这块地板上实际能认出来的
# 特征对不上。经过把标准球场模型反投影到 court_plate.png 逐条比对，
# 确认下面 5 个槽位（其余 6 个见 VERTICES_MM 里的说明）：
#
#   槽位 5  → 三分弧顶      （参考图写的是 free_throw_center，错的）
#   槽位 8  → 三分弧左折点  （参考图写的是 paint_bl，错的）
#   槽位 9  → 三分弧右折点  （参考图写的是 paint_br，错的）
#   槽位 10 → 三分线左底端  （和参考图一致）
#   槽位 11 → 三分线右底端  （和参考图一致）
#
# 判据：把 (0.9,0)-(0.9,2.99) 和 (14.1,0)-(14.1,2.99) 两段直线逆投影回
# 画面后，与地板上那两条深蓝色三分直线段**逐像素重合**；
# 半径 6.75 的弧同时精确穿过槽位 5。
POINT_NAMES = (
    "paint_base_left",     # 1  限制区 · 底线端 左角
    "paint_base_right",    # 2  限制区 · 底线端 右角
    "three_base_left",     # 3  三分线 · 底线左端点
    "three_base_right",    # 4  三分线 · 底线右端点
    "three_break_left",    # 5  三分弧 · 左折点（直线段与弧的交界）
    "three_break_right",   # 6  三分弧 · 右折点
    "paint_ft_left",       # 7  限制区 · 罚球线端 左角
    "paint_ft_right",      # 8  限制区 · 罚球线端 右角
    "ft_circle_far",       # 9  罚球圈 · 离底线最远的那一点（中轴上）
    "three_arc_top",       # 10 三分弧 · 弧顶（中轴上）
    "centre_circle_near",  # 11 中圈 · 离底线最近的那一点（中轴上）
    "halfway_left",        # 12 中线 × 左边线
    "halfway_right",       # 13 中线 × 右边线
)
assert len(POINT_NAMES) == NUM_POINTS

# ---------------------------------------------------------------------------
# 13 个点的球场坐标，单位 **毫米整数**，(X, Y)。
#
# 全部由上面那几个 FIBA 标准尺寸推出，没有一个是硬编码的猜测值 ——
# 想换场地只需改上面的常量。
#
# None 表示「还没填」；如果哪个点你现场量出来和标准不一致，改这里的数即可。
# calibrate / validate 会把所有 None 的编号列出来并报错，
# 不会拿假坐标算出一个看起来能跑的错矩阵。
# ---------------------------------------------------------------------------
_W = COURT_WIDTH_MM
_L = COURT_LENGTH_MM

# 三分弧顶：在中轴上，距底线 = 篮筐距底线 + 半径
_ARC_TOP_Y = BASKET_DIST_MM + THREE_RADIUS_MM                # 8325

# 罚球圈 / 中圈半径（FIBA 都是 1800 mm）
CIRCLE_R_MM = 1_800
_FT_FAR_Y = FREE_THROW_DIST_MM + CIRCLE_R_MM                 # 7600
_CENTRE_NEAR_Y = COURT_LENGTH_MM - CIRCLE_R_MM               # 12200

VERTICES_MM: tuple[tuple[int, int] | None, ...] = (
    # ---- 底线一带（最靠近篮筐）----------------------------------------
    (_CX - _PAINT_HALF, 0),                    # 1  限制区底线左角   (5050, 0)
    (_CX + _PAINT_HALF, 0),                    # 2  限制区底线右角   (9950, 0)
    (THREE_INSET_MM, 0),                       # 3  三分线底线左端   (900, 0)
    (_W - THREE_INSET_MM, 0),                  # 4  三分线底线右端   (14100, 0)
    # ---- 三分弧折点 -----------------------------------------------------
    (THREE_INSET_MM, _THREE_BREAK_Y),          # 5  左折点           (900, 2990)
    (_W - THREE_INSET_MM, _THREE_BREAK_Y),     # 6  右折点           (14100, 2990)
    # ---- 罚球线一带 -----------------------------------------------------
    (_CX - _PAINT_HALF, FREE_THROW_DIST_MM),   # 7  限制区罚球线左角 (5050, 5800)
    (_CX + _PAINT_HALF, FREE_THROW_DIST_MM),   # 8  限制区罚球线右角 (9950, 5800)
    (_CX, _FT_FAR_Y),                          # 9  罚球圈最远点     (7500, 7600)
    (_CX, _ARC_TOP_Y),                         # 10 三分弧顶         (7500, 8325)
    # ---- 中场（远场约束，最关键）----------------------------------------
    (_CX, _CENTRE_NEAR_Y),                     # 11 中圈最近点       (7500, 12200)
    (0, _L),                                   # 12 中线左端         (0, 14000)
    (_W, _L),                                  # 13 中线右端         (15000, 14000)
)
assert len(VERTICES_MM) == NUM_POINTS


class Court3x3Configuration:
    """3x3 半场 13 个特征点的球场标准坐标。

    内部毫米整数，对外 ``.vertices`` 给米。未填的点在 ``.vertices`` 里是 NaN，
    并由 ``.defined_mask`` 标出，供 calibrate 做双边同步过滤。
    """

    width_mm = COURT_WIDTH_MM
    length_mm = COURT_LENGTH_MM
    num_points = NUM_POINTS
    names = POINT_NAMES

    def __init__(self, vertices_mm=VERTICES_MM):
        if len(vertices_mm) != NUM_POINTS:
            raise ValueError(
                f"球场坐标表长度是 {len(vertices_mm)}，应为 {NUM_POINTS}（参考图是 13 个点）"
            )
        for i, v in enumerate(vertices_mm):
            if v is None:
                continue
            if len(v) != 2 or not all(isinstance(c, (int, np.integer)) for c in v):
                raise TypeError(
                    f"{self.label(i)} 的坐标应为两个毫米整数，实际是 {v!r}。"
                    " 内部一律用整数毫米，不要写浮点。"
                )
        self._mm = tuple(vertices_mm)

    # -- 基本查询 ----------------------------------------------------------
    def label(self, index0: int) -> str:
        """给出形如 ``#6 three_left_break`` 的可读标签（index0 从 0 开始）。"""
        return f"#{index0 + 1} {POINT_NAMES[index0]}"

    @property
    def defined_mask(self) -> np.ndarray:
        """(13,) bool —— 哪些点已经填了真实坐标。"""
        return np.array([v is not None for v in self._mm], dtype=bool)

    @property
    def todo_indices(self) -> list[int]:
        """还没填的点的 0-based 下标。"""
        return [i for i, v in enumerate(self._mm) if v is None]

    @property
    def vertices_mm(self) -> np.ndarray:
        """(13, 2) float —— 毫米。未填的点为 NaN。"""
        out = np.full((NUM_POINTS, 2), np.nan, dtype=np.float64)
        for i, v in enumerate(self._mm):
            if v is not None:
                out[i] = v
        return out

    @property
    def vertices(self) -> np.ndarray:
        """(13, 2) float32 —— **米**，供下游使用。未填的点为 NaN。

        NaN 是刻意的：它保证任何未填的点一旦被误用于计算，结果会立刻变成
        NaN 而不是一个看似合理的数字。
        """
        return (self.vertices_mm / 1000.0).astype(np.float32)

    # -- 校验 --------------------------------------------------------------
    def require_all_defined(self) -> None:
        """所有 13 个点都填了才通过，否则抛异常并列出缺哪些。"""
        todo = self.todo_indices
        if todo:
            lines = "\n".join("    " + self.label(i) for i in todo)
            raise ValueError(
                f"court_config.py 里还有 {len(todo)} 个点是 TODO（VERTICES_MM 中的 None）：\n"
                f"{lines}\n"
                "  请照参考图 standardone.png 填上毫米整数坐标。\n"
                "  只想先拿已填的角点跑通流程，就加 --allow-todo。"
            )

    def outline_m(self) -> np.ndarray:
        """(4, 2) 球场外框四角，米，按 bl → br → tr → tl 顺序。

        只依赖 15000×11000，和 TODO 无关，所以可视化任何时候都能画。
        """
        w, l = self.width_mm / 1000.0, self.length_mm / 1000.0
        return np.array([[0, 0], [w, 0], [w, l], [0, l]], dtype=np.float32)

    def __repr__(self) -> str:
        return (f"<Court3x3Configuration {self.width_mm}x{self.length_mm}mm, "
                f"{NUM_POINTS - len(self.todo_indices)}/{NUM_POINTS} 点已填>")


# 下游统一从这里取，避免各处重复构造
CONFIG = Court3x3Configuration()
