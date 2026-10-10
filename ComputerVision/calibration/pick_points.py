"""交互式取点工具：在底图上按编号顺序点出 13 个特征点。

用法
----
    python -m calibration.pick_points --image court_plate.png --no-hints
    python -m calibration.pick_points --image court_plate.png --margin 3000

画面外的点怎么标
----------------
底线两角常常被镜头切掉。但单应变换作用于**整个像平面**，不是只作用于
传感器那块矩形 —— 画面外的点只要能定位，像素坐标（哪怕是负数、哪怕是
几千）一样有效。

本工具的视野是**无边界**的：可以一直缩小、一直平移到图像外任意远处，
图像周围就是空白。不存在「外扩不够」这回事，也不占额外内存
（不会真去造一张放大的画布，只是把图像画在视口里该在的位置）。

    z / 滚轮向下   一直缩小，直到看见你要的那片空白
    f              自动把视野缩放到「图像 + 所有已放的点和辅助线交点」都能看见
    0              回到「图像 + margin」的初始视野

标画面外的点，用辅助线求交：

    1. 按 g，沿一条线（比如左边线）在**能看清的地方**点两下 -> 建立一条无限长直线
    2. 再按 g，沿另一条线（比如底线）点两下
    3. 按 x -> 工具直接把两条辅助线的**交点**作为当前编号的点落下

角点是**算出来的**，不是在空白处估的。

操作
----
    左键单击      放下当前编号的点
    g             辅助线模式：接下来两次左键定义一条无限长直线
    x             用最后两条辅助线的交点作为当前编号的点
    C             清空所有辅助线
    u             撤销上一个点     k  跳过当前点（看不见就跳过）
    滚轮 / + -    缩放（无下限，可以一直缩小）
    右键/中键拖动  平移（无边界）   方向键 也可平移
    f             自适应视野       0  复位视野
    s             保存并退出       q / ESC  不保存退出

⚠ 屏幕上文字一律用英文：cv2.putText 渲染不了中文，中文说明打在控制台里。
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import cv2
import numpy as np

from .court_config import CONFIG, NUM_POINTS, POINT_NAMES
from .io_utils import asset, calib, imread_unicode
from .pixel_frames import STILL_RESOLUTION, VIDEO_RESOLUTION

WINDOW = "pick court points  |  g=guide  x=intersect  f=fit  u=undo  k=skip  s=save  q=quit"
WIN_W, WIN_H = 1600, 900
LOUPE_SIZE = 220
LOUPE_ZOOM = 8
OUTPUT_JSON = calib("image_points.json")

C_DONE = (60, 220, 60)
C_NEXT = (0, 200, 255)
C_HINT = (255, 140, 0)
C_SKIP = (140, 140, 140)
C_GUIDE = (0, 165, 255)
C_HUD = (255, 255, 255)
C_VOID = (22, 22, 22)


# ---------------------------------------------------------------------------
def detect_marks(annotated: np.ndarray, clean: np.ndarray) -> list[tuple[float, float]]:
    """差分出手画标记的质心（仅作提示）。"""
    if annotated.shape != clean.shape:
        raise ValueError(
            f"标注图 {annotated.shape[1]}x{annotated.shape[0]} 和干净图 "
            f"{clean.shape[1]}x{clean.shape[0]} 尺寸不一致，无法差分。"
        )
    diff = cv2.absdiff(annotated, clean).max(axis=2)
    mask = (diff > 40).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, _lab, stats, cent = cv2.connectedComponentsWithStats(mask)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 800 or w < 25 or h < 20:
            continue
        out.append((float(cent[i][0]), float(cent[i][1])))
    out.sort(key=lambda p: (round(p[1] / 60), p[0]))
    return out


def line_intersection(a1, a2, b1, b2):
    """两条直线（各由两点确定）的交点。平行则抛异常。"""
    l1 = np.cross([a1[0], a1[1], 1.0], [a2[0], a2[1], 1.0])
    l2 = np.cross([b1[0], b1[1], 1.0], [b2[0], b2[1], 1.0])
    p = np.cross(l1, l2)
    if abs(p[2]) < 1e-9:
        raise ValueError("两条辅助线平行（或重合），没有交点。")
    return float(p[0] / p[2]), float(p[1] / p[2])


# ---------------------------------------------------------------------------
class Picker:
    """坐标一律是**原图像素**，可以是负数或远超宽高 —— 视口无边界。"""

    # 视口允许漫游的范围，以图像尺寸为单位（够用到离谱，又不至于迷路）
    ROAM = 12.0
    ZOOM_MIN_ABS = 0.01

    def __init__(self, img, image_name, pixel_frame, hints, margin):
        self.img = img
        self.h, self.w = img.shape[:2]
        self.image_name = image_name
        self.pixel_frame = pixel_frame
        self.hints = list(hints)
        self.margin = float(margin)
        self.points: list[tuple[float, float] | None] = []
        self.guides: list[tuple[tuple[float, float], tuple[float, float]]] = []
        self.guide_buf: list[tuple[float, float]] = []
        self.guide_mode = False
        self.cursor = (0.0, 0.0)
        self._drag = None
        self.reset_view()

    # -- 视野（无边界）------------------------------------------------------
    def set_view(self, x0, y0, x1, y1):
        """把视野设成能装下 [x0,x1]x[y0,y1] 这块区域。"""
        w, h = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
        self.zoom = max(min(WIN_W / w, WIN_H / h), self.ZOOM_MIN_ABS)
        self.ox = (x0 + x1) / 2.0 - WIN_W / (2 * self.zoom)
        self.oy = (y0 + y1) / 2.0 - WIN_H / (2 * self.zoom)

    def reset_view(self):
        m = self.margin
        self.set_view(-m, -m, self.w + m, self.h + m)

    def fit_all(self):
        """缩放到能同时看见图像、所有已放的点、所有辅助线交点。"""
        xs = [0.0, float(self.w)]
        ys = [0.0, float(self.h)]
        for p in self.points:
            if p is not None:
                xs.append(p[0]); ys.append(p[1])
        for i in range(len(self.guides)):
            for j in range(i + 1, len(self.guides)):
                try:
                    q = line_intersection(*self.guides[i], *self.guides[j])
                except ValueError:
                    continue
                if abs(q[0]) < 1e6 and abs(q[1]) < 1e6:
                    xs.append(q[0]); ys.append(q[1])
        pad = 0.08 * max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
        self.set_view(min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)

    def _clamp(self):
        r = self.ROAM
        self.ox = float(np.clip(self.ox, -r * self.w, r * self.w))
        self.oy = float(np.clip(self.oy, -r * self.h, r * self.h))

    def to_image(self, sx, sy):
        return self.ox + sx / self.zoom, self.oy + sy / self.zoom

    def to_screen(self, ix, iy):
        return (ix - self.ox) * self.zoom, (iy - self.oy) * self.zoom

    def zoom_at(self, sx, sy, factor):
        ix, iy = self.to_image(sx, sy)
        self.zoom = float(np.clip(self.zoom * factor, self.ZOOM_MIN_ABS, 60.0))
        self.ox, self.oy = ix - sx / self.zoom, iy - sy / self.zoom
        self._clamp()

    def pan(self, dx, dy):
        self.ox -= dx / self.zoom
        self.oy -= dy / self.zoom
        self._clamp()

    # -- 取点 --------------------------------------------------------------
    @property
    def next_index(self):
        return len(self.points)

    @property
    def done(self):
        return len(self.points) >= NUM_POINTS

    def click(self, ix, iy):
        if self.guide_mode:
            self.guide_buf.append((ix, iy))
            if len(self.guide_buf) == 2:
                self.guides.append((self.guide_buf[0], self.guide_buf[1]))
                self.guide_buf = []
                self.guide_mode = False
                msg = f"  辅助线 L{len(self.guides)} 已建立"
                if len(self.guides) >= 2:
                    try:
                        q = line_intersection(*self.guides[-2], *self.guides[-1])
                        msg += f"  —— 与 L{len(self.guides)-1} 交于 ({q[0]:.1f}, {q[1]:.1f})，按 x 采用"
                    except ValueError:
                        msg += "  —— 与上一条平行，没有交点"
                print(msg)
            return
        self.add(ix, iy)

    def add(self, ix, iy):
        if self.done:
            print("  ! 13 个点已点满，按 s 保存或 u 撤销")
            return
        self.points.append((float(ix), float(iy)))
        i = len(self.points)
        outside = not (0 <= ix < self.w and 0 <= iy < self.h)
        tag = "   [画面外，坐标依然有效]" if outside else ""
        print(f"  #{i:2d} {POINT_NAMES[i - 1]:<20} = ({ix:9.2f}, {iy:9.2f}){tag}")

    def intersect(self):
        if len(self.guides) < 2:
            print("  ! 至少要有两条辅助线（按 g 建立）")
            return
        try:
            p = line_intersection(*self.guides[-2], *self.guides[-1])
        except ValueError as e:
            print(f"  ! {e}")
            return
        self.add(*p)

    def skip(self):
        if self.done:
            return
        self.points.append(None)
        print(f"  #{len(self.points):2d} {POINT_NAMES[len(self.points) - 1]:<20} = 跳过（未标注）")

    def undo(self):
        if not self.points:
            print("  ! 没有可撤销的点")
            return
        removed = self.points.pop()
        i = len(self.points) + 1
        if removed is None:
            print(f"  撤销 #{i} {POINT_NAMES[i - 1]}（原为跳过）")
        else:
            print(f"  撤销 #{i} {POINT_NAMES[i - 1]} 原为 ({removed[0]:.2f}, {removed[1]:.2f})")

    def clear_guides(self):
        self.guides.clear()
        self.guide_buf.clear()
        self.guide_mode = False
        print("  辅助线已清空")

    # -- 渲染 --------------------------------------------------------------
    def render(self):
        # 视口可以在图像外任意远处；warpAffine 用 BORDER_CONSTANT 填空白，
        # 所以不需要真的造一张放大的画布。
        interp = cv2.INTER_NEAREST if self.zoom >= 2.0 else cv2.INTER_LINEAR
        M = np.array([[self.zoom, 0, -self.ox * self.zoom],
                      [0, self.zoom, -self.oy * self.zoom]], np.float32)
        canvas = cv2.warpAffine(self.img, M, (WIN_W, WIN_H), flags=interp,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=C_VOID)

        # 图像边界，提醒哪里开始是画面外
        a = self.to_screen(0, 0)
        b = self.to_screen(self.w, self.h)
        cv2.rectangle(canvas, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                      (110, 110, 110), 1)

        for gi, (p, q) in enumerate(self.guides, 1):
            self._draw_infinite(canvas, p, q, C_GUIDE, gi)
        # 辅助线交点画个记号，方便瞄准
        for i in range(len(self.guides)):
            for j in range(i + 1, len(self.guides)):
                try:
                    q = line_intersection(*self.guides[i], *self.guides[j])
                except ValueError:
                    continue
                s = self.to_screen(*q)
                if -50 < s[0] < WIN_W + 50 and -50 < s[1] < WIN_H + 50:
                    cv2.drawMarker(canvas, (int(s[0]), int(s[1])), C_GUIDE,
                                   cv2.MARKER_TILTED_CROSS, 26, 2)
        for p in self.guide_buf:
            s = self.to_screen(*p)
            cv2.drawMarker(canvas, (int(s[0]), int(s[1])), C_GUIDE, cv2.MARKER_DIAMOND, 16, 2)

        used = [p for p in self.points if p is not None]
        for hx, hy in self.hints:
            if any((hx - px) ** 2 + (hy - py) ** 2 < 60 ** 2 for px, py in used):
                continue
            s = self.to_screen(hx, hy)
            if -40 < s[0] < WIN_W + 40 and -40 < s[1] < WIN_H + 40:
                cv2.drawMarker(canvas, (int(s[0]), int(s[1])), C_HINT,
                               cv2.MARKER_TILTED_CROSS, 22, 2)

        for i, p in enumerate(self.points):
            if p is None:
                continue
            s = self.to_screen(*p)
            if not (-60 < s[0] < WIN_W + 60 and -60 < s[1] < WIN_H + 60):
                continue
            q = (int(s[0]), int(s[1]))
            cv2.circle(canvas, q, 9, C_DONE, 2)
            cv2.drawMarker(canvas, q, C_DONE, cv2.MARKER_CROSS, 16, 1)
            cv2.putText(canvas, str(i + 1), (q[0] + 12, q[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(canvas, str(i + 1), (q[0] + 12, q[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_DONE, 2, cv2.LINE_AA)

        self._draw_hud(canvas)
        self._draw_loupe(canvas)
        return canvas

    def _draw_infinite(self, canvas, a, b, col, idx):
        d = np.array([b[0] - a[0], b[1] - a[1]], float)
        n = np.linalg.norm(d)
        if n < 1e-6:
            return
        d /= n
        big = 1e6
        s0 = self.to_screen(a[0] - d[0] * big, a[1] - d[1] * big)
        s1 = self.to_screen(a[0] + d[0] * big, a[1] + d[1] * big)
        # 坐标可能极大，clipLine 之前先夹到 int32 安全范围
        lim = 1e7
        p0 = (int(np.clip(s0[0], -lim, lim)), int(np.clip(s0[1], -lim, lim)))
        p1 = (int(np.clip(s1[0], -lim, lim)), int(np.clip(s1[1], -lim, lim)))
        cv2.line(canvas, p0, p1, col, 2, cv2.LINE_AA)
        m = self.to_screen((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        if -100 < m[0] < WIN_W + 100 and -100 < m[1] < WIN_H + 100:
            cv2.putText(canvas, f"L{idx}", (int(m[0]) + 6, int(m[1]) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)

    def _draw_hud(self, canvas):
        cv2.rectangle(canvas, (0, 0), (WIN_W, 74), (25, 25, 25), -1)
        n_skip = sum(1 for p in self.points if p is None)
        if self.guide_mode:
            head = f"GUIDE MODE: click 2 points along a line  ({len(self.guide_buf)}/2)"
            col = C_GUIDE
        elif self.done:
            head = f"ALL {NUM_POINTS} DONE  ({n_skip} skipped)  ->  press 's' to save"
            col = C_DONE
        else:
            i = self.next_index
            head = f"NEXT  #{i + 1}/{NUM_POINTS}   {POINT_NAMES[i]}"
            col = C_NEXT
        cv2.putText(canvas, head, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, col, 2, cv2.LINE_AA)
        cx, cy = self.cursor
        cv2.putText(canvas,
                    f"zoom {self.zoom:.3f}x   cursor ({cx:.1f}, {cy:.1f}) px   "
                    f"guides {len(self.guides)}   "
                    f"g=guide x=intersect C=clear f=fit 0=reset u=undo k=skip s=save",
                    (16, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.46, C_HUD, 1, cv2.LINE_AA)
        for i in range(NUM_POINTS):
            x0 = WIN_W - 20 - (NUM_POINTS - i) * 22
            if i < len(self.points):
                c = C_SKIP if self.points[i] is None else C_DONE
            else:
                c = C_NEXT if i == self.next_index else (70, 70, 70)
            cv2.rectangle(canvas, (x0, 46), (x0 + 16, 62), c, -1)

    def _draw_loupe(self, canvas):
        cx, cy = self.cursor
        half = LOUPE_SIZE / (2 * LOUPE_ZOOM)
        M = np.array([[LOUPE_ZOOM, 0, -(cx - half) * LOUPE_ZOOM],
                      [0, LOUPE_ZOOM, -(cy - half) * LOUPE_ZOOM]], np.float32)
        loupe = cv2.warpAffine(self.img, M, (LOUPE_SIZE, LOUPE_SIZE), flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=C_VOID)
        for a, b in self.guides:                     # 辅助线画进放大镜，方便对准交点
            d = np.array([b[0] - a[0], b[1] - a[1]], float)
            nn = np.linalg.norm(d)
            if nn < 1e-6:
                continue
            d /= nn
            f = lambda p: (int(np.clip((p[0] - (cx - half)) * LOUPE_ZOOM, -1e6, 1e6)),
                           int(np.clip((p[1] - (cy - half)) * LOUPE_ZOOM, -1e6, 1e6)))
            cv2.line(loupe, f((a[0] - d[0] * 1e5, a[1] - d[1] * 1e5)),
                     f((a[0] + d[0] * 1e5, a[1] + d[1] * 1e5)), C_GUIDE, 1, cv2.LINE_AA)
        h = LOUPE_SIZE // 2
        cv2.line(loupe, (h, 0), (h, LOUPE_SIZE), (0, 0, 255), 1)
        cv2.line(loupe, (0, h), (LOUPE_SIZE, h), (0, 0, 255), 1)
        x0, y0 = WIN_W - LOUPE_SIZE - 14, 86
        canvas[y0:y0 + LOUPE_SIZE, x0:x0 + LOUPE_SIZE] = loupe
        cv2.rectangle(canvas, (x0 - 1, y0 - 1), (x0 + LOUPE_SIZE, y0 + LOUPE_SIZE), C_HUD, 1)

    # -- 保存 --------------------------------------------------------------
    def save(self, path):
        if len(self.points) < NUM_POINTS:
            raise RuntimeError(
                f"只点了 {len(self.points)}/{NUM_POINTS} 个点就想保存。"
                " 看不见的点请按 k 显式跳过，不要留空 —— 留空会让编号错位。"
            )
        pts, n_out = [], 0
        for i, p in enumerate(self.points):
            if p is None:
                pts.append({"id": i + 1, "name": POINT_NAMES[i], "x": None, "y": None,
                            "labeled": False, "outside_frame": False})
                continue
            outside = not (0 <= p[0] < self.w and 0 <= p[1] < self.h)
            n_out += outside
            pts.append({"id": i + 1, "name": POINT_NAMES[i],
                        "x": round(p[0], 2), "y": round(p[1], 2),
                        "labeled": True, "outside_frame": bool(outside)})
        payload = {
            "source_image": self.image_name,
            "source_resolution": [self.w, self.h],
            "pixel_frame": self.pixel_frame,
            "num_points": NUM_POINTS,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "note": "coordinates are original-image pixels; points constructed by "
                    "extending lines may fall outside [0,W)x[0,H) and that is valid",
            "points": pts,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        n_lab = sum(1 for p in self.points if p is not None)
        print(f"\n已保存 {n_lab}/{NUM_POINTS} 个已标注点（其中 {n_out} 个在画面外）-> {path}")


# ---------------------------------------------------------------------------
def _on_mouse(event, x, y, flags, pk: Picker):
    pk.cursor = pk.to_image(x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        pk.click(*pk.to_image(x, y))
    elif event in (cv2.EVENT_RBUTTONDOWN, cv2.EVENT_MBUTTONDOWN):
        pk._drag = (x, y)
    elif event in (cv2.EVENT_RBUTTONUP, cv2.EVENT_MBUTTONUP):
        pk._drag = None
    elif event == cv2.EVENT_MOUSEMOVE and pk._drag is not None:
        px, py = pk._drag
        pk.pan(x - px, y - py)
        pk._drag = (x, y)
    elif event == cv2.EVENT_MOUSEWHEEL:
        pk.zoom_at(x, y, 1.25 if cv2.getMouseWheelDelta(flags) > 0 else 1 / 1.25)


def _infer_pixel_frame(w, h):
    if (w, h) == VIDEO_RESOLUTION:
        return "video"
    if (w, h) == STILL_RESOLUTION:
        return "still"
    raise ValueError(
        f"底图分辨率是 {w}x{h}，既不是视频的 {VIDEO_RESOLUTION[0]}x{VIDEO_RESOLUTION[1]}，"
        f"也不是标注图的 {STILL_RESOLUTION[0]}x{STILL_RESOLUTION[1]}。\n"
        "  这两者之间的换算关系是实测标定过的（见 pixel_frames.py）；"
        "换成第三种分辨率就没有已知的换算关系，矩阵不能迁移。"
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="在底图上按编号点出 13 个球场特征点")
    ap.add_argument("--image", default="court_plate.png", help="底图文件名")
    ap.add_argument("--clean", default="realcourt2.png", help="用于差分红圈提示的干净图")
    ap.add_argument("--no-hints", action="store_true", help="不显示红圈提示")
    ap.add_argument("--margin", type=float, default=1200,
                    help="初始视野在图像四周留多少空白像素（默认 1200）。"
                         "视野本身无边界，随时可以继续缩小/平移")
    ap.add_argument("--out", default=OUTPUT_JSON, help="输出 json 路径")
    args = ap.parse_args(argv)

    img_path = args.image if os.path.isabs(args.image) else asset(args.image)
    img = imread_unicode(img_path)
    h, w = img.shape[:2]
    pixel_frame = _infer_pixel_frame(w, h)

    hints = []
    if not args.no_hints:
        clean_path = args.clean if os.path.isabs(args.clean) else asset(args.clean)
        if os.path.isfile(clean_path):
            try:
                hints = detect_marks(img, imread_unicode(clean_path))
            except ValueError as e:
                print(f"[提示] 红圈提示不可用：{e}")

    print("=" * 78)
    print(f"底图   : {os.path.basename(img_path)}  ({w}x{h}, pixel_frame='{pixel_frame}')")
    print(f"初始视野: 图像四周各留 {args.margin:g} px 空白（视野无边界，可随意缩小/平移）")
    print(f"输出   : {args.out}")
    print("-" * 78)
    print("按顺序点击（对照 pick_guide.png）：")
    for i, name in enumerate(POINT_NAMES):
        mm = CONFIG.vertices_mm[i]
        coord = "—" if not np.isfinite(mm).all() else f"({mm[0]/1000:.2f}, {mm[1]/1000:.2f}) m"
        print(f"    {i + 1:2d}. {name:<20} {coord}")
    print("-" * 78)
    print("画面外的点：按 g 沿一条线点两下建辅助线，再建第二条，按 x 取交点。")
    print("看不到那片空白就一直滚轮缩小，或按 f 自动适配视野。")
    print("左键=放点  g=辅助线  x=交点  C=清空  u=撤销  k=跳过")
    print("滚轮=缩放  右键拖动=平移  f=自适应  0=复位  s=保存退出  q/ESC=不保存")
    print("=" * 78)

    pk = Picker(img, os.path.basename(img_path), pixel_frame, hints, args.margin)
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, _on_mouse, pk)

    while True:
        cv2.imshow(WINDOW, pk.render())
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            print("\n未保存，已退出。")
            cv2.destroyAllWindows()
            return 1
        if key == ord("s"):
            pk.save(args.out)
            cv2.destroyAllWindows()
            return 0
        if key == ord("g"):
            pk.guide_mode = True
            pk.guide_buf = []
            print("  辅助线模式：沿一条线点两下")
        elif key == ord("x"):
            pk.intersect()
        elif key == ord("C"):
            pk.clear_guides()
        elif key == ord("u"):
            pk.undo()
        elif key == ord("k"):
            pk.skip()
        elif key == ord("f"):
            pk.fit_all()
        elif key == ord("0"):
            pk.reset_view()
        elif key in (ord("+"), ord("=")):
            pk.zoom_at(WIN_W // 2, WIN_H // 2, 1.25)
        elif key in (ord("-"), ord("_"), ord("z")):
            pk.zoom_at(WIN_W // 2, WIN_H // 2, 1 / 1.25)
        elif key == 81:
            pk.pan(60, 0)
        elif key == 83:
            pk.pan(-60, 0)
        elif key == 82:
            pk.pan(0, 60)
        elif key == 84:
            pk.pan(0, -60)

        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            print("\n窗口被关闭，未保存。")
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
