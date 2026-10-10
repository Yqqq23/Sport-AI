"""从抽取出来的 CSV 里，按球衣号取出某个球员的轨迹。

    from trajectory.paths import load_records, list_players, build_track
    df = load_records("outputs/run1/tracks.csv")
    print(list_players(df))
    tr = build_track(df, jersey=7)

⚠ 必须先平滑再算距离
--------------------
30 fps 下，检测框底边每帧抖几个像素，投影到球场就是几厘米。直接对**逐帧
原始位置**求相邻点距离再累加，噪声会一路累积 —— 一个几乎站着不动的球员
也能「跑」出几百米。所以：

    跑动距离一律用平滑后的轨迹算，原始轨迹只用来画点。

默认用 Savitzky-Golay（保形状、不像滑动平均那样把急停急转抹平）；
scipy 不可用时退回滑动平均，并打日志说明。

⚠ 球衣号会串
------------
OCR 有误识别，identity registry 也可能把两个人并成一个号。所以：
    - 同一帧同一个号出现多条记录时，只保留**离上一帧位置最近**的那条
      （置信度高但瞬移十米的那条一定是别人）
    - 跳变超过 ``max_step_m`` 的点直接丢弃，不参与轨迹
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .court import COURT_L_M, COURT_W_M

LOGGER = logging.getLogger(__name__)

#: 相邻两帧之间球员最多可能移动多少米。人类冲刺约 10 m/s，30fps 下一帧 0.33 m；
#: 留到 2.0 m 是给漏检后的跨帧衔接和投影误差留余量，再大就一定是换人了。
MAX_STEP_M = 2.0

#: 中间断了多少帧以内还算同一段轨迹（会插值补上），超过就断开
MAX_GAP_FRAMES = 15


@dataclass
class Track:
    """一个球员的一条轨迹。"""

    label: str
    frame_idx: np.ndarray          # (N,)
    t: np.ndarray                  # (N,) 秒
    xy_raw: np.ndarray             # (N,2) 米，未平滑
    xy: np.ndarray                 # (N,2) 米，平滑后 —— 算距离用这个
    interpolated: np.ndarray       # (N,) bool，True = 这一点是插值补的
    dropped: int = 0               # 整帧被跳变门槛丢掉的候选数
    outvoted: int = 0              # 同帧有多个候选、竞争落选的那些（多半是误识别成同号的别人）
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.frame_idx)

    # -- 指标 --------------------------------------------------------------
    @property
    def step_m(self) -> np.ndarray:
        """(N-1,) 相邻点位移（米），基于平滑轨迹。"""
        if len(self) < 2:
            return np.zeros(0)
        return np.linalg.norm(np.diff(self.xy, axis=0), axis=1)

    @property
    def distance_m(self) -> float:
        return float(self.step_m.sum())

    @property
    def speed_mps(self) -> np.ndarray:
        """(N-1,) 瞬时速度。时间间隔按实际帧差算，兼容 stride 抽帧。"""
        if len(self) < 2:
            return np.zeros(0)
        dt = np.diff(self.t)
        dt[dt <= 0] = np.nan
        return self.step_m / dt

    @property
    def duration_sec(self) -> float:
        return float(self.t[-1] - self.t[0]) if len(self) >= 2 else 0.0

    def summary(self) -> str:
        sp = self.speed_mps
        sp = sp[np.isfinite(sp)]
        return "\n".join("  " + s for s in [
            f"球员         : {self.label}",
            f"采样点       : {len(self)}  （插值补 {int(self.interpolated.sum())}，"
            f"跳变丢弃 {self.dropped}，同帧落选 {self.outvoted}）",
            f"时间跨度     : {self.duration_sec:.1f} s  "
            f"(frame {int(self.frame_idx[0])} → {int(self.frame_idx[-1])})",
            f"跑动距离     : {self.distance_m:.1f} m",
            f"平均速度     : {np.nanmean(sp) if sp.size else 0:.2f} m/s",
            f"最高速度     : {np.nanpercentile(sp, 99) if sp.size else 0:.2f} m/s  (P99)",
            f"活动范围     : X {self.xy[:,0].min():.1f}–{self.xy[:,0].max():.1f} m, "
            f"Y {self.xy[:,1].min():.1f}–{self.xy[:,1].max():.1f} m",
        ])


# ---------------------------------------------------------------------------
def load_records(path: str | Path) -> pd.DataFrame:
    """读 extract 写出来的 CSV。"""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"轨迹 CSV 不存在: {path}\n"
            "  先跑：python -m trajectory.cli extract --video <视频> --out <目录>"
        )
    df = pd.read_csv(path)
    need = {"frame_idx", "timestamp_sec", "slot", "jersey_number",
            "court_x_m", "court_y_m", "sane"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{path} 缺少字段: {sorted(missing)}")
    df["jersey_number"] = pd.to_numeric(df["jersey_number"], errors="coerce")
    return df


def list_players(df: pd.DataFrame) -> pd.DataFrame:
    """有哪些球员可以画，按出现帧数排序。"""
    ok = df[(df["sane"] == 1) & df["court_x_m"].notna()]
    rows = []
    for num, g in ok[ok["jersey_number"].notna()].groupby("jersey_number"):
        rows.append({
            "key": f"#{int(num)}", "kind": "jersey", "value": int(num),
            "frames": len(g),
            "first_frame": int(g["frame_idx"].min()),
            "last_frame": int(g["frame_idx"].max()),
            "seconds": round(float(g["timestamp_sec"].max() - g["timestamp_sec"].min()), 1),
        })
    for slot, g in ok[ok["jersey_number"].isna()].groupby("slot"):
        if slot == 0:
            continue
        rows.append({
            "key": f"slot:{int(slot)}", "kind": "slot", "value": int(slot),
            "frames": len(g),
            "first_frame": int(g["frame_idx"].min()),
            "last_frame": int(g["frame_idx"].max()),
            "seconds": round(float(g["timestamp_sec"].max() - g["timestamp_sec"].min()), 1),
        })
    if not rows:
        return pd.DataFrame(columns=["key", "kind", "value", "frames",
                                     "first_frame", "last_frame", "seconds"])
    return pd.DataFrame(rows).sort_values("frames", ascending=False).reset_index(drop=True)


def _select(df: pd.DataFrame, jersey=None, slot=None, identity=None) -> tuple[pd.DataFrame, str]:
    if sum(v is not None for v in (jersey, slot, identity)) != 1:
        raise ValueError("jersey / slot / identity 三选一，必须且只能给一个")
    if jersey is not None:
        return df[df["jersey_number"] == float(jersey)], f"#{int(jersey)}"
    if slot is not None:
        return df[df["slot"] == int(slot)], f"slot:{int(slot)}"
    return df[df["identity_key"] == str(identity)], str(identity)


#: 用前多少帧的候选点来定「起始锚点」
SEED_WINDOW_FRAMES = 45


def _seed_anchor(sub: pd.DataFrame, window: int = SEED_WINDOW_FRAMES) -> np.ndarray | None:
    """用开头一段候选点的**中位数**当锚点，用来挑第一个点。

    为什么不能直接挑置信度最高的那条：链式过滤是逐帧往前推的，**第一个点
    一旦选错，整条轨迹就废了** —— 后面所有真点都会因为离这个错误起点太远
    而被当成跳变丢掉。而误识别恰恰常常伴随高置信度（把别人的号看成了这个号，
    OCR 对那个号本身是很确信的）。

    中位数对少数离群点免疫：13 个诱饵混在 1183 个真点里，中位数稳稳落在
    真实轨迹上，于是第一帧就能挑对。
    """
    if sub.empty:
        return None
    frames = np.sort(sub["frame_idx"].unique())
    early = sub[sub["frame_idx"] <= frames[min(window, len(frames)) - 1]]
    pts = early[["court_x_m", "court_y_m"]].to_numpy(dtype=float)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] == 0:
        return None
    return np.median(pts, axis=0)


def _dedup_and_gate(sub: pd.DataFrame, max_step_m: float):
    """一帧只留一条记录，并丢掉瞬移点。

    同一帧出现多条同号记录时，选**离上一个已接受位置最近**的那条。
    单纯挑置信度最高的会被「高置信度的另一个人」骗过去。
    第一帧没有「上一个位置」，改用开头一段的中位数锚点（见 _seed_anchor）。
    """
    sub = sub.sort_values(["frame_idx", "conf"], ascending=[True, False])
    anchor = _seed_anchor(sub)
    frames, xs, ys, ts = [], [], [], []
    dropped = 0
    outvoted = 0
    last = None
    for f, g in sub.groupby("frame_idx", sort=True):
        cand = g[["court_x_m", "court_y_m", "timestamp_sec"]].to_numpy(dtype=float)
        cand = cand[np.isfinite(cand[:, 0]) & np.isfinite(cand[:, 1])]
        if cand.shape[0] == 0:
            continue
        if last is None:
            if anchor is None:
                pick = cand[0]
            else:
                d0 = np.hypot(cand[:, 0] - anchor[0], cand[:, 1] - anchor[1])
                pick = cand[int(np.argmin(d0))]
        else:
            d = np.hypot(cand[:, 0] - last[0], cand[:, 1] - last[1])
            k = int(np.argmin(d))
            # 距离上一个点太远：这一帧不接受（多半是把别人算成了这个号）
            span = max(1, int(f) - int(frames[-1]))
            if d[k] > max_step_m * span:
                dropped += cand.shape[0]
                continue
            pick = cand[k]
        outvoted += cand.shape[0] - 1          # 同帧其余候选：竞争落选
        frames.append(int(f)); xs.append(pick[0]); ys.append(pick[1]); ts.append(pick[2])
        last = (pick[0], pick[1])
    if not frames:
        return (np.zeros(0, int), np.zeros(0), np.zeros((0, 2)), dropped, outvoted)
    return (np.asarray(frames), np.asarray(ts),
            np.stack([np.asarray(xs), np.asarray(ys)], axis=1), dropped, outvoted)


def _fill_gaps(frames, t, xy, max_gap_frames: int):
    """短缺口线性插值补上，长缺口保留（不连线）。返回补齐后的数组 + 插值标记。"""
    if len(frames) < 2:
        return frames, t, xy, np.zeros(len(frames), dtype=bool)
    out_f, out_t, out_xy, out_i = [frames[0]], [t[0]], [xy[0]], [False]
    for k in range(1, len(frames)):
        gap = frames[k] - frames[k - 1]
        if 1 < gap <= max_gap_frames:
            for j in range(1, int(gap)):
                a = j / gap
                out_f.append(frames[k - 1] + j)
                out_t.append(t[k - 1] + a * (t[k] - t[k - 1]))
                out_xy.append(xy[k - 1] + a * (xy[k] - xy[k - 1]))
                out_i.append(True)
        out_f.append(frames[k]); out_t.append(t[k]); out_xy.append(xy[k]); out_i.append(False)
    return (np.asarray(out_f), np.asarray(out_t),
            np.asarray(out_xy), np.asarray(out_i, dtype=bool))


def _smooth(xy: np.ndarray, window: int, poly: int = 2) -> np.ndarray:
    """平滑轨迹。优先 Savitzky-Golay，没有 scipy 就滑动平均。"""
    n = len(xy)
    if n < 5 or window < 3:
        return xy.copy()
    win = min(window if window % 2 == 1 else window + 1, n if n % 2 == 1 else n - 1)
    if win < 5:
        return xy.copy()
    try:
        from scipy.signal import savgol_filter
        return np.stack([savgol_filter(xy[:, 0], win, poly),
                         savgol_filter(xy[:, 1], win, poly)], axis=1)
    except Exception:
        LOGGER.info("scipy 不可用，改用滑动平均平滑（窗口 %d）", win)
        k = np.ones(win) / win
        pad = win // 2
        out = []
        for c in range(2):
            padded = np.pad(xy[:, c], pad, mode="edge")
            out.append(np.convolve(padded, k, mode="valid"))
        return np.stack(out, axis=1)


def build_track(
    df: pd.DataFrame,
    *,
    jersey=None,
    slot=None,
    identity=None,
    smooth_window: int = 15,
    max_step_m: float = MAX_STEP_M,
    max_gap_frames: int = MAX_GAP_FRAMES,
    keep_out_of_court: bool = True,
) -> Track:
    """按球衣号 / 槽位 / identity 取出一条清理好的轨迹。"""
    sub, label = _select(df, jersey, slot, identity)
    sub = sub[sub["sane"] == 1]
    if not keep_out_of_court:
        sub = sub[sub["in_court"] == 1]
    sub = sub[sub["court_x_m"].notna() & sub["court_y_m"].notna()]
    if sub.empty:
        raise ValueError(
            f"{label} 没有任何可用样本。\n"
            "  用 list_players() 看看 CSV 里到底有哪些球员。"
        )

    frames, t, xy_raw, dropped, outvoted = _dedup_and_gate(sub, max_step_m)
    if len(frames) == 0:
        raise ValueError(f"{label} 的样本全部被跳变过滤掉了（max_step_m={max_step_m}）")
    frames, t, xy_raw, interp = _fill_gaps(frames, t, xy_raw, max_gap_frames)
    xy = _smooth(xy_raw, smooth_window)

    return Track(
        label=label, frame_idx=frames, t=t, xy_raw=xy_raw, xy=xy,
        interpolated=interp, dropped=dropped, outvoted=outvoted,
        meta={"smooth_window": smooth_window, "max_step_m": max_step_m,
              "max_gap_frames": max_gap_frames,
              "court_w_m": COURT_W_M, "court_l_m": COURT_L_M},
    )


def occupancy_grid(track: Track, cell_m: float = 0.5) -> tuple[np.ndarray, tuple]:
    """把轨迹打成热区网格。返回 (计数矩阵, extent)。

    网格按**球场米**划分，不是画面像素 —— 这正是米制标定的意义所在。
    """
    nx = int(np.ceil(COURT_W_M / cell_m))
    ny = int(np.ceil(COURT_L_M / cell_m))
    grid = np.zeros((ny, nx), dtype=np.int32)
    for x, y in track.xy:
        ix, iy = int(x / cell_m), int(y / cell_m)
        if 0 <= ix < nx and 0 <= iy < ny:
            grid[iy, ix] += 1
    return grid, (0.0, COURT_W_M, 0.0, COURT_L_M)
