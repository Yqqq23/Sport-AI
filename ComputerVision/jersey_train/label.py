"""给挖出来的 crop 打标签。

    python -m jersey_train.cli label --src data/mined --out data/mined_labelled

界面
----
一屏显示一个槽位的一批 crop（默认 5×8 = 40 张）。同一个槽位在一段时间内
就是同一个球员，所以**看清楚是几号，输入数字，回车 —— 这一批全部归档**。
这是让人工标注成本可控的关键：几百张图往往只需要十几次按键。

操作
    数字键 0-9   输入号码
    退格         删掉最后一位
    回车         把当前这批全部标成输入的号码
    s            跳过这一批（看不清就跳过，别猜 —— 错标比缺数据糟得多）
    d            这一批不是球衣号（裁到脸/手/背景了），扔进 _reject
    ←  →         上一批 / 下一批
    q            退出（已标的不丢）

产出
    <out>/7/slot_003_f001234.png ...
    <out>/12/...
    <out>/_reject/...
就是 ``data.load_from_folders`` 认的结构，可以直接拿去训练。

⚠ 看不清就跳过
--------------
标错一张的破坏力远大于少一张。分类器会老老实实把错标当真值学，
而且这种错误在验证集上还会**自我一致**（因为验证集也来自同样的错标），
所以你根本看不出来。拿不准就按 s。
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)

WINDOW = "label jersey crops  |  digits=number  enter=apply  s=skip  d=reject  arrows=nav  q=quit"
GRID_COLS, GRID_ROWS = 8, 5
CELL = 110
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def _imread(path: Path):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def _batches(src: Path, batch_size: int) -> list[tuple[str, list[Path]]]:
    """按槽位目录切成一批一批，每批来自同一个槽位。"""
    out: list[tuple[str, list[Path]]] = []
    for d in sorted(p for p in src.iterdir() if p.is_dir()):
        if d.name.startswith("_"):
            continue
        files = sorted(f for f in d.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES)
        for i in range(0, len(files), batch_size):
            out.append((d.name, files[i:i + batch_size]))
    return out


def _render(group: str, files: list[Path], typed: str, idx: int, total: int,
            done: int) -> np.ndarray:
    h = GRID_ROWS * CELL + 96
    w = GRID_COLS * CELL
    canvas = np.full((h, w, 3), 32, np.uint8)
    for k, f in enumerate(files[:GRID_COLS * GRID_ROWS]):
        img = _imread(f)
        if img is None:
            continue
        img = cv2.resize(img, (CELL - 8, CELL - 8), interpolation=cv2.INTER_CUBIC)
        r, c = divmod(k, GRID_COLS)
        y0, x0 = 88 + r * CELL + 4, c * CELL + 4
        canvas[y0:y0 + CELL - 8, x0:x0 + CELL - 8] = img

    cv2.rectangle(canvas, (0, 0), (w, 84), (20, 20, 20), -1)
    cv2.putText(canvas, f"{group}   batch {idx + 1}/{total}   {len(files)} crops",
                (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235, 235, 235), 2, cv2.LINE_AA)
    label = typed if typed else "_"
    cv2.putText(canvas, f"number: {label}", (14, 62), cv2.FONT_HERSHEY_SIMPLEX,
                0.85, (60, 220, 60) if typed else (120, 120, 120), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"labelled {done}   |  enter=apply  s=skip  d=reject  q=quit",
                (280, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    return canvas


def run(src: str | Path, out: str | Path, *, batch_size: int = GRID_COLS * GRID_ROWS,
        move: bool = False) -> int:
    src, out = Path(src), Path(out)
    if not src.is_dir():
        raise FileNotFoundError(f"源目录不存在: {src}\n  先跑 mine 子命令")
    out.mkdir(parents=True, exist_ok=True)

    batches = _batches(src, batch_size)
    if not batches:
        raise RuntimeError(f"{src} 里没有 slot_* 目录。先跑 mine 子命令。")

    print("=" * 74)
    print(f"源目录 : {src}")
    print(f"输出   : {out}")
    print(f"共 {len(batches)} 批，每批最多 {batch_size} 张")
    print("-" * 74)
    print("数字键输入号码 -> 回车归档整批；s 跳过；d 标为非球衣号；← → 翻批；q 退出")
    print("看不清就按 s。错标比缺数据糟得多，而且验证集上看不出来。")
    print("=" * 74)

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    i, typed, done = 0, "", 0

    def archive(files: list[Path], label: str) -> int:
        d = out / label
        d.mkdir(parents=True, exist_ok=True)
        n = 0
        for f in files:
            dst = d / f"{f.parent.name}_{f.name}"
            try:
                shutil.move(str(f), str(dst)) if move else shutil.copy2(str(f), str(dst))
                n += 1
            except Exception as exc:
                LOGGER.warning("归档失败 %s: %s", f, exc)
        return n

    while 0 <= i < len(batches):
        group, files = batches[i]
        cv2.imshow(WINDOW, _render(group, files, typed, i, len(batches), done))
        key = cv2.waitKey(20) & 0xFF
        if key == 255:
            continue
        if key == ord("q"):
            break
        if ord("0") <= key <= ord("9"):
            if len(typed) < 2:
                typed += chr(key)
        elif key in (8, 127):                       # backspace
            typed = typed[:-1]
        elif key in (13, 10):                       # enter
            if not typed:
                print("  ! 先输入号码")
                continue
            label = typed.lstrip("0") or None
            if label is None:
                print("  ! '0'/'00' 上游 clean_jersey_number 会拒绝，不能作为类别")
                typed = ""
                continue
            n = archive(files, label)
            done += n
            print(f"  {group} 批 {i+1}: {n} 张 -> #{label}")
            typed = ""
            i += 1
        elif key == ord("s"):
            print(f"  {group} 批 {i+1}: 跳过")
            typed = ""
            i += 1
        elif key == ord("d"):
            n = archive(files, "_reject")
            print(f"  {group} 批 {i+1}: {n} 张 -> _reject")
            typed = ""
            i += 1
        elif key == 83:                             # right
            typed = ""
            i = min(i + 1, len(batches) - 1)
        elif key == 81:                             # left
            typed = ""
            i = max(i - 1, 0)

        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()
    print(f"\n共标注 {done} 张 -> {out}")
    counts = {d.name: len(list(d.glob('*'))) for d in sorted(out.iterdir()) if d.is_dir()}
    if counts:
        print("每类数量:", counts)
        thin = {k: v for k, v in counts.items() if not k.startswith("_") and v < 30}
        if thin:
            print(f"⚠ 这些类样本还很少: {thin}  —— 建议再挖一段视频补上")
    return done
