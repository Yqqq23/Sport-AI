"""清空标注库和它的数据文件，回到「刚装好」的状态。

    python scripts/reset_db.py                 # 先看会删什么（不改任何东西）
    python scripts/reset_db.py --yes           # 真的删

为什么要清
----------
库里 90 行预测的 source 列 90/90 全是 'mock' —— 不是识别结果，是
mock_recognition.py 按 track_id 算出来的等距色带，号码来自一个七项的写死
查表。基于它产生的 manual_annotations 标的也是那些假框的位置。

这批数据**不能迁移**：框的几何（28%/18%/22%/48%）和推理时那套
（20%/15%/20%/55%）IoU 只有 0.625，留着它等于在新数据集里掺进一批
系统性偏移的样本，而且事后分不出哪些是。

删什么
------
    表      video_assets / video_frames / jersey_predictions /
            manual_annotations / dataset_versions
    文件    data/frames、data/crops、data/exports 下的内容

保留什么
--------
    表结构和迁移状态（只 DELETE，不 DROP，schema 原样留着）
    data/videos 下的原始视频        —— 删了就得重新上传，且它们不是脏数据
    磁盘上独立那份 jersey-active-learning  —— 本脚本不碰它

删之前一定备份成 jersey_active_learning.backup.<时间戳>.db，不可跳过。
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

#: 顺序有意义：先删引用别人的，再删被引用的，避免外键约束报错。
TABLES_IN_DELETE_ORDER = (
    "manual_annotations",
    "jersey_predictions",
    "video_frames",
    "processing_jobs",
    "dataset_versions",
    "video_assets",
)

#: 这些目录下的内容全部清掉。data/videos 不在列表里 —— 见模块 docstring。
DATA_SUBDIRS = ("frames", "crops", "exports")


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return row is not None


def _row_counts(conn: sqlite3.Connection) -> dict[str, int]:
    counts: dict[str, int] = {}
    for table in TABLES_IN_DELETE_ORDER:
        if _table_exists(conn, table):
            counts[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        else:
            counts[table] = -1        # -1 表示表不存在，区别于 0 行
    return counts


def _count_files(root: Path) -> tuple[int, int]:
    """(文件数, 总字节数)。目录不存在返回 (0, 0)。"""
    if not root.is_dir():
        return 0, 0
    files = [p for p in root.rglob("*") if p.is_file()]
    return len(files), sum(p.stat().st_size for p in files)


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}GB"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="清空标注库（默认只预览）")
    parser.add_argument("--db", default=None, help="库文件路径，不给就用 DATABASE_URL")
    parser.add_argument("--data-dir", default=None, help="数据目录，不给就用 DATA_DIR")
    parser.add_argument("--yes", action="store_true", help="真的执行。不给就只打印会删什么")
    parser.add_argument("--keep-files", action="store_true",
                        help="只清表，不动 data/frames 等目录")
    args = parser.parse_args(argv)

    from app.core.config import settings

    if args.db:
        db_path = Path(args.db).resolve()
    else:
        url = settings.database_url
        if not url.startswith("sqlite"):
            print(f"[FAIL] 只支持 sqlite，当前 DATABASE_URL = {url}")
            return 1
        raw = url.split("///", 1)[-1]
        db_path = (_BACKEND / raw).resolve() if not Path(raw).is_absolute() else Path(raw)

    data_dir = Path(args.data_dir).resolve() if args.data_dir else (_BACKEND / settings.data_dir).resolve()

    if not db_path.is_file():
        print(f"[FAIL] 库文件不存在: {db_path}")
        return 1

    print("=" * 70)
    print(f"库文件  : {db_path}")
    print(f"数据目录: {data_dir}")
    print(f"模式    : {'执行' if args.yes else '预览（不改任何东西）'}")
    print("=" * 70)

    conn = sqlite3.connect(str(db_path))
    try:
        before = _row_counts(conn)
    finally:
        conn.close()

    print("\n表行数（清理前）")
    for table, count in before.items():
        print(f"  {table:<22} {'表不存在' if count < 0 else count}")

    file_stats = {name: _count_files(data_dir / name) for name in DATA_SUBDIRS}
    print("\n数据文件")
    for name, (n, size) in file_stats.items():
        print(f"  data/{name:<17} {n} 个文件, {_human(size)}")
    kept_n, kept_size = _count_files(data_dir / "videos")
    print(f"  data/{'videos':<17} {kept_n} 个文件, {_human(kept_size)}   <- 保留，不删")

    if not args.yes:
        print("\n以上都没有被改动。确认无误后加 --yes 再跑一次。")
        return 0

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = db_path.with_name(f"{db_path.stem}.backup.{stamp}.db")
    shutil.copy2(db_path, backup)
    print(f"\n已备份: {backup}")

    conn = sqlite3.connect(str(db_path))
    try:
        # 外键约束按顺序删已经够了，但显式关掉更稳：有些库是在
        # PRAGMA foreign_keys=ON 之前建的，约束状态不一定一致。
        conn.execute("PRAGMA foreign_keys = OFF")
        for table in TABLES_IN_DELETE_ORDER:
            if before[table] >= 0:
                conn.execute(f"DELETE FROM {table}")
        # 自增 id 归零，让新导入的 video_id 从 1 开始。表不存在时静默跳过。
        if _table_exists(conn, "sqlite_sequence"):
            placeholders = ",".join("?" for _ in TABLES_IN_DELETE_ORDER)
            conn.execute(
                f"DELETE FROM sqlite_sequence WHERE name IN ({placeholders})",
                TABLES_IN_DELETE_ORDER,
            )
        conn.commit()
        conn.execute("VACUUM")
        after = _row_counts(conn)
    finally:
        conn.close()

    removed_files = 0
    if not args.keep_files:
        for name in DATA_SUBDIRS:
            target = data_dir / name
            if not target.is_dir():
                continue
            for child in target.iterdir():
                if child.is_dir():
                    removed_files += _count_files(child)[0]
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    removed_files += 1
                    child.unlink(missing_ok=True)
            target.mkdir(parents=True, exist_ok=True)

    print("\n" + "-" * 70)
    print(f"{'表':<22}{'清理前':>10}{'清理后':>10}")
    print("-" * 70)
    for table in TABLES_IN_DELETE_ORDER:
        if before[table] < 0:
            print(f"{table:<22}{'表不存在':>10}{'-':>10}")
        else:
            print(f"{table:<22}{before[table]:>10}{after[table]:>10}")
    print("-" * 70)
    print(f"删除文件: {removed_files}" if not args.keep_files else "数据文件: 按 --keep-files 保留")
    print(f"备份    : {backup}")
    print("\n下一步:")
    print("  1) 重新上传视频（网页 Upload，会写入 fps / frame_interval / decode_path）")
    print("  2) jersey_train mine --video ... --annotation-db ... --video-id <新 id>")
    print("  3) python scripts/import_crops.py --manifest ... --video-id <新 id>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
