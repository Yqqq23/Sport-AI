"""把已有的 SQLite 库升到当前 schema。可重复跑。

    python scripts/migrate_db.py
    python scripts/migrate_db.py --db backend/jersey_active_learning.db

做什么
------
只加列，不改也不删已有数据。实际的 ALTER 语句在
``app/db/session.py::ensure_sqlite_schema()`` —— 那个函数在后端启动时也会跑，
所以正常情况下你**不需要**手动执行本脚本；它存在是为了两种场合：

  1. 想在启动服务之前先确认迁移能过；
  2. 要看清楚这次到底加了哪几列（启动时那次是静默的）。

新加的三列（2026-09）都在 video_assets 上，且全部 NULLable：

    fps            抽帧当时读到的帧率
    frame_interval 抽帧当时用的间隔（每隔几帧存一张）
    decode_path    'ffmpeg_transcode' | 'direct'

老视频这三列会是 NULL。这是有意的：那些帧抽的时候没人记录这些值，
填一个「合理的默认值」等于伪造记录。要用 frame_interval 的地方
（比如 mine 定 stride）会自己退回到帧号差分，并说明它退回了。
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

TRACKED_TABLES = (
    "video_assets",
    "video_frames",
    "jersey_predictions",
    "manual_annotations",
    "dataset_versions",
)


def _columns(db_path: Path, table: str) -> list[str]:
    conn = sqlite3.connect(str(db_path))
    try:
        return [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]
    finally:
        conn.close()


def _snapshot(db_path: Path) -> dict[str, list[str]]:
    return {table: _columns(db_path, table) for table in TRACKED_TABLES}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="升级标注库 schema（只加列）")
    parser.add_argument("--db", default=None,
                        help="库文件路径。不给就用 app.core.config 里的 DATABASE_URL")
    parser.add_argument("--no-backup", action="store_true",
                        help="不先备份。只在库本身是一次性测试库时才用")
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

    if not db_path.is_file():
        print(f"[FAIL] 库文件不存在: {db_path}")
        print("       后端第一次启动时会自动建库，先跑一次服务。")
        return 1

    print(f"库文件: {db_path}")

    if not args.no_backup:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = db_path.with_suffix(f".pre-migrate.{stamp}.db")
        shutil.copy2(db_path, backup)
        print(f"已备份: {backup}")

    before = _snapshot(db_path)

    # ensure_sqlite_schema 读的是 settings.database_url，不是上面那个 db_path。
    # 两者不一致时迁移会落到别的库上，所以这里对齐一次再调用。
    settings.database_url = f"sqlite:///{db_path.as_posix()}"
    import app.db.session as session_module
    from sqlalchemy import create_engine

    session_module.engine = create_engine(
        settings.database_url, connect_args={"check_same_thread": False}
    )
    session_module.ensure_sqlite_schema()

    after = _snapshot(db_path)

    print("-" * 66)
    changed = False
    for table in TRACKED_TABLES:
        added = [c for c in after[table] if c not in before[table]]
        if added:
            changed = True
            print(f"  {table}: +{len(added)} 列 -> {', '.join(added)}")
        elif not after[table]:
            print(f"  {table}: 表不存在（跳过）")
        else:
            print(f"  {table}: 无变化（{len(after[table])} 列）")
    print("-" * 66)
    print("schema 已是最新" if not changed else "迁移完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
