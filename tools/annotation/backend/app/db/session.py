from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.core.config import settings


class Base(DeclarativeBase):
    pass


connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def ensure_sqlite_schema() -> None:
    if not settings.database_url.startswith("sqlite"):
        return
    with engine.begin() as conn:
        existing = {row[1] for row in conn.execute(text("PRAGMA table_info(video_frames)"))}
        if existing:
            if "image_width" not in existing:
                conn.execute(text("ALTER TABLE video_frames ADD COLUMN image_width INTEGER"))
            if "image_height" not in existing:
                conn.execute(text("ALTER TABLE video_frames ADD COLUMN image_height INTEGER"))

        annotation_columns = {row[1] for row in conn.execute(text("PRAGMA table_info(manual_annotations)"))}
        if annotation_columns and "deleted_at" not in annotation_columns:
            conn.execute(text("ALTER TABLE manual_annotations ADD COLUMN deleted_at DATETIME"))
        if annotation_columns and "class_id" not in annotation_columns:
            # Backfill to the jersey class. Every annotation that existed before
            # this column was added was a jersey box - the tool could not express
            # anything else - so this is a faithful backfill, not a guess.
            from app.services.basketball_classes import DEFAULT_CLASS_ID

            conn.execute(
                text(
                    "ALTER TABLE manual_annotations "
                    f"ADD COLUMN class_id INTEGER NOT NULL DEFAULT {int(DEFAULT_CLASS_ID)}"
                )
            )

        prediction_columns = {row[1] for row in conn.execute(text("PRAGMA table_info(jersey_predictions)"))}
        if prediction_columns and "crop_path" not in prediction_columns:
            # NULLable：历史行（mock / yolo_preannotation）从来没有对应的裁剪
            # 图文件，给它们编一个路径出来只会让下游去读一个不存在的文件。
            conn.execute(text("ALTER TABLE jersey_predictions ADD COLUMN crop_path VARCHAR(1024)"))
        if prediction_columns and "source" not in prediction_columns:
            # Existing rows cannot be attributed after the fact, so they are
            # marked 'unknown' rather than guessed at. The UI treats unknown as
            # "not verified real" and says so.
            from app.models.dataset import PREDICTION_SOURCE_UNKNOWN

            conn.execute(
                text(
                    "ALTER TABLE jersey_predictions ADD COLUMN source "
                    f"VARCHAR(32) NOT NULL DEFAULT '{PREDICTION_SOURCE_UNKNOWN}'"
                )
            )

        # 抽帧参数的留痕。全部 NULLable：老视频抽帧时这些值没被记下来，
        # 填一个「合理的默认值」等于伪造记录 —— NULL 才是诚实的「不知道」。
        video_columns = {row[1] for row in conn.execute(text("PRAGMA table_info(video_assets)"))}
        if video_columns:
            for column_name, column_type in {
                "fps": "FLOAT",
                "frame_interval": "INTEGER",
                "decode_path": "VARCHAR(32)",
            }.items():
                if column_name not in video_columns:
                    conn.execute(text(f"ALTER TABLE video_assets ADD COLUMN {column_name} {column_type}"))

        dataset_version_columns = {row[1] for row in conn.execute(text("PRAGMA table_info(dataset_versions)"))}
        if dataset_version_columns:
            missing_columns = {
                "temporal_block_size": "INTEGER NOT NULL DEFAULT 30",
                "strict_split": "INTEGER NOT NULL DEFAULT 0",
                "source_data_fingerprint": "VARCHAR(64) NOT NULL DEFAULT ''",
            }
            for column_name, column_type in missing_columns.items():
                if column_name not in dataset_version_columns:
                    conn.execute(text(f"ALTER TABLE dataset_versions ADD COLUMN {column_name} {column_type}"))
