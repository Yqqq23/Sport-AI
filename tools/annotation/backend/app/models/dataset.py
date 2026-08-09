from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base
from app.services.basketball_classes import DEFAULT_CLASS_ID


class VideoAsset(Base):
    __tablename__ = "video_assets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    path: Mapped[str] = mapped_column(String(1024), nullable=False)
    status: Mapped[str] = mapped_column(String(50), default="uploaded")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    frames: Mapped[list["VideoFrame"]] = relationship(
        back_populates="video",
        cascade="all, delete-orphan",
        order_by="VideoFrame.frame_id",
    )
    predictions: Mapped[list["JerseyPrediction"]] = relationship(
        back_populates="video",
        cascade="all, delete-orphan",
        order_by="JerseyPrediction.frame_id",
    )


class VideoFrame(Base):
    __tablename__ = "video_frames"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    video_id: Mapped[int] = mapped_column(ForeignKey("video_assets.id"), index=True, nullable=False)
    frame_id: Mapped[int] = mapped_column(Integer, nullable=False)
    frame_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    image_width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    image_height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    video: Mapped[VideoAsset] = relationship(back_populates="frames")
    predictions: Mapped[list["JerseyPrediction"]] = relationship(back_populates="frame", cascade="all, delete-orphan")


#: Value of JerseyPrediction.source meaning "these boxes and numbers are FAKE".
#: Kept as a stored column rather than inferred, so a mock prediction stays
#: identifiable after it has been written to the database and reloaded.
PREDICTION_SOURCE_MOCK = "mock"
PREDICTION_SOURCE_YOLO = "yolo_preannotation"
PREDICTION_SOURCE_REAL = "real_recognition"
PREDICTION_SOURCE_UNKNOWN = "unknown"


class JerseyPrediction(Base):
    __tablename__ = "jersey_predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    video_id: Mapped[int] = mapped_column(ForeignKey("video_assets.id"), index=True, nullable=False)
    frame_row_id: Mapped[int] = mapped_column(ForeignKey("video_frames.id"), index=True, nullable=False)
    frame_id: Mapped[int] = mapped_column(Integer, nullable=False)
    track_id: Mapped[int] = mapped_column(Integer, nullable=False)
    player_bbox: Mapped[str] = mapped_column(String(255), nullable=False)
    jersey_bbox: Mapped[str] = mapped_column(String(255), nullable=False)
    predicted_number: Mapped[str] = mapped_column(String(16), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    #: Which producer made this row. The whole reason this column exists is that
    #: mock_recognition emits plausible-looking boxes and confidences that are
    #: entirely invented; without a stored marker they are indistinguishable from
    #: real detections once they are in the database, and a reviewer can spend an
    #: afternoon "correcting" pure noise.
    source: Mapped[str] = mapped_column(String(32), nullable=False, default=PREDICTION_SOURCE_UNKNOWN)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    video: Mapped[VideoAsset] = relationship(back_populates="predictions")
    frame: Mapped[VideoFrame] = relationship(back_populates="predictions")


class ProcessingJob(Base):
    __tablename__ = "processing_jobs"

    job_id: Mapped[str] = mapped_column(String(64), primary_key=True, index=True)
    job_type: Mapped[str] = mapped_column(String(64), nullable=False)
    video_id: Mapped[int | None] = mapped_column(ForeignKey("video_assets.id"), index=True, nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="queued")
    progress: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    message: Mapped[str | None] = mapped_column(String(255), nullable=True)
    result_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ManualAnnotation(Base):
    """One human-drawn box on one frame.

    Originally every row was implicitly a jersey box, so the table carried no
    class column at all. It now stores an explicit ``class_id`` from the
    basketball-v2 scheme (see app.services.basketball_classes), because a
    5-class dataset needs several boxes of DIFFERENT classes on the same frame.

    ``prediction_id`` is the link back to a YOLO pre-annotation. Two disjoint
    ranges are in use, and the sign is what tells them apart:

      >= 0   a real JerseyPrediction row this annotation corrects. At most one
             annotation per prediction, enforced by the upsert in save_annotation.
      <  0   a box drawn from scratch with no prediction behind it. These ids are
             allocated to be unique per box, NOT per frame, which is what allows
             many boxes on one frame. See routes._allocate_manual_prediction_id.

    ``corrected_number`` only ever applies to the jersey class; every other class
    stores NULL. The number is not a detection class and never reaches a YOLO
    label file - it is exported separately as jersey_numbers.json.
    """

    __tablename__ = "manual_annotations"

    annotation_id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    prediction_id: Mapped[int] = mapped_column(Integer, index=True, nullable=False)
    video_id: Mapped[int] = mapped_column(ForeignKey("video_assets.id"), index=True, nullable=False)
    frame_id: Mapped[int] = mapped_column(Integer, nullable=False)
    track_id: Mapped[int] = mapped_column(Integer, nullable=False)
    #: Index into the basketball-v2 class scheme (train/basketball.yaml).
    #: Defaults to the jersey class: every row that existed before this column
    #: was added was a jersey box, so that is the only correct backfill value.
    class_id: Mapped[int] = mapped_column(Integer, nullable=False, default=DEFAULT_CLASS_ID)
    image_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    image_url: Mapped[str] = mapped_column(String(1024), nullable=False)
    original_bbox: Mapped[str] = mapped_column(Text, nullable=False)
    corrected_bbox: Mapped[str] = mapped_column(Text, nullable=False)
    original_number: Mapped[str] = mapped_column(String(16), nullable=False)
    corrected_number: Mapped[str | None] = mapped_column(String(16), nullable=True)
    label_status: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"

    dataset_version_id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    version_name: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    dataset_format: Mapped[str] = mapped_column(String(64), nullable=False)
    task_type: Mapped[str] = mapped_column(String(64), nullable=False)
    split_strategy: Mapped[str] = mapped_column(String(32), nullable=False)
    random_seed: Mapped[int] = mapped_column(Integer, nullable=False)
    train_ratio: Mapped[float] = mapped_column(Float, nullable=False)
    val_ratio: Mapped[float] = mapped_column(Float, nullable=False)
    test_ratio: Mapped[float] = mapped_column(Float, nullable=False)
    temporal_block_size: Mapped[int] = mapped_column(Integer, nullable=False, default=30)
    strict_split: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    source_video_ids: Mapped[str] = mapped_column(Text, nullable=False)
    source_annotation_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    exported_image_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    exported_annotation_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    train_image_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    val_image_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    test_image_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    train_annotation_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    val_annotation_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    test_annotation_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    export_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    metadata_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="building")
    validation_result_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_data_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
