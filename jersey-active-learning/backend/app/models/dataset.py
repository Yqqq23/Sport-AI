from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


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
    __tablename__ = "manual_annotations"

    annotation_id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    prediction_id: Mapped[int] = mapped_column(Integer, index=True, nullable=False)
    video_id: Mapped[int] = mapped_column(ForeignKey("video_assets.id"), index=True, nullable=False)
    frame_id: Mapped[int] = mapped_column(Integer, nullable=False)
    track_id: Mapped[int] = mapped_column(Integer, nullable=False)
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
