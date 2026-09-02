from datetime import datetime

from typing import Optional

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
    #: 抽帧当时的实际参数。三个字段都是**记录**而非配置：改这里不会改变已经
    #: 抽好的帧，它们存在只为回答「这批帧当时是怎么来的」。
    #:
    #: 没有它们的话，事后只能靠 video_frames.frame_id 的差分倒推 interval，
    #: 而 fps 和解码路径根本无从查起 —— 解码路径尤其要命：装了 ffmpeg 的机器
    #: 数的是转码产物，没装的数的是原视频，同一份代码两套帧号语义。
    fps: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    frame_interval: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    #: 'ffmpeg_transcode' | 'direct'，见 app.services.storage.decode_path()
    decode_path: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
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

#: 由 basketball-v2 的 jersey_train mine 挖出、经 scripts/import_crops.py 导入。
#: 这是 2026-09 之后**唯一**会写进来的两个值，上面三个只可能出现在历史行上。
#:
#: 分成两个值而不是一个 'mine'，因为这两条路的框质量差一个量级：
#:   detector       检测器的 jersey 类框（detect.CLASS_JERSEY == 3）
#:   torso_fallback 检测器一个 jersey 框都没给出来时的几何回退。它不知道球员
#:                  是不是背对镜头，侧身球员上裁到的多半是空球衣。
#: 混成一个值，就没法在训练时把回退框的系统性偏差单独拎出来看。
PREDICTION_SOURCE_MINE_DETECTOR = "mine_detector"
PREDICTION_SOURCE_MINE_TORSO = "mine_torso_fallback"

#: manifest.jsonl 的 source 字段 -> 本表 source 列。
MINE_SOURCE_MAP = {
    "detector": PREDICTION_SOURCE_MINE_DETECTOR,
    "torso_fallback": PREDICTION_SOURCE_MINE_TORSO,
}


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
    #: 这一行对应的裁剪图，相对 data/crops 的路径。只有 mine 导进来的行有值。
    #:
    #: 两个用途，第二个更重要：
    #:   1. 能直接看到「模型推理时**真正**会看到的那张图」，而不是从整帧上
    #:      按框重切一次（重切就又多了一处可能和 crop_jersey 走样的地方）;
    #:   2. 当导入的幂等键。mine 的文件名是 f{frame_idx}__box_{box_idx}.png，
    #:      对 (video, frame_idx, box_idx) 唯一，所以 (video_id, crop_path)
    #:      天然就是那个「重复导入不产生重复行」的键，不用另开一列 box_idx。
    crop_path: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
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
