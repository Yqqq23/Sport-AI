from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, model_validator


LABEL_STATUS_VALUES = {"correct", "wrong_bbox", "wrong_number", "unclear"}


class VideoAssetCreate(BaseModel):
    filename: str
    path: str


class VideoAssetRead(BaseModel):
    id: int
    video_id: int | None = None
    filename: str
    original_filename: str | None = None
    path: str
    status: str
    frame_count: int | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class VideoFrameRead(BaseModel):
    id: int
    video_id: int
    frame_id: int
    frame_path: str
    image_width: int | None = None
    image_height: int | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class UploadVideoResponse(BaseModel):
    video_id: int
    original_filename: str
    saved_video_path: str
    frame_count: int
    elapsed_seconds: float
    status: str


class JerseyPredictionRead(BaseModel):
    prediction_id: int
    video_id: int
    frame_id: int
    track_id: int
    player_bbox: list[int]
    jersey_bbox: list[int]
    predicted_number: str
    confidence: float
    frame_path: str | None = None
    image_width: int | None = None
    image_height: int | None = None
    created_at: datetime


class MockRecognitionResponse(BaseModel):
    video_id: int
    prediction_count: int
    status: str


class RecognitionResponse(BaseModel):
    video_id: int
    prediction_count: int
    status: str
    message: str | None = None


class RecognitionSummaryResponse(BaseModel):
    prediction_count: int
    candidate_count: int
    low_confidence_count: int
    inconsistency_count: int
    missing_number_count: int
    average_confidence: float
    status: str


class ReviewItemRead(BaseModel):
    review_id: str
    prediction_id: int | None = None
    video_id: int
    frame_id: int
    track_id: int
    image_url: str | None = None
    frame_path: str | None = None
    image_width: int | None = None
    image_height: int | None = None
    player_bbox: list[int] | None = None
    jersey_bbox: list[int] | None = None
    predicted_number: str = ""
    confidence: float = 0.0
    annotation_id: int | None = None
    corrected_bbox: list[int] | None = None
    corrected_number: str | None = None
    label_status: str | None = None


class ReviewItemsResponse(BaseModel):
    total_count: int
    items: list[ReviewItemRead]
    message: str | None = None


class FailureFrameRead(JerseyPredictionRead):
    reason: str


class FailureFramesResponse(BaseModel):
    total_count: int
    items: list[FailureFrameRead]


class TemporalContext(BaseModel):
    previous_number: str
    current_number: str
    next_number: str


class InconsistentFrameRead(JerseyPredictionRead):
    reason: str
    context: TemporalContext | None = None


class InconsistentFramesResponse(BaseModel):
    total_count: int
    items: list[InconsistentFrameRead]


class AnnotationCandidateRead(JerseyPredictionRead):
    reasons: list[str]
    context: TemporalContext | None = None
    image_url: str | None = None


class AnnotationCandidatesResponse(BaseModel):
    total_count: int
    items: list[AnnotationCandidateRead]
    message: str | None = None


class ProcessingJobRead(BaseModel):
    job_id: str
    job_type: str
    video_id: int | None = None
    status: str
    progress: int
    message: str | None = None
    result_json: Any | None = None
    error_message: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None


class JobQueuedResponse(BaseModel):
    job_id: str
    status: str
    poll_url: str


class ManualAnnotationCreate(BaseModel):
    prediction_id: int | None = None
    video_id: int
    frame_id: int
    track_id: int
    image_path: str
    image_url: str
    original_bbox: list[int] = Field(min_length=4, max_length=4)
    corrected_bbox: list[int] = Field(min_length=4, max_length=4)
    original_number: str
    corrected_number: str | None = None
    label_status: str
    image_width: int | None = None
    image_height: int | None = None

    @model_validator(mode="after")
    def validate_payload(self):
        if self.label_status not in LABEL_STATUS_VALUES:
            raise ValueError("label_status must be one of: correct, wrong_bbox, wrong_number, unclear")

        if len(self.corrected_bbox) != 4:
            raise ValueError("corrected_bbox must be [x1, y1, x2, y2]")
        x1, y1, x2, y2 = self.corrected_bbox
        if x1 >= x2 or y1 >= y2:
            raise ValueError("corrected_bbox must satisfy x1 < x2 and y1 < y2")
        if min(self.corrected_bbox) < 0:
            raise ValueError("corrected_bbox must stay within image boundaries")
        if self.image_width is not None and x2 > self.image_width:
            raise ValueError("corrected_bbox x2 must stay inside image width")
        if self.image_height is not None and y2 > self.image_height:
            raise ValueError("corrected_bbox y2 must stay inside image height")

        cleaned_number = (self.corrected_number or "").strip()
        if self.label_status == "unclear":
            self.corrected_number = cleaned_number or None
        else:
            if not cleaned_number:
                raise ValueError("corrected_number cannot be empty unless label_status is 'unclear'")
            self.corrected_number = cleaned_number

        self.original_number = self.original_number.strip()

        return self


class ManualAnnotationRead(BaseModel):
    annotation_id: int
    prediction_id: int
    video_id: int
    frame_id: int
    track_id: int
    image_path: str
    image_url: str
    original_bbox: list[int]
    corrected_bbox: list[int]
    original_number: str
    corrected_number: str | None = None
    label_status: str
    created_at: datetime
    deleted_at: datetime | None = None


class ManualAnnotationSaveResponse(BaseModel):
    annotation_id: int
    status: str


class ManualAnnotationUpdate(BaseModel):
    corrected_bbox: list[int] = Field(min_length=4, max_length=4)
    corrected_number: str | None = None
    label_status: str
    image_width: int | None = None
    image_height: int | None = None

    @model_validator(mode="after")
    def validate_payload(self):
        if self.label_status not in LABEL_STATUS_VALUES:
            raise ValueError("label_status must be one of: correct, wrong_bbox, wrong_number, unclear")

        x1, y1, x2, y2 = self.corrected_bbox
        if x1 >= x2 or y1 >= y2:
            raise ValueError("corrected_bbox must satisfy x1 < x2 and y1 < y2")
        if min(self.corrected_bbox) < 0:
            raise ValueError("corrected_bbox must stay within image boundaries")
        if self.image_width is not None and x2 > self.image_width:
            raise ValueError("corrected_bbox x2 must stay inside image width")
        if self.image_height is not None and y2 > self.image_height:
            raise ValueError("corrected_bbox y2 must stay inside image height")

        cleaned_number = (self.corrected_number or "").strip()
        if self.label_status == "unclear":
            self.corrected_number = cleaned_number or None
        else:
            if not cleaned_number:
                raise ValueError("corrected_number cannot be empty unless label_status is 'unclear'")
            self.corrected_number = cleaned_number
        return self


class AnnotationProgressResponse(BaseModel):
    video_id: int
    total_candidates: int
    annotated_count: int
    remaining_count: int
    progress_percent: float


class InternalExportResponse(BaseModel):
    export_id: str
    video_id: int
    annotation_count: int
    export_path: str
    files_generated: list[str]


class VideoExportRead(BaseModel):
    export_id: str
    video_id: int
    export_path: str
    created_at: str
    files_generated: list[str]


class ExportDownloadResponse(BaseModel):
    export_id: str
    export_path: str


class CocoExportResponse(BaseModel):
    export_id: str
    video_id: int
    dataset_format: str
    image_count: int
    annotation_count: int
    skipped_unclear_count: int
    skipped_invalid_bbox_count: int
    skipped_missing_image_count: int
    export_path: str
    files_generated: list[str]


class CocoExportRead(BaseModel):
    export_id: str
    video_id: int
    dataset_format: str
    image_count: int
    annotation_count: int
    skipped_unclear_count: int
    skipped_invalid_bbox_count: int
    skipped_missing_image_count: int
    export_path: str
    created_at: str


class YoloExportResponse(BaseModel):
    export_id: str
    video_id: int
    dataset_format: str
    target_project: str
    detected_framework: str
    detected_model_version: str
    image_count: int
    annotation_count: int
    label_file_count: int
    skipped_unclear_count: int
    skipped_invalid_bbox_count: int
    skipped_missing_image_count: int
    sport_cv_compatible: bool
    export_path: str
    files_generated: list[str]


class YoloExportRead(BaseModel):
    export_id: str
    video_id: int
    dataset_format: str
    target_project: str
    detected_framework: str
    detected_model_version: str
    image_count: int
    annotation_count: int
    label_file_count: int
    skipped_unclear_count: int
    skipped_invalid_bbox_count: int
    skipped_missing_image_count: int
    sport_cv_compatible: bool
    export_path: str
    created_at: str


DATASET_SPLIT_STRATEGIES = {"video", "track", "temporal_block"}


class DatasetVersionCreate(BaseModel):
    version_name: str
    description: str | None = None
    video_ids: list[int] = Field(min_length=1)
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42
    split_strategy: str = "video"
    temporal_block_size: int = 30
    strict_split: bool = False

    @model_validator(mode="after")
    def validate_payload(self):
        if self.split_strategy not in DATASET_SPLIT_STRATEGIES:
            raise ValueError("split_strategy must be one of: video, track, temporal_block")
        if self.train_ratio <= 0:
            raise ValueError("train_ratio must be > 0")
        if self.val_ratio < 0 or self.test_ratio < 0:
            raise ValueError("val_ratio and test_ratio must be >= 0")
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
        if self.temporal_block_size <= 0:
            raise ValueError("temporal_block_size must be > 0")
        if len(set(self.video_ids)) != len(self.video_ids):
            raise ValueError("video_ids must be unique")
        return self


class DatasetVersionRead(BaseModel):
    dataset_version_id: int
    version_name: str
    description: str | None = None
    dataset_format: str
    task_type: str
    split_strategy: str
    random_seed: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    temporal_block_size: int
    strict_split: bool
    source_video_ids: list[int]
    source_annotation_count: int
    exported_image_count: int
    exported_annotation_count: int
    train_image_count: int
    val_image_count: int
    test_image_count: int
    train_annotation_count: int
    val_annotation_count: int
    test_annotation_count: int
    export_path: str
    metadata_path: str
    status: str
    validation_result_json: Any | None = None
    source_data_fingerprint: str
    created_at: datetime
    completed_at: datetime | None = None


class DatasetVersionSummaryRead(BaseModel):
    dataset_version_id: int
    version_name: str
    status: str
    split_strategy: str
    source_video_ids: list[int]
    exported_image_count: int
    exported_annotation_count: int
    created_at: datetime


class DatasetVersionManifestResponse(BaseModel):
    dataset_version_id: int
    version_name: str
    manifest: Any


class DatasetVersionSplitReportResponse(BaseModel):
    dataset_version_id: int
    version_name: str
    split_report: Any
