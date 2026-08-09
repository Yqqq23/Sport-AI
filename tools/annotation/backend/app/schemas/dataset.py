from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, model_validator

from app.services.basketball_classes import (
    BASKETBALL_CLASS_NAMES,
    DEFAULT_CLASS_ID,
    NUMBER_BEARING_CLASS_ID,
    class_name,
    normalize_jersey_number,
    validate_class_id,
)


LABEL_STATUS_VALUES = {"correct", "wrong_bbox", "wrong_number", "unclear"}


def _apply_class_and_number_rules(model):
    """Shared validation for the class id and the jersey-number side channel.

    Two rules, both enforced here so create and update cannot drift apart:

    1. class_id must be one of the basketball-v2 classes.
    2. Only the jersey class carries a number. Every other class stores NULL.
       A number typed onto a rim or ball box is rejected rather than quietly
       dropped, because silently discarding it would make the operator believe
       ground truth was recorded when none was.

    Jersey numbers are normalized through src/jersey.py's clean_jersey_number,
    the same function the inference pipeline votes with, so "01" and "1" become
    one value on both sides of an OCR evaluation.
    """
    try:
        model.class_id = validate_class_id(model.class_id)
    except ValueError as exc:
        raise ValueError(str(exc)) from exc

    raw_number = (model.corrected_number or "").strip()

    if model.class_id != NUMBER_BEARING_CLASS_ID:
        if raw_number:
            raise ValueError(
                f"corrected_number is only meaningful on the "
                f"'{class_name(NUMBER_BEARING_CLASS_ID)}' class "
                f"(class_id {NUMBER_BEARING_CLASS_ID}); this box is "
                f"'{class_name(model.class_id)}' (class_id {model.class_id}). "
                "Leave the number empty."
            )
        model.corrected_number = None
        return model

    if model.label_status == "unclear":
        # An unreadable number is a legitimate outcome and must stay recordable:
        # the box may still be a useful jersey detection even when the digits
        # cannot be read. It is excluded from export by EXPORTABLE_LABEL_STATUSES.
        model.corrected_number = normalize_jersey_number(raw_number) if raw_number else None
        return model

    if not raw_number:
        raise ValueError(
            "corrected_number cannot be empty for a jersey box unless "
            "label_status is 'unclear'"
        )
    model.corrected_number = normalize_jersey_number(raw_number)
    return model


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


class SavedBoxRead(BaseModel):
    """One already-saved box on a review frame.

    A review item used to expose at most one annotation, because the tool could
    only hold one jersey box per frame. A 5-class dataset needs many boxes of
    different classes on the same image, so the frame now carries a list.
    """

    annotation_id: int
    prediction_id: int
    class_id: int
    class_name: str
    corrected_bbox: list[int]
    corrected_number: str | None = None
    label_status: str
    track_id: int


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
    #: True when player_bbox/jersey_bbox/predicted_number came from
    #: mock_recognition rather than from a real model. Surfaced so a mock box is
    #: never mistaken for a detection.
    is_mock_prediction: bool = False
    prediction_source: str = "manual"
    annotation_id: int | None = None
    corrected_bbox: list[int] | None = None
    corrected_number: str | None = None
    label_status: str | None = None
    #: EVERY saved box on this frame, whatever its class. This is what the
    #: annotation UI renders and edits.
    saved_boxes: list[SavedBoxRead] = Field(default_factory=list)


class ReviewItemsResponse(BaseModel):
    total_count: int
    items: list[ReviewItemRead]
    message: str | None = None
    #: Set when any item on this video carries mock predictions.
    mock_data_warning: str | None = None


class BasketballClassRead(BaseModel):
    id: int
    name: str


class BasketballClassesResponse(BaseModel):
    """The class scheme the UI must use, served from train/basketball.yaml.

    The frontend deliberately does not hardcode the class list: if it did, a
    class appended upstream would leave the UI offering a stale set while the
    exporter used the new one.
    """

    class_count: int
    classes: list[BasketballClassRead]
    number_bearing_class_id: int
    number_bearing_class_name: str
    jersey_number_rule: str
    source_of_truth: str


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
    #: Negative values mean "no YOLO prediction behind this box". The server
    #: reallocates them to a unique id per box so one frame can hold many boxes;
    #: whatever the client sends in the negative range is only a hint.
    prediction_id: int
    video_id: int
    frame_id: int
    track_id: int
    #: Index into the basketball-v2 class scheme. Defaults to jersey so existing
    #: clients that predate the 5-class change keep working unchanged.
    class_id: int = DEFAULT_CLASS_ID
    image_path: str
    image_url: str
    original_bbox: list[int] = Field(min_length=4, max_length=4)
    corrected_bbox: list[int] = Field(min_length=4, max_length=4)
    original_number: str = ""
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

        self.original_number = (self.original_number or "").strip()

        return _apply_class_and_number_rules(self)


class ManualAnnotationRead(BaseModel):
    annotation_id: int
    prediction_id: int
    video_id: int
    frame_id: int
    track_id: int
    class_id: int
    #: Resolved server-side so the UI never has to keep its own copy of the
    #: index-to-name mapping.
    class_name: str
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
    #: Reclassifying an existing box is allowed - mislabelling a referee as a
    #: player is a common correction and forcing a delete-and-redraw would lose
    #: the box geometry that was already right.
    class_id: int = DEFAULT_CLASS_ID
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

        return _apply_class_and_number_rules(self)


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
    class_count: int
    #: Per-class annotation counts, keyed by class name. Exposed so an empty
    #: class is visible immediately rather than after a training run.
    annotation_counts_by_class: dict[str, int] = Field(default_factory=dict)
    #: Rows written to jersey_numbers.json - the OCR ground-truth side channel.
    jersey_number_count: int = 0
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
    class_count: int = 0
    annotation_counts_by_class: dict[str, int] = Field(default_factory=dict)
    jersey_number_count: int = 0
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
