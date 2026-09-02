import logging
import time
import json
import csv
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import numpy as np
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import SessionLocal, get_db
from app.models.dataset import DatasetVersion, JerseyPrediction, ManualAnnotation, ProcessingJob, VideoAsset, VideoFrame
from app.schemas.dataset import AnnotationCandidateRead, AnnotationCandidatesResponse, AnnotationProgressResponse, BasketballClassRead, BasketballClassesResponse, CocoExportRead, CocoExportResponse, DatasetVersionCreate, DatasetVersionManifestResponse, DatasetVersionRead, DatasetVersionSplitReportResponse, DatasetVersionSummaryRead, ExportDownloadResponse, FailureFrameRead, FailureFramesResponse, InconsistentFrameRead, InconsistentFramesResponse, InternalExportResponse, JerseyPredictionRead, JobQueuedResponse, ManualAnnotationCreate, ManualAnnotationRead, ManualAnnotationSaveResponse, ManualAnnotationUpdate, MockRecognitionResponse, ProcessingJobRead, RecognitionResponse, RecognitionSummaryResponse, ReviewItemsResponse, ReviewItemRead, SavedBoxRead, TemporalContext, UploadVideoResponse, VideoAssetRead, VideoExportRead, VideoFrameRead, YoloExportRead, YoloExportResponse
from app.services.basketball_classes import BASKETBALL_CLASS_NAMES, CLASS_COUNT, NUMBER_BEARING_CLASS_ID, assert_matches_basketball_yaml, class_mapping_for_metadata, class_name, classes_txt_content, clean_jersey_number, describe as describe_basketball_classes, render_names_block, validate_class_id
from app.services.coco_export import COCO_CATEGORIES, coco_category_id, validate_coco_dataset
from app.services.dataset_versioning import build_dataset_version, list_dataset_versions, load_dataset_manifest, rebuild_dataset_version, validate_dataset_version
from app.services.fixed7_adapter import Fixed7AdapterError, fixed7_diagnostic_report, run_fixed7_on_video
from app.services.yolo_export import JERSEY_NUMBERS_FILENAME, JERSEY_NUMBERS_SCHEMA, SPORT_CV_DETECTED_FRAMEWORK, SPORT_CV_DETECTED_MODEL_VERSION, SPORT_CV_TARGET_PROJECT, validate_yolo_dataset
from app.services.mock_recognition import MOCK_PREDICTION_WARNING, generate_mock_predictions, is_mock_prediction
from app.services.real_recognition import RealJerseyRecognizer, RealRecognitionConfigurationError
from app.services.storage import (
    FrameCountMismatchError,
    decode_path,
    ensure_data_dirs,
    extract_frames,
    frame_output_dir,
    probe_fps,
    video_output_path,
)
from app.services.yolo_preannotation import YoloPreannotationError, YoloPreannotator


router = APIRouter()
LOGGER = logging.getLogger(__name__)


def _canonical_frame_path(video_id: int, frame_id: int) -> Path:
    return (settings.data_dir / "frames" / str(video_id) / f"frame_{frame_id:06d}.jpg").resolve()


def _resolve_frame_dimensions(frame_row: VideoFrame | None, frame_path: Path | None = None) -> tuple[int | None, int | None]:
    width = frame_row.image_width if frame_row is not None else None
    height = frame_row.image_height if frame_row is not None else None
    if width is not None and height is not None and int(width) > 0 and int(height) > 0:
        return int(width), int(height)

    if frame_path is None and frame_row is not None and frame_row.frame_path:
        frame_path = Path(frame_row.frame_path)
    if frame_path is not None and not frame_path.exists() and frame_row is not None:
        alt = _canonical_frame_path(frame_row.video_id, frame_row.frame_id)
        if alt.exists():
            frame_path = alt
    image = None
    if frame_path is not None and frame_path.exists():
        try:
            image_bytes = np.fromfile(str(frame_path), dtype=np.uint8)
            if image_bytes.size > 0:
                image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        except Exception:
            image = None
    if image is None:
        return None, None
    h, w = image.shape[:2]
    return int(w), int(h)


def _parse_bbox(raw_value: object, *, field_name: str, prediction_id: int) -> list[int]:
    value = raw_value
    if isinstance(raw_value, str):
        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Prediction {prediction_id} has invalid {field_name} JSON") from exc
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"Prediction {prediction_id} {field_name} must be a list")
    try:
        return [int(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Prediction {prediction_id} {field_name} must contain integers") from exc


def _resolve_frame_path(row: JerseyPrediction) -> Path | None:
    if row.frame is not None and row.frame.frame_path:
        frame_path = Path(row.frame.frame_path)
        if frame_path.exists():
            return frame_path
    canonical_path = _canonical_frame_path(row.video_id, row.frame_id)
    if canonical_path.exists():
        return canonical_path
    if row.frame is not None and row.frame.frame_path:
        return Path(row.frame.frame_path)
    return None


def _prediction_to_schema(row: JerseyPrediction, frame_path: Path | None, width: int | None, height: int | None) -> JerseyPredictionRead:
    return JerseyPredictionRead(
        prediction_id=row.id,
        video_id=row.video_id,
        frame_id=row.frame_id,
        track_id=row.track_id,
        player_bbox=_parse_bbox(row.player_bbox, field_name="player_bbox", prediction_id=row.id),
        jersey_bbox=_parse_bbox(row.jersey_bbox, field_name="jersey_bbox", prediction_id=row.id),
        predicted_number=row.predicted_number,
        confidence=row.confidence,
        frame_path=str(frame_path) if frame_path is not None else None,
        image_width=width,
        image_height=height,
        created_at=row.created_at,
    )


def _frame_image_url(video_id: int, frame_id: int, frame_path: Path | None = None) -> str | None:
    canonical_path = _canonical_frame_path(video_id, frame_id)
    if canonical_path.exists():
        return f"/static/frames/{video_id}/frame_{frame_id:06d}.jpg"
    if frame_path is None:
        return None
    frames_root = (settings.data_dir / "frames").resolve()
    try:
        relative_path = frame_path.resolve().relative_to(frames_root)
    except Exception:
        return None
    return "/static/frames/" + "/".join(relative_path.parts)


def _ensure_frame_metadata(row: JerseyPrediction) -> tuple[Path | None, int | None, int | None, bool]:
    frame_path = _resolve_frame_path(row)
    width, height = _resolve_frame_dimensions(row.frame, frame_path)
    needs_update = False
    if row.frame is not None and width is not None and height is not None and (
        row.frame.image_width is None
        or row.frame.image_height is None
        or int(row.frame.image_width or 0) <= 0
        or int(row.frame.image_height or 0) <= 0
    ):
        row.frame.image_width = width
        row.frame.image_height = height
        needs_update = True
    return frame_path, width, height, needs_update


def _normalize_reasons(*values: object) -> list[str]:
    normalized: list[str] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            if value.strip():
                normalized.append(value.strip())
            continue
        if isinstance(value, (list, tuple, set)):
            for item in value:
                if isinstance(item, str) and item.strip():
                    normalized.append(item.strip())
    return sorted(set(normalized))


def _build_annotation_candidate(
    row: JerseyPrediction,
    *,
    reasons: object,
    context: TemporalContext | dict[str, object] | None = None,
) -> tuple[AnnotationCandidateRead | None, bool]:
    frame_path, width, height, needs_update = _ensure_frame_metadata(row)
    if row.frame is None and frame_path is None:
        LOGGER.warning(
            "Skipping annotation candidate prediction_id=%s video_id=%s frame_id=%s because frame relationship and frame file are missing",
            row.id,
            row.video_id,
            row.frame_id,
        )
        return None, needs_update

    try:
        normalized_context = TemporalContext.model_validate(context) if context is not None else None
        base = _prediction_to_schema(row, frame_path, width, height)
        candidate = AnnotationCandidateRead(
            **base.model_dump(),
            reasons=_normalize_reasons(reasons),
            context=normalized_context,
            image_url=_frame_image_url(row.video_id, row.frame_id, frame_path),
        )
        return candidate, needs_update
    except Exception:
        LOGGER.exception(
            "Failed to build annotation candidate prediction_id=%s video_id=%s frame_id=%s",
            row.id,
            row.video_id,
            row.frame_id,
        )
        return None, needs_update


def _job_to_schema(job: ProcessingJob) -> ProcessingJobRead:
    parsed_result = None
    if job.result_json:
        try:
            parsed_result = json.loads(job.result_json)
        except Exception:
            parsed_result = job.result_json
    return ProcessingJobRead(
        job_id=job.job_id,
        job_type=job.job_type,
        video_id=job.video_id,
        status=job.status,
        progress=job.progress,
        message=job.message,
        result_json=parsed_result,
        error_message=job.error_message,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
    )


def _annotation_to_schema(row: ManualAnnotation) -> ManualAnnotationRead:
    return ManualAnnotationRead(
        annotation_id=row.annotation_id,
        prediction_id=row.prediction_id,
        video_id=row.video_id,
        frame_id=row.frame_id,
        track_id=row.track_id,
        class_id=int(row.class_id),
        class_name=class_name(int(row.class_id)),
        image_path=row.image_path,
        image_url=row.image_url,
        original_bbox=_parse_bbox(row.original_bbox, field_name="original_bbox", prediction_id=row.prediction_id),
        corrected_bbox=_parse_bbox(row.corrected_bbox, field_name="corrected_bbox", prediction_id=row.prediction_id),
        original_number=row.original_number,
        corrected_number=row.corrected_number,
        label_status=row.label_status,
        created_at=row.created_at,
        deleted_at=row.deleted_at,
    )


def _annotation_to_saved_box(row: ManualAnnotation) -> SavedBoxRead:
    return SavedBoxRead(
        annotation_id=row.annotation_id,
        prediction_id=row.prediction_id,
        class_id=int(row.class_id),
        class_name=class_name(int(row.class_id)),
        corrected_bbox=_parse_bbox(row.corrected_bbox, field_name="corrected_bbox", prediction_id=row.prediction_id),
        corrected_number=row.corrected_number,
        label_status=row.label_status,
        track_id=int(row.track_id),
    )


def _active_annotation_query(db: Session):
    return db.query(ManualAnnotation).filter(ManualAnnotation.deleted_at.is_(None))


def _annotation_matches(db: Session, prediction_id: int, video_id: int) -> list[ManualAnnotation]:
    return (
        db.query(ManualAnnotation)
        .filter(ManualAnnotation.prediction_id == prediction_id, ManualAnnotation.video_id == video_id)
        .order_by(ManualAnnotation.annotation_id.asc())
        .all()
    )


#: First id handed out to a hand-drawn box. Manual ids run DOWNWARD from here.
#:
#: Two disjoint negative ranges are in play and they must not collide:
#:   -(frame_id + 1)          synthesised by list_review_items for a frame that
#:                            has no prediction. Reaches -1000000 only past
#:                            frame 999998.
#:   <= MANUAL_BOX_ID_BASE    allocated here, one per hand-drawn box.
#: Everything at or below the base is guaranteed to be a hand-drawn box.
MANUAL_BOX_ID_BASE = -1_000_000


def _allocate_manual_prediction_id(db: Session, video_id: int) -> int:
    """Reserve a unique prediction_id for a newly drawn box.

    A 5-class dataset needs SEVERAL boxes on one frame - players, a ball, a rim,
    referees - but the original tool keyed annotation identity on
    (prediction_id, video_id) and synthesised one id per FRAME, which capped it
    at one hand-drawn box per image. Allocating one id per BOX lifts that cap
    while leaving the upsert-by-prediction behaviour for real YOLO predictions
    (positive ids) exactly as it was.

    Ids also stay unique across the whole video because dataset_versioning treats
    a repeated source_prediction_id as a split-leakage error.
    """
    lowest = (
        db.query(ManualAnnotation.prediction_id)
        .filter(ManualAnnotation.video_id == video_id)
        .order_by(ManualAnnotation.prediction_id.asc())
        .limit(1)
        .scalar()
    )
    if lowest is None or int(lowest) > MANUAL_BOX_ID_BASE:
        return MANUAL_BOX_ID_BASE
    return int(lowest) - 1


def _get_annotation_or_404(db: Session, annotation_id: int) -> ManualAnnotation:
    row = _active_annotation_query(db).filter(ManualAnnotation.annotation_id == annotation_id).first()
    if row is None:
        raise HTTPException(status_code=404, detail="Annotation not found")
    return row


def _exports_root() -> Path:
    root = (settings.data_dir / "exports" / "internal").resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _coco_exports_root() -> Path:
    root = (settings.data_dir / "exports" / "coco").resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _yolo_exports_root() -> Path:
    root = (settings.data_dir / "exports" / "yolo").resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _export_dir(video_id: int, export_id: str) -> Path:
    path = _exports_root() / export_id
    path.mkdir(parents=True, exist_ok=True)
    (path / "images").mkdir(parents=True, exist_ok=True)
    (path / "crops").mkdir(parents=True, exist_ok=True)
    return path


def _coco_export_dir(export_id: str) -> Path:
    path = _coco_exports_root() / export_id
    path.mkdir(parents=True, exist_ok=True)
    (path / "images").mkdir(parents=True, exist_ok=True)
    (path / "annotations").mkdir(parents=True, exist_ok=True)
    return path


def _yolo_export_dir(export_id: str) -> Path:
    path = _yolo_exports_root() / export_id
    path.mkdir(parents=True, exist_ok=True)
    (path / "images").mkdir(parents=True, exist_ok=True)
    (path / "labels").mkdir(parents=True, exist_ok=True)
    return path


def _write_image_file(path: Path, image: np.ndarray) -> None:
    success, encoded = cv2.imencode(path.suffix or ".jpg", image)
    if not success:
        raise RuntimeError(f"Failed to encode image: {path}")
    encoded.tofile(str(path))


def _frame_image_path_from_annotation(row: ManualAnnotation) -> Path:
    if row.image_path:
        candidate = Path(row.image_path)
        if candidate.exists():
            return candidate
    canonical = _canonical_frame_path(row.video_id, row.frame_id)
    if canonical.exists():
        return canonical
    raise FileNotFoundError(f"Frame image not found for annotation_id={row.annotation_id}")


def _clamp_bbox_xyxy(bbox: list[int], image_width: int, image_height: int) -> tuple[list[int], bool]:
    if len(bbox) != 4:
        return bbox, False
    x1, y1, x2, y2 = [int(value) for value in bbox]
    x1 = max(0, min(x1, image_width))
    y1 = max(0, min(y1, image_height))
    x2 = max(0, min(x2, image_width))
    y2 = max(0, min(y2, image_height))
    valid = x1 < x2 and y1 < y2
    return [x1, y1, x2, y2], valid


def _create_export_zip(export_dir: Path) -> Path:
    zip_path = export_dir.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in sorted(export_dir.rglob("*")):
            if file_path.is_file():
                archive.write(file_path, arcname=str(file_path.relative_to(export_dir)))
    return zip_path


def _read_json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _int_metadata_value(payload: dict[str, Any], key: str, default: int = 0) -> int:
    value = payload.get(key, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _update_job(
    db: Session,
    job: ProcessingJob,
    *,
    status: str | None = None,
    progress: int | None = None,
    message: str | None = None,
    result_json: dict[str, object] | None = None,
    error_message: str | None = None,
    started: bool = False,
    completed: bool = False,
) -> None:
    now = datetime.now(timezone.utc)
    if status is not None:
        job.status = status
    if progress is not None:
        job.progress = progress
    if message is not None:
        job.message = message
    if result_json is not None:
        job.result_json = json.dumps(result_json, ensure_ascii=False)
    if error_message is not None:
        job.error_message = error_message
    if started and job.started_at is None:
        job.started_at = now
    if completed:
        job.completed_at = now
    db.add(job)
    db.commit()
    db.refresh(job)


def _run_mock_jersey_recognition_job(job_id: str, video_id: int) -> None:
    db = SessionLocal()
    try:
        job = db.query(ProcessingJob).filter(ProcessingJob.job_id == job_id).first()
        video = db.query(VideoAsset).filter(VideoAsset.id == video_id).first()
        if job is None or video is None:
            return

        _update_job(db, job, status="running", progress=5, message="Loading extracted frames", started=True)
        frames = db.query(VideoFrame).filter(VideoFrame.video_id == video_id).order_by(VideoFrame.frame_id.asc()).all()
        if not frames:
            raise RuntimeError("No extracted frames found for this video")

        _update_job(db, job, progress=25, message="Generating mock jersey predictions")
        db.query(JerseyPrediction).filter(JerseyPrediction.video_id == video_id).delete()
        db.commit()

        predictions = generate_mock_predictions(video, frames)
        _update_job(db, job, progress=70, message="Saving predictions to SQLite")
        db.add_all(predictions)
        video.status = "recognition_ready"
        db.add(video)
        db.commit()

        _update_job(
            db,
            job,
            status="completed",
            progress=100,
            message="Legacy mock recognition completed",
            result_json={"video_id": video.id, "prediction_count": len(predictions)},
            completed=True,
        )
    except Exception as exc:
        db.rollback()
        job = db.query(ProcessingJob).filter(ProcessingJob.job_id == job_id).first()
        if job is not None:
            _update_job(
                db,
                job,
                status="failed",
                progress=min(job.progress or 0, 99),
                message="Legacy mock recognition failed",
                error_message=str(exc),
                completed=True,
            )
    finally:
        db.close()


def _run_jersey_recognition_job(job_id: str, video_id: int) -> None:
    db = SessionLocal()
    try:
        job = db.query(ProcessingJob).filter(ProcessingJob.job_id == job_id).first()
        video = db.query(VideoAsset).filter(VideoAsset.id == video_id).first()
        if job is None or video is None:
            return

        _update_job(db, job, status="running", progress=5, message="Loading extracted frames", started=True)
        video.status = "recognition_running"
        db.add(video)
        db.commit()

        recognizer = RealJerseyRecognizer()
        _update_job(db, job, progress=25, message="Running jersey recognition")
        db.query(JerseyPrediction).filter(JerseyPrediction.video_id == video_id).delete()
        db.commit()

        predictions = recognizer.run(video_id=video_id, db=db)
        _update_job(db, job, progress=80, message="Saving recognition results")
        db.add_all(predictions)
        video.status = "recognition_ready"
        db.add(video)
        db.commit()

        _update_job(
            db,
            job,
            status="completed",
            progress=100,
            message=recognizer.last_run_metadata.message or "Recognition completed",
            result_json={
                "video_id": video.id,
                "prediction_count": len(predictions),
                "ocr_available": recognizer.last_run_metadata.ocr_available,
                "warnings": recognizer.last_run_metadata.warnings,
            },
            completed=True,
        )
    except RealRecognitionConfigurationError as exc:
        db.rollback()
        job = db.query(ProcessingJob).filter(ProcessingJob.job_id == job_id).first()
        video = db.query(VideoAsset).filter(VideoAsset.id == video_id).first()
        if video is not None:
            video.status = "frames_extracted"
            db.add(video)
            db.commit()
        if job is not None:
            _update_job(
                db,
                job,
                status="failed",
                progress=min(job.progress or 0, 99),
                message="Recognition failed",
                error_message=str(exc),
                completed=True,
            )
    except Exception as exc:
        db.rollback()
        job = db.query(ProcessingJob).filter(ProcessingJob.job_id == job_id).first()
        video = db.query(VideoAsset).filter(VideoAsset.id == video_id).first()
        if video is not None:
            video.status = "frames_extracted"
            db.add(video)
            db.commit()
        if job is not None:
            _update_job(
                db,
                job,
                status="failed",
                progress=min(job.progress or 0, 99),
                message="Recognition failed",
                error_message=str(exc),
                completed=True,
            )
    finally:
        db.close()


def _run_yolo_preannotation_job(job_id: str, video_id: int) -> None:
    db = SessionLocal()
    try:
        job = db.query(ProcessingJob).filter(ProcessingJob.job_id == job_id).first()
        video = db.query(VideoAsset).filter(VideoAsset.id == video_id).first()
        if job is None or video is None:
            return

        _update_job(db, job, status="running", progress=10, message="Running YOLO pre-annotation", started=True)
        video.status = "recognition_running"
        db.add(video)
        db.commit()

        preannotator = YoloPreannotator()
        db.query(JerseyPrediction).filter(JerseyPrediction.video_id == video_id).delete()
        db.commit()
        predictions, model_name = preannotator.run(video_id=video_id, db=db)
        db.add_all(predictions)
        video.status = "recognition_ready"
        db.add(video)
        db.commit()
        _update_job(
            db,
            job,
            status="completed",
            progress=100,
            message=f"YOLO pre-annotation completed ({model_name})",
            result_json={"video_id": video_id, "prediction_count": len(predictions), "model": model_name},
            completed=True,
        )
    except YoloPreannotationError as exc:
        db.rollback()
        job = db.query(ProcessingJob).filter(ProcessingJob.job_id == job_id).first()
        video = db.query(VideoAsset).filter(VideoAsset.id == video_id).first()
        if video is not None:
            video.status = "frames_extracted"
            db.add(video)
            db.commit()
        if job is not None:
            _update_job(db, job, status="failed", progress=min(job.progress or 0, 99), message="YOLO pre-annotation failed", error_message=str(exc), completed=True)
    finally:
        db.close()


@router.get(
    "/videos",
    response_model=list[VideoAssetRead],
    summary="查看视频列表",
    description="返回所有已经上传的视频，包括 video_id、文件名、状态和抽帧数量。",
    tags=["视频管理"],
)
def list_videos(db: Session = Depends(get_db)):
    rows = db.query(VideoAsset).order_by(VideoAsset.id.desc()).all()
    return [
        VideoAssetRead(
            id=row.id,
            video_id=row.id,
            filename=row.filename,
            original_filename=row.filename,
            path=row.path,
            status=row.status,
            frame_count=len(row.frames),
            created_at=row.created_at,
        )
        for row in rows
    ]


@router.get(
    "/videos/{video_id}/frames",
    response_model=list[VideoFrameRead],
    summary="查看视频抽帧结果",
    description="查看某个视频已经抽取出来的帧列表，包括 frame_id、frame_path 和尺寸信息。",
    tags=["视频管理"],
)
def list_video_frames(video_id: int, db: Session = Depends(get_db)):
    video = db.query(VideoAsset).filter(VideoAsset.id == video_id).first()
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found")
    return db.query(VideoFrame).filter(VideoFrame.video_id == video_id).order_by(VideoFrame.frame_id.asc()).all()


@router.get(
    "/jobs/{job_id}",
    response_model=ProcessingJobRead,
    summary="查看识别任务进度",
    description="查询异步球衣识别任务的状态、进度、错误信息和完成结果。",
    tags=["球衣识别"],
)
def get_job(job_id: str, db: Session = Depends(get_db)):
    job = db.query(ProcessingJob).filter(ProcessingJob.job_id == job_id).first()
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return _job_to_schema(job)


# ==========================================================================
# 已停用的出框路径（2026-09）
# ==========================================================================
# 下面这些函数不再挂 @router，因此**没有 HTTP 入口**。函数体保留是为了
# 历史可读性和 import 兼容，不是为了随时能开回来。
#
# 为什么全部停用：仓库里曾经有四套互不相同的球衣框几何，谁在跑取决于点了
# 哪个按钮，而它们和推理时真正用的那套（basketball-v2 的
# src/jersey.py::torso_region_from_player_box，以及优先于它的
# detect.CLASS_JERSEY 检测框）IoU 只有 0.625–0.87。用其中任何一套标出来的
# 数据去微调，训练分布和推理分布就是错开的，而这种错开在验证集上完全看不出来。
#
# 出框现在统一由 basketball-v2 的 jersey_train mine 提供，经
# scripts/import_crops.py 导入，source 列记录框来自 detector 还是 torso_fallback。
#
#   run_mock_jersey_recognition / _async      mock_recognition.py     纯造假
#   run_jersey_recognition / _async           real_recognition.py     yolov8n + 硬编码几何
#   run_yolo_preannotation_async              yolo_preannotation.py   yolov8n + 另一套硬编码几何
#   recognition_fixed7_check / fixed7_dry_run fixed7_adapter.py       第三个仓库，又一套几何
#
# 四个 service 文件都留在磁盘上，顶部有同样说明。
# ==========================================================================


def recognition_fixed7_check() -> dict[str, Any]:
    """[已停用] 见上方说明。原路由 GET /recognition/fixed7-check"""
    return fixed7_diagnostic_report()


def fixed7_dry_run(video_id: int, db: Session = Depends(get_db)) -> dict[str, Any]:
    """[已停用] 见上方说明。原路由 POST /videos/{video_id}/fixed7-dry-run"""
    video = db.query(VideoAsset).filter(VideoAsset.id == video_id).first()
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found")
    video_path = Path(video.path)
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Saved video file not found")

    output_dir = (settings.data_dir / "fixed7_dry_run" / str(video_id)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        records = run_fixed7_on_video(str(video_path.resolve()), str(output_dir))
        return {
            "ok": True,
            "record_count": len(records),
            "output_dir": str(output_dir),
            "sample": records[:3],
        }
    except Fixed7AdapterError as exc:
        return exc.to_dict()


@router.get(
    "/videos/{video_id}/predictions",
    response_model=list[JerseyPredictionRead],
    summary="查看原始识别结果",
    description="返回某个视频的原始 JerseyPrediction 结果，包括 player_bbox、jersey_bbox、predicted_number、confidence 和 track_id。",
    tags=["球衣识别"],
)
def list_video_predictions(video_id: int, db: Session = Depends(get_db)):
    video = db.query(VideoAsset).filter(VideoAsset.id == video_id).first()
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found")
    rows = db.query(JerseyPrediction).filter(JerseyPrediction.video_id == video_id).order_by(JerseyPrediction.frame_id.asc(), JerseyPrediction.track_id.asc()).all()
    LOGGER.info(
        "Predictions endpoint video_id=%s prediction_count=%s first_confidences=%s first_numbers=%s",
        video_id,
        len(rows),
        [round(float(row.confidence or 0.0), 3) for row in rows[:3]],
        [row.predicted_number for row in rows[:3]],
    )
    response: list[JerseyPredictionRead] = []
    updated_any = False
    for row in rows:
        try:
            frame_path, width, height, needs_update = _ensure_frame_metadata(row)
            updated_any = updated_any or needs_update
            if row.frame is None and frame_path is None:
                LOGGER.warning(
                    "Skipping prediction_id=%s video_id=%s frame_id=%s because frame relationship and frame file are missing",
                    row.id,
                    row.video_id,
                    row.frame_id,
                )
                continue
            response.append(_prediction_to_schema(row, frame_path, width, height))
        except Exception:
            LOGGER.exception(
                "Failed to serialize prediction prediction_id=%s video_id=%s frame_id=%s",
                row.id,
                row.video_id,
                row.frame_id,
            )
    if updated_any:
        db.commit()
    return response


@router.get("/videos/{video_id}/failure-frames", response_model=FailureFramesResponse, include_in_schema=False)
def list_failure_frames(video_id: int, threshold: float = 0.7, db: Session = Depends(get_db)):
    video = db.query(VideoAsset).filter(VideoAsset.id == video_id).first()
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found")

    rows = (
        db.query(JerseyPrediction)
        .filter(JerseyPrediction.video_id == video_id, JerseyPrediction.confidence < threshold)
        .order_by(JerseyPrediction.confidence.asc(), JerseyPrediction.frame_id.asc())
        .all()
    )

    items: list[FailureFrameRead] = []
    updated_any = False
    for row in rows:
        try:
            frame_path, width, height, needs_update = _ensure_frame_metadata(row)
            updated_any = updated_any or needs_update
            if row.frame is None and frame_path is None:
                LOGGER.warning(
                    "Skipping failure-frame prediction_id=%s video_id=%s frame_id=%s because frame relationship and frame file are missing",
                    row.id,
                    row.video_id,
                    row.frame_id,
                )
                continue
            base = _prediction_to_schema(row, frame_path, width, height)
            items.append(FailureFrameRead(**base.model_dump(), reason="low_confidence"))
        except Exception:
            LOGGER.exception(
                "Failed to serialize failure-frame prediction_id=%s video_id=%s frame_id=%s",
                row.id,
                row.video_id,
                row.frame_id,
            )
    if updated_any:
        db.commit()

    return FailureFramesResponse(total_count=len(items), items=items)


@router.get("/videos/{video_id}/inconsistent-frames", response_model=InconsistentFramesResponse, include_in_schema=False)
def list_inconsistent_frames(video_id: int, db: Session = Depends(get_db)):
    video = db.query(VideoAsset).filter(VideoAsset.id == video_id).first()
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found")

    rows = (
        db.query(JerseyPrediction)
        .filter(JerseyPrediction.video_id == video_id)
        .order_by(JerseyPrediction.track_id.asc(), JerseyPrediction.frame_id.asc())
        .all()
    )

    grouped: dict[int, list[JerseyPrediction]] = {}
    for row in rows:
        grouped.setdefault(row.track_id, []).append(row)

    items: list[InconsistentFrameRead] = []
    updated_any = False
    for _, track_rows in grouped.items():
        if len(track_rows) < 3:
            continue
        for idx in range(1, len(track_rows) - 1):
            prev_row = track_rows[idx - 1]
            curr_row = track_rows[idx]
            next_row = track_rows[idx + 1]
            prev_num = prev_row.predicted_number
            curr_num = curr_row.predicted_number
            next_num = next_row.predicted_number
            if prev_num == next_num and curr_num != prev_num:
                try:
                    frame_path, width, height, needs_update = _ensure_frame_metadata(curr_row)
                    updated_any = updated_any or needs_update
                    if curr_row.frame is None and frame_path is None:
                        LOGGER.warning(
                            "Skipping inconsistent-frame prediction_id=%s video_id=%s frame_id=%s because frame relationship and frame file are missing",
                            curr_row.id,
                            curr_row.video_id,
                            curr_row.frame_id,
                        )
                        continue
                    base = _prediction_to_schema(curr_row, frame_path, width, height)
                    items.append(
                        InconsistentFrameRead(
                            **base.model_dump(),
                            reason="temporal_inconsistency",
                            context=TemporalContext(
                                previous_number=prev_num,
                                current_number=curr_num,
                                next_number=next_num,
                            ),
                        )
                    )
                except Exception:
                    LOGGER.exception(
                        "Failed to serialize inconsistent-frame prediction_id=%s video_id=%s frame_id=%s",
                        curr_row.id,
                        curr_row.video_id,
                        curr_row.frame_id,
                    )
    if updated_any:
        db.commit()

    items.sort(key=lambda item: item.frame_id)
    return InconsistentFramesResponse(total_count=len(items), items=items)


@router.get(
    "/videos/{video_id}/annotation-candidates",
    response_model=AnnotationCandidatesResponse,
    summary="查看需要人工复查的候选帧",
    description="返回模型可能识别错误的球衣号码样本。候选帧来自两种规则：1）模型置信度低于阈值；2）同一个 track_id 的号码在时间上不一致，例如 23 → 28 → 23。",
    tags=["球衣识别"],
)
def list_annotation_candidates(video_id: int, threshold: float = 0.7, db: Session = Depends(get_db)):
    try:
        video = db.query(VideoAsset).filter(VideoAsset.id == video_id).first()
        if video is None:
            raise HTTPException(status_code=404, detail="Video not found")

        rows = (
            db.query(JerseyPrediction)
            .filter(JerseyPrediction.video_id == video_id)
            .order_by(JerseyPrediction.track_id.asc(), JerseyPrediction.frame_id.asc(), JerseyPrediction.id.asc())
            .all()
        )

        if not rows:
            LOGGER.info(
                "Annotation candidates video_id=%s threshold=%s total_predictions=0 low_confidence_count=0 inconsistent_count=0 combined_candidate_count=0 first_confidences=[] first_numbers=[]",
                video_id,
                threshold,
            )
            return AnnotationCandidatesResponse(
                total_count=0,
                items=[],
                message="No predictions found. Please run Jersey Recognition first.",
            )

        low_confidence_ids = {
            row.id
            for row in rows
            if row.confidence is not None and float(row.confidence) < threshold
        }

        inconsistent_contexts: dict[int, TemporalContext] = {}
        grouped: dict[int, list[JerseyPrediction]] = {}
        for row in rows:
            grouped.setdefault(row.track_id, []).append(row)

        for track_rows in grouped.values():
            if len(track_rows) < 3:
                continue
            for idx in range(1, len(track_rows) - 1):
                prev_row = track_rows[idx - 1]
                curr_row = track_rows[idx]
                next_row = track_rows[idx + 1]
                if prev_row.predicted_number == next_row.predicted_number and curr_row.predicted_number != prev_row.predicted_number:
                    inconsistent_contexts[curr_row.id] = TemporalContext(
                        previous_number=prev_row.predicted_number,
                        current_number=curr_row.predicted_number,
                        next_number=next_row.predicted_number,
                    )

        merged: dict[int, AnnotationCandidateRead] = {}
        updated_any = False
        for row in rows:
            reasons: list[str] = []
            if row.id in low_confidence_ids:
                reasons.append("low_confidence")
            context = inconsistent_contexts.get(row.id)
            if context is not None:
                reasons.append("temporal_inconsistency")
            if not reasons:
                continue

            candidate, needs_update = _build_annotation_candidate(row, reasons=reasons, context=context)
            updated_any = updated_any or needs_update
            if candidate is None:
                continue

            existing = merged.get(candidate.prediction_id)
            if existing is None:
                merged[candidate.prediction_id] = candidate
                continue

            existing.reasons = _normalize_reasons(existing.reasons, candidate.reasons)
            if existing.context is None:
                existing.context = candidate.context
            if existing.image_url is None:
                existing.image_url = candidate.image_url
            if existing.frame_path is None:
                existing.frame_path = candidate.frame_path
            if existing.image_width is None:
                existing.image_width = candidate.image_width
            if existing.image_height is None:
                existing.image_height = candidate.image_height

        if updated_any:
            db.commit()

        items = sorted(merged.values(), key=lambda item: (item.frame_id, item.track_id, item.prediction_id))
        LOGGER.info(
            "Annotation candidates video_id=%s threshold=%s total_predictions=%s low_confidence_count=%s inconsistent_count=%s combined_candidate_count=%s first_confidences=%s first_numbers=%s",
            video_id,
            threshold,
            len(rows),
            len(low_confidence_ids),
            len(inconsistent_contexts),
            len(items),
            [round(float(row.confidence or 0.0), 3) for row in rows[:3]],
            [row.predicted_number for row in rows[:3]],
        )
        if not items:
            return AnnotationCandidatesResponse(
                total_count=0,
                items=[],
                message="No candidates found. Try increasing threshold or check recognition output.",
            )
        return AnnotationCandidatesResponse(total_count=len(items), items=items)
    except HTTPException:
        raise
    except Exception:
        LOGGER.exception(
            "Annotation candidates endpoint failed video_id=%s threshold=%s",
            video_id,
            threshold,
        )
        raise


def run_mock_jersey_recognition(video_id: int, db: Session = Depends(get_db)):
    """[已停用] 原路由 POST /videos/{video_id}/run-mock-jersey-recognition"""
    video = db.query(VideoAsset).filter(VideoAsset.id == video_id).first()
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found")
    frames = db.query(VideoFrame).filter(VideoFrame.video_id == video_id).order_by(VideoFrame.frame_id.asc()).all()
    if not frames:
        raise HTTPException(status_code=400, detail="No extracted frames found for this video")

    db.query(JerseyPrediction).filter(JerseyPrediction.video_id == video_id).delete()
    predictions = generate_mock_predictions(video, frames)
    db.add_all(predictions)
    video.status = "recognition_ready"
    db.commit()

    return MockRecognitionResponse(video_id=video.id, prediction_count=len(predictions), status=video.status)


def run_jersey_recognition(video_id: int, db: Session = Depends(get_db)):
    """[已停用] 原路由 POST /videos/{video_id}/run-jersey-recognition"""
    video = db.query(VideoAsset).filter(VideoAsset.id == video_id).first()
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found")
    frames = db.query(VideoFrame).filter(VideoFrame.video_id == video_id).order_by(VideoFrame.frame_id.asc()).all()
    if not frames:
        raise HTTPException(status_code=400, detail="No extracted frames found for this video")

    recognizer = RealJerseyRecognizer()
    try:
        db.query(JerseyPrediction).filter(JerseyPrediction.video_id == video_id).delete()
        db.commit()
        video.status = "recognition_running"
        db.add(video)
        db.commit()
        predictions = recognizer.run(video_id=video_id, db=db)
        db.add_all(predictions)
        video.status = "recognition_ready"
        db.add(video)
        db.commit()
        return RecognitionResponse(
            video_id=video.id,
            prediction_count=len(predictions),
            status=video.status,
            message=recognizer.last_run_metadata.message or "Recognition completed",
        )
    except RealRecognitionConfigurationError as exc:
        db.rollback()
        video.status = "frames_extracted"
        db.add(video)
        db.commit()
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def run_mock_jersey_recognition_async(video_id: int, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """[已停用] 原路由 POST /videos/{video_id}/run-mock-jersey-recognition-async"""
    video = db.query(VideoAsset).filter(VideoAsset.id == video_id).first()
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found")

    job = ProcessingJob(
        job_id=uuid4().hex,
        job_type="mock_jersey_recognition",
        video_id=video_id,
        status="queued",
        progress=0,
        message="Job queued",
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    background_tasks.add_task(_run_mock_jersey_recognition_job, job.job_id, video_id)
    return JobQueuedResponse(job_id=job.job_id, status=job.status, poll_url=f"/jobs/{job.job_id}")


def run_jersey_recognition_async(video_id: int, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """[已停用] 原路由 POST /videos/{video_id}/run-jersey-recognition-async"""
    video = db.query(VideoAsset).filter(VideoAsset.id == video_id).first()
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found")

    job = ProcessingJob(
        job_id=uuid4().hex,
        job_type="jersey_recognition",
        video_id=video_id,
        status="queued",
        progress=0,
        message="Recognition queued",
    )
    db.add(job)
    video.status = "recognition_running"
    db.add(video)
    db.commit()
    db.refresh(job)

    background_tasks.add_task(_run_jersey_recognition_job, job.job_id, video_id)
    return JobQueuedResponse(job_id=job.job_id, status=job.status, poll_url=f"/jobs/{job.job_id}")


def run_yolo_preannotation_async(video_id: int, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """[已停用] 原路由 POST /videos/{video_id}/run-yolo-preannotation-async

    这条是前端**唯一**在用的出框链路，所以它的停用是整件事的重点：
    yolov8n 的 COCO person 类 + yolo_preannotation._estimate_jersey_bbox()
    的 25%/15%/25%/45% 几何，和推理时的 20%/15%/20%/55% IoU 只有 0.625。
    """
    video = db.query(VideoAsset).filter(VideoAsset.id == video_id).first()
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found")
    job = ProcessingJob(
        job_id=uuid4().hex,
        job_type="yolo_preannotation",
        video_id=video_id,
        status="queued",
        progress=0,
        message="YOLO pre-annotation queued",
    )
    db.add(job)
    video.status = "recognition_running"
    db.add(video)
    db.commit()
    db.refresh(job)
    background_tasks.add_task(_run_yolo_preannotation_job, job.job_id, video_id)
    return JobQueuedResponse(job_id=job.job_id, status=job.status, poll_url=f"/jobs/{job.job_id}")


@router.get(
    "/videos/{video_id}/recognition-summary",
    response_model=RecognitionSummaryResponse,
    summary="查看识别结果概览",
    description="返回该视频的识别统计信息，例如预测数量、候选帧数量、低置信度数量、时间不一致数量、缺失号码数量和平均置信度。",
    tags=["球衣识别"],
)
def get_recognition_summary(video_id: int, threshold: float = 0.7, db: Session = Depends(get_db)):
    video = db.query(VideoAsset).filter(VideoAsset.id == video_id).first()
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found")

    rows = (
        db.query(JerseyPrediction)
        .filter(JerseyPrediction.video_id == video_id)
        .order_by(JerseyPrediction.track_id.asc(), JerseyPrediction.frame_id.asc(), JerseyPrediction.id.asc())
        .all()
    )
    low_confidence_count = sum(1 for row in rows if row.confidence is not None and float(row.confidence) < threshold)
    missing_number_count = sum(1 for row in rows if not (row.predicted_number or "").strip())
    inconsistent_count = 0
    grouped: dict[int, list[JerseyPrediction]] = {}
    for row in rows:
        grouped.setdefault(row.track_id, []).append(row)
    for track_rows in grouped.values():
        if len(track_rows) < 3:
            continue
        for idx in range(1, len(track_rows) - 1):
            prev_row = track_rows[idx - 1]
            curr_row = track_rows[idx]
            next_row = track_rows[idx + 1]
            if prev_row.predicted_number == next_row.predicted_number and curr_row.predicted_number != prev_row.predicted_number:
                inconsistent_count += 1
    candidate_ids = set()
    for row in rows:
        if row.confidence is not None and float(row.confidence) < threshold:
            candidate_ids.add(row.id)
    for track_rows in grouped.values():
        if len(track_rows) < 3:
            continue
        for idx in range(1, len(track_rows) - 1):
            prev_row = track_rows[idx - 1]
            curr_row = track_rows[idx]
            next_row = track_rows[idx + 1]
            if prev_row.predicted_number == next_row.predicted_number and curr_row.predicted_number != prev_row.predicted_number:
                candidate_ids.add(curr_row.id)
    average_confidence = round(sum(float(row.confidence or 0.0) for row in rows) / len(rows), 3) if rows else 0.0
    return RecognitionSummaryResponse(
        prediction_count=len(rows),
        candidate_count=len(candidate_ids),
        low_confidence_count=low_confidence_count,
        inconsistency_count=inconsistent_count,
        missing_number_count=missing_number_count,
        average_confidence=average_confidence,
        status=video.status,
    )


@router.get(
    "/videos/{video_id}/review-items",
    response_model=ReviewItemsResponse,
    summary="加载标注复查帧",
    description="返回用于人工标注的复查条目。若存在 YOLO 预标注，则返回预测框；若没有预测，也会返回可手工绘制 bbox 的视频帧。",
    tags=["人工标注"],
)
def list_review_items(video_id: int, db: Session = Depends(get_db)):
    video = db.query(VideoAsset).filter(VideoAsset.id == video_id).first()
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found")
    frames = db.query(VideoFrame).filter(VideoFrame.video_id == video_id).order_by(VideoFrame.frame_id.asc()).all()
    predictions = db.query(JerseyPrediction).filter(JerseyPrediction.video_id == video_id).order_by(JerseyPrediction.frame_id.asc(), JerseyPrediction.track_id.asc(), JerseyPrediction.id.asc()).all()
    annotations = db.query(ManualAnnotation).filter(ManualAnnotation.video_id == video_id, ManualAnnotation.deleted_at.is_(None)).all()
    annotations_by_key = {(int(row.video_id), int(row.prediction_id)): row for row in annotations}
    # Every saved box on a frame, whatever its class and however it was created.
    # A 5-class frame holds many boxes, so a per-prediction lookup alone can no
    # longer tell the UI what is already drawn on the image.
    annotations_by_frame: dict[int, list[ManualAnnotation]] = {}
    for row in annotations:
        annotations_by_frame.setdefault(int(row.frame_id), []).append(row)
    for rows in annotations_by_frame.values():
        rows.sort(key=lambda item: (int(item.class_id), int(item.annotation_id)))

    predictions_by_frame: dict[int, list[JerseyPrediction]] = {}
    for prediction in predictions:
        predictions_by_frame.setdefault(int(prediction.frame_id), []).append(prediction)

    mock_prediction_count = sum(1 for prediction in predictions if is_mock_prediction(prediction))

    # 进不了队列的两类条目，分开计数因为原因完全不同。
    skipped_too_small = 0     # 框太小，推理时也会被 gate 挡掉，标了没用
    skipped_no_box = 0        # 这一帧没有任何框

    items: list[ReviewItemRead] = []
    for frame in frames:
        frame_predictions = predictions_by_frame.get(int(frame.frame_id), [])
        saved_boxes = [
            _annotation_to_saved_box(row) for row in annotations_by_frame.get(int(frame.frame_id), [])
        ]
        if frame_predictions:
            for prediction in frame_predictions:
                jersey_bbox = _parse_bbox(
                    prediction.jersey_bbox, field_name="jersey_bbox", prediction_id=prediction.id
                )
                # 尺寸门限。和导入侧（scripts/import_crops.py）读的是同一份配置，
                # 数值本身来自 basketball-v2 的 OCRGateConfig。推理时会被挡掉的
                # 图，摆到人面前只是浪费标注时间。
                if (
                    (jersey_bbox[2] - jersey_bbox[0]) < settings.min_crop_width
                    or (jersey_bbox[3] - jersey_bbox[1]) < settings.min_crop_height
                ):
                    skipped_too_small += 1
                    continue
                frame_path, width, height, _ = _ensure_frame_metadata(prediction)
                annotation = annotations_by_key.get((video_id, int(prediction.id)))
                items.append(
                    ReviewItemRead(
                        review_id=f"prediction:{prediction.id}",
                        prediction_id=prediction.id,
                        video_id=video_id,
                        frame_id=prediction.frame_id,
                        track_id=prediction.track_id,
                        image_url=_frame_image_url(video_id, prediction.frame_id, frame_path),
                        frame_path=str(frame_path) if frame_path is not None else None,
                        image_width=width,
                        image_height=height,
                        player_bbox=_parse_bbox(prediction.player_bbox, field_name="player_bbox", prediction_id=prediction.id),
                        jersey_bbox=jersey_bbox,
                        predicted_number=prediction.predicted_number,
                        confidence=float(prediction.confidence or 0.0),
                        is_mock_prediction=is_mock_prediction(prediction),
                        prediction_source=str(getattr(prediction, "source", "unknown")),
                        annotation_id=annotation.annotation_id if annotation is not None else None,
                        corrected_bbox=_parse_bbox(annotation.corrected_bbox, field_name="corrected_bbox", prediction_id=prediction.id) if annotation is not None else None,
                        corrected_number=annotation.corrected_number if annotation is not None else None,
                        label_status=annotation.label_status if annotation is not None else None,
                        saved_boxes=saved_boxes,
                    )
                )
        elif not saved_boxes:
            # 既没有导入的框、也没有已保存的标注 —— 跳过。
            #
            # 这里原来会合成一个「整帧 35%–65% / 20%–50%」的兜底框塞进队列。
            # 那个框和球员位置毫无关系（1920×1080 上是固定的 576×324 大块），
            # 人对着它调出来的东西既不是检测结果、也不是干净的手工标注，
            # 只是给数据集掺沙子。
            skipped_no_box += 1
        else:
            # 没有导入的框，但这一帧上已经有人工画过的框。必须留下 —— 不然
            # 已经标好的东西在界面上就消失了，看起来像标注丢了。
            # jersey_bbox 给 None：没有预测就是没有预测，不编一个出来。
            # 前端的取值链是 corrected_bbox ?? jersey_bbox，走前一半。
            frame_path = Path(frame.frame_path) if frame.frame_path else _canonical_frame_path(video_id, frame.frame_id)
            manual_prediction_id = -(int(frame.frame_id) + 1)
            annotation = annotations_by_key.get((video_id, manual_prediction_id))
            items.append(
                ReviewItemRead(
                    review_id=f"frame:{frame.frame_id}",
                    prediction_id=manual_prediction_id,
                    video_id=video_id,
                    frame_id=frame.frame_id,
                    track_id=0,
                    image_url=_frame_image_url(video_id, frame.frame_id, frame_path),
                    frame_path=str(frame_path) if frame_path is not None else None,
                    image_width=frame.image_width,
                    image_height=frame.image_height,
                    player_bbox=None,
                    jersey_bbox=None,
                    predicted_number="",
                    confidence=0.0,
                    is_mock_prediction=False,
                    prediction_source="manual",
                    annotation_id=annotation.annotation_id if annotation is not None else None,
                    corrected_bbox=_parse_bbox(annotation.corrected_bbox, field_name="corrected_bbox", prediction_id=manual_prediction_id) if annotation is not None else None,
                    corrected_number=annotation.corrected_number if annotation is not None else None,
                    label_status=annotation.label_status if annotation is not None else None,
                    saved_boxes=saved_boxes,
                )
            )

    message = None
    if not items:
        if not frames:
            message = "No frames found. Please upload a video first."
        else:
            message = (
                "No annotatable boxes for this video. "
                "Run jersey_train mine, then scripts/import_crops.py to import crops."
            )
    elif skipped_too_small or skipped_no_box:
        message = (
            f"{len(items)} boxes queued. "
            f"Skipped {skipped_too_small} below the "
            f"{settings.min_crop_width}x{settings.min_crop_height} crop gate, "
            f"{skipped_no_box} frames with no imported box."
        )

    LOGGER.info(
        "review items video_id=%s frames=%s predictions=%s queued=%s "
        "skipped_too_small=%s skipped_no_box=%s gate=%sx%s",
        video_id, len(frames), len(predictions), len(items),
        skipped_too_small, skipped_no_box,
        settings.min_crop_width, settings.min_crop_height,
    )

    mock_warning = None
    if mock_prediction_count:
        mock_warning = f"{mock_prediction_count} of {len(predictions)} predictions are MOCK. {MOCK_PREDICTION_WARNING}"
        LOGGER.warning("video_id=%s review items include mock predictions: %s", video_id, mock_warning)

    return ReviewItemsResponse(
        total_count=len(items), items=items, message=message, mock_data_warning=mock_warning
    )


@router.get(
    "/basketball-classes",
    response_model=BasketballClassesResponse,
    summary="Class scheme served from train/basketball.yaml",
    description=(
        "The 5-class basketball-v2 scheme. The annotation UI reads its class list "
        "from here rather than hardcoding one, so appending a class upstream cannot "
        "leave the UI offering a stale set while the exporter uses the new one."
    ),
    tags=["Dataset export"],
)
def get_basketball_classes() -> BasketballClassesResponse:
    described = describe_basketball_classes()
    return BasketballClassesResponse(
        class_count=described["class_count"],
        classes=[BasketballClassRead(**item) for item in described["classes"]],
        number_bearing_class_id=described["number_bearing_class_id"],
        number_bearing_class_name=described["number_bearing_class_name"],
        jersey_number_rule=described["jersey_number_rule"],
        source_of_truth=described["source_of_truth"],
    )


@router.post(
    "/annotations",
    response_model=ManualAnnotationSaveResponse,
    summary="保存人工标注",
    description="保存人工修正后的 bbox 和球衣号码。若同一个 prediction_id + video_id 已存在标注，则自动更新已有标注。",
    tags=["人工标注"],
)
def save_annotation(payload: ManualAnnotationCreate, db: Session = Depends(get_db)):
    """Save one box.

    Two paths, decided by the sign of prediction_id:

    * prediction_id >= 0 - the box corrects a real YOLO prediction. At most one
      annotation may exist per prediction, so this UPSERTS, exactly as before.
    * prediction_id < 0 - the box was drawn by hand. Always INSERTS, under a
      freshly allocated unique id, so a frame can carry as many boxes as the
      image actually contains. Editing an existing hand-drawn box goes through
      PUT /annotations/{annotation_id}, not through here.
    """
    video = db.query(VideoAsset).filter(VideoAsset.id == payload.video_id).first()
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found")

    is_manual_box = payload.prediction_id < 0
    if is_manual_box:
        prediction_id = _allocate_manual_prediction_id(db, payload.video_id)
        matches: list[ManualAnnotation] = []
        row = ManualAnnotation(prediction_id=prediction_id)
    else:
        prediction_id = payload.prediction_id
        matches = _annotation_matches(db, prediction_id, payload.video_id)
        row = (
            next((item for item in matches if item.deleted_at is None), None)
            or (matches[0] if matches else None)
            or ManualAnnotation(prediction_id=prediction_id)
        )

    row.prediction_id = prediction_id
    row.video_id = payload.video_id
    row.frame_id = payload.frame_id
    row.track_id = payload.track_id
    row.class_id = payload.class_id
    row.image_path = payload.image_path
    row.image_url = payload.image_url
    row.original_bbox = json.dumps(payload.original_bbox)
    row.corrected_bbox = json.dumps(payload.corrected_bbox)
    row.original_number = payload.original_number
    row.corrected_number = payload.corrected_number
    row.label_status = payload.label_status
    row.deleted_at = None

    now = datetime.now(timezone.utc)
    for duplicate in matches:
        if duplicate.annotation_id != row.annotation_id:
            duplicate.deleted_at = now
            db.add(duplicate)

    db.add(row)
    if video is not None:
        video.status = "annotated"
        db.add(video)
    db.commit()
    db.refresh(row)
    return ManualAnnotationSaveResponse(annotation_id=row.annotation_id, status="saved")


@router.get("/annotations/{annotation_id}", response_model=ManualAnnotationRead, include_in_schema=False)
def get_annotation(annotation_id: int, db: Session = Depends(get_db)):
    return _annotation_to_schema(_get_annotation_or_404(db, annotation_id))


@router.put("/annotations/{annotation_id}", response_model=ManualAnnotationSaveResponse, include_in_schema=False)
def update_annotation(annotation_id: int, payload: ManualAnnotationUpdate, db: Session = Depends(get_db)):
    row = _get_annotation_or_404(db, annotation_id)
    video = db.query(VideoAsset).filter(VideoAsset.id == row.video_id).first()
    row.corrected_bbox = json.dumps(payload.corrected_bbox)
    # Reclassification is allowed here: correcting a referee mislabelled as a
    # player should not require deleting a box whose geometry is already right.
    row.class_id = payload.class_id
    row.corrected_number = payload.corrected_number
    row.label_status = payload.label_status
    db.add(row)
    if video is not None:
        video.status = "annotated"
        db.add(video)
    db.commit()
    db.refresh(row)
    return ManualAnnotationSaveResponse(annotation_id=row.annotation_id, status="updated")


@router.delete("/annotations/{annotation_id}", response_model=ManualAnnotationSaveResponse, include_in_schema=False)
def delete_annotation(annotation_id: int, db: Session = Depends(get_db)):
    row = _get_annotation_or_404(db, annotation_id)
    row.deleted_at = datetime.now(timezone.utc)
    db.add(row)
    db.commit()
    return ManualAnnotationSaveResponse(annotation_id=annotation_id, status="deleted")


@router.get(
    "/videos/{video_id}/annotations",
    response_model=list[ManualAnnotationRead],
    summary="查看视频人工标注",
    description="返回该视频已经保存的人工标注结果。",
    tags=["人工标注"],
)
def list_manual_annotations(video_id: int, db: Session = Depends(get_db)):
    video = db.query(VideoAsset).filter(VideoAsset.id == video_id).first()
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found")

    rows = (
        _active_annotation_query(db)
        .filter(ManualAnnotation.video_id == video_id)
        .order_by(ManualAnnotation.frame_id.asc(), ManualAnnotation.track_id.asc(), ManualAnnotation.annotation_id.asc())
        .all()
    )
    return [_annotation_to_schema(row) for row in rows]


@router.get(
    "/videos/{video_id}/annotation-progress",
    response_model=AnnotationProgressResponse,
    summary="查看人工标注进度",
    description="返回该视频当前已经标注的候选帧数量、剩余数量和进度百分比。",
    tags=["人工标注"],
)
def get_annotation_progress(video_id: int, db: Session = Depends(get_db)):
    video = db.query(VideoAsset).filter(VideoAsset.id == video_id).first()
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found")

    candidates = list_annotation_candidates(video_id=video_id, threshold=0.7, db=db)
    total_candidates = candidates.total_count
    annotated_count = _active_annotation_query(db).filter(ManualAnnotation.video_id == video_id).count()
    remaining_count = max(total_candidates - annotated_count, 0)
    progress_percent = round((annotated_count / total_candidates) * 100, 1) if total_candidates else 0.0
    return AnnotationProgressResponse(
        video_id=video_id,
        total_candidates=total_candidates,
        annotated_count=annotated_count,
        remaining_count=remaining_count,
        progress_percent=progress_percent,
    )


@router.post(
    "/videos/{video_id}/export-internal-dataset",
    response_model=InternalExportResponse,
    summary="导出修正后的数据集",
    description="导出该视频当前有效的人工修正标注，并生成可下载的数据集文件。",
    tags=["数据导出"],
)
def export_internal_dataset(video_id: int, db: Session = Depends(get_db)):
    video = db.query(VideoAsset).filter(VideoAsset.id == video_id).first()
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found")

    annotations = (
        _active_annotation_query(db)
        .filter(ManualAnnotation.video_id == video_id)
        .order_by(ManualAnnotation.frame_id.asc(), ManualAnnotation.track_id.asc(), ManualAnnotation.annotation_id.asc())
        .all()
    )
    if not annotations:
        raise HTTPException(status_code=400, detail="No annotations available for export.")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    export_id = f"{video_id}_{timestamp}"
    export_dir = _export_dir(video_id, export_id)
    images_dir = export_dir / "images"
    crops_dir = export_dir / "crops"

    rows: list[dict[str, object]] = []
    copied_images: set[str] = set()
    crop_files: list[str] = []

    for annotation in annotations:
        frame_path = _frame_image_path_from_annotation(annotation)
        image_filename = f"annotation_{annotation.annotation_id}_frame_{annotation.frame_id:06d}{frame_path.suffix or '.jpg'}"
        image_target = images_dir / image_filename
        if image_filename not in copied_images:
            shutil.copy2(frame_path, image_target)
            copied_images.add(image_filename)

        image_bytes = np.fromfile(str(frame_path), dtype=np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=500, detail=f"Failed to read source image for annotation {annotation.annotation_id}")

        corrected_bbox = _parse_bbox(annotation.corrected_bbox, field_name="corrected_bbox", prediction_id=annotation.prediction_id)
        x1, y1, x2, y2 = corrected_bbox
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            raise HTTPException(status_code=400, detail=f"Corrected bbox produced empty crop for annotation {annotation.annotation_id}")

        crop_filename = f"annotation_{annotation.annotation_id}_crop.jpg"
        crop_target = crops_dir / crop_filename
        _write_image_file(crop_target, crop)
        crop_files.append(crop_filename)

        width, height = _resolve_frame_dimensions(None, frame_path)
        row = {
            "annotation_id": annotation.annotation_id,
            "prediction_id": annotation.prediction_id,
            "video_id": annotation.video_id,
            "frame_id": annotation.frame_id,
            "track_id": annotation.track_id,
            "image_filename": image_filename,
            "crop_filename": crop_filename,
            "image_path": str(image_target.resolve()),
            "crop_path": str(crop_target.resolve()),
            "original_bbox": _parse_bbox(annotation.original_bbox, field_name="original_bbox", prediction_id=annotation.prediction_id),
            "corrected_bbox": corrected_bbox,
            "original_number": annotation.original_number,
            "corrected_number": annotation.corrected_number,
            "label_status": annotation.label_status,
            "image_width": width,
            "image_height": height,
            "created_at": annotation.created_at.isoformat(),
        }
        rows.append(row)

    csv_path = export_dir / "annotations.csv"
    fieldnames = [
        "annotation_id",
        "prediction_id",
        "video_id",
        "frame_id",
        "track_id",
        "image_filename",
        "crop_filename",
        "image_path",
        "crop_path",
        "original_bbox",
        "corrected_bbox",
        "original_number",
        "corrected_number",
        "label_status",
        "image_width",
        "image_height",
        "created_at",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                **row,
                "original_bbox": json.dumps(row["original_bbox"], ensure_ascii=False),
                "corrected_bbox": json.dumps(row["corrected_bbox"], ensure_ascii=False),
            })

    json_path = export_dir / "annotations.json"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    metadata = {
        "video_id": video_id,
        "export_time": datetime.now(timezone.utc).isoformat(),
        "annotation_count": len(rows),
        "image_count": len(copied_images),
        "crop_count": len(crop_files),
        "source": "jersey-active-learning",
        "dataset_format": "internal_v1",
    }
    metadata_path = export_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    video.status = "exported"
    db.add(video)
    db.commit()

    return InternalExportResponse(
        export_id=export_id,
        video_id=video_id,
        annotation_count=len(rows),
        export_path=str(export_dir.resolve()),
        files_generated=[
            str((export_dir / "images").resolve()),
            str((export_dir / "crops").resolve()),
            str(csv_path.resolve()),
            str(json_path.resolve()),
            str(metadata_path.resolve()),
        ],
    )


@router.get("/videos/{video_id}/exports", response_model=list[VideoExportRead], include_in_schema=False)
def list_video_exports(video_id: int, db: Session = Depends(get_db)):
    video = db.query(VideoAsset).filter(VideoAsset.id == video_id).first()
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found")

    exports: list[VideoExportRead] = []
    prefix = f"{video_id}_"
    for path in sorted(_exports_root().glob(f"{prefix}*"), key=lambda item: item.name, reverse=True):
        if not path.is_dir():
            continue
        created_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
        files_generated = [item.name for item in sorted(path.iterdir(), key=lambda item: item.name)]
        exports.append(
            VideoExportRead(
                export_id=path.name,
                video_id=video_id,
                export_path=str(path.resolve()),
                created_at=created_at,
                files_generated=files_generated,
            )
        )
    return exports


@router.get(
    "/exports/{export_id}/download",
    response_model=ExportDownloadResponse,
    summary="下载导出数据集",
    description="下载已经导出的内部数据集 zip 文件。",
    tags=["数据导出"],
)
def download_export(export_id: str, raw: bool = Query(default=False)):
    path = (_exports_root() / export_id).resolve()
    if not path.exists() or not path.is_dir():
        raise HTTPException(status_code=404, detail="Export not found")
    zip_path = _create_export_zip(path)
    if raw:
        return FileResponse(zip_path, filename=zip_path.name, media_type="application/zip")
    return ExportDownloadResponse(export_id=export_id, export_path=str(zip_path.resolve()))


@router.post("/videos/{video_id}/export-coco-dataset", response_model=CocoExportResponse, include_in_schema=False)
def export_coco_dataset(video_id: int, db: Session = Depends(get_db)):
    video = db.query(VideoAsset).filter(VideoAsset.id == video_id).first()
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found")

    annotations = (
        _active_annotation_query(db)
        .filter(ManualAnnotation.video_id == video_id)
        .order_by(ManualAnnotation.frame_id.asc(), ManualAnnotation.track_id.asc(), ManualAnnotation.annotation_id.asc())
        .all()
    )
    if not annotations:
        raise HTTPException(status_code=400, detail="No annotations available for COCO export.")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    export_id = f"{video_id}_{timestamp}"
    export_dir = _coco_export_dir(export_id)
    images_dir = export_dir / "images"
    annotations_dir = export_dir / "annotations"

    coco_images: list[dict[str, object]] = []
    coco_annotations: list[dict[str, object]] = []
    labels_rows: list[dict[str, object]] = []
    image_map: dict[tuple[int, int], dict[str, object]] = {}
    skipped_annotations: list[dict[str, object]] = []
    skipped_unclear_count = 0
    skipped_invalid_bbox_count = 0
    skipped_missing_image_count = 0
    coco_annotation_id = 1
    next_image_id = 1

    for annotation in annotations:
        if annotation.label_status == "unclear":
            skipped_unclear_count += 1
            skipped_annotations.append({
                "annotation_id": annotation.annotation_id,
                "frame_id": annotation.frame_id,
                "reason": "unclear_label_status",
            })
            continue

        try:
            frame_path = _frame_image_path_from_annotation(annotation)
        except FileNotFoundError:
            skipped_missing_image_count += 1
            skipped_annotations.append({
                "annotation_id": annotation.annotation_id,
                "frame_id": annotation.frame_id,
                "reason": "missing_image",
            })
            continue

        image_bytes = np.fromfile(str(frame_path), dtype=np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        if image is None:
            skipped_missing_image_count += 1
            skipped_annotations.append({
                "annotation_id": annotation.annotation_id,
                "frame_id": annotation.frame_id,
                "reason": "missing_image",
            })
            continue

        image_height, image_width = image.shape[:2]
        try:
            raw_bbox = _parse_bbox(annotation.corrected_bbox, field_name="corrected_bbox", prediction_id=annotation.prediction_id)
        except Exception:
            skipped_invalid_bbox_count += 1
            skipped_annotations.append({
                "annotation_id": annotation.annotation_id,
                "frame_id": annotation.frame_id,
                "reason": "invalid_bbox",
            })
            continue

        corrected_bbox, valid_bbox = _clamp_bbox_xyxy(raw_bbox, image_width, image_height)
        if not valid_bbox:
            skipped_invalid_bbox_count += 1
            skipped_annotations.append({
                "annotation_id": annotation.annotation_id,
                "frame_id": annotation.frame_id,
                "reason": "invalid_bbox",
            })
            continue

        x1, y1, x2, y2 = corrected_bbox
        coco_x = x1
        coco_y = y1
        coco_width = x2 - x1
        coco_height = y2 - y1
        area = coco_width * coco_height
        if coco_width <= 0 or coco_height <= 0 or area <= 0:
            skipped_invalid_bbox_count += 1
            skipped_annotations.append({
                "annotation_id": annotation.annotation_id,
                "frame_id": annotation.frame_id,
                "reason": "invalid_bbox",
            })
            continue

        image_key = (annotation.video_id, annotation.frame_id)
        image_entry = image_map.get(image_key)
        if image_entry is None:
            image_id = next_image_id
            next_image_id += 1
            image_filename = f"video_{annotation.video_id}_frame_{annotation.frame_id:06d}{frame_path.suffix or '.jpg'}"
            shutil.copy2(frame_path, images_dir / image_filename)
            image_entry = {
                "id": image_id,
                "file_name": image_filename,
                "width": int(image_width),
                "height": int(image_height),
            }
            image_map[image_key] = image_entry
            coco_images.append(image_entry)

        coco_annotations.append({
            "id": coco_annotation_id,
            "image_id": image_entry["id"],
            "category_id": coco_category_id(validate_class_id(annotation.class_id)),
            "class_id": validate_class_id(annotation.class_id),
            "class_name": class_name(validate_class_id(annotation.class_id)),
            "bbox": [coco_x, coco_y, coco_width, coco_height],
            "area": area,
            "iscrowd": 0,
            "source_annotation_id": annotation.annotation_id,
            "prediction_id": annotation.prediction_id,
            "video_id": annotation.video_id,
            "frame_id": annotation.frame_id,
            "track_id": annotation.track_id,
            "jersey_number": annotation.corrected_number,
            "label_status": annotation.label_status,
        })
        labels_rows.append({
            "coco_annotation_id": coco_annotation_id,
            "coco_image_id": image_entry["id"],
            "source_annotation_id": annotation.annotation_id,
            "prediction_id": annotation.prediction_id,
            "video_id": annotation.video_id,
            "frame_id": annotation.frame_id,
            "track_id": annotation.track_id,
            "image_filename": image_entry["file_name"],
            "jersey_number": annotation.corrected_number,
            "label_status": annotation.label_status,
            "bbox_x1": x1,
            "bbox_y1": y1,
            "bbox_x2": x2,
            "bbox_y2": y2,
            "coco_x": coco_x,
            "coco_y": coco_y,
            "coco_width": coco_width,
            "coco_height": coco_height,
        })
        coco_annotation_id += 1

    if not coco_annotations:
        detail = "No valid annotations available for COCO export."
        if skipped_unclear_count == len(annotations):
            detail = "No valid annotations available for COCO export."
        raise HTTPException(status_code=400, detail=detail)

    instances_payload = {
        "info": {
            "description": "Jersey Active Learning COCO Detection Dataset",
            "version": "coco_detection_v1",
            "year": 2026,
            "date_created": datetime.now(timezone.utc).isoformat(),
        },
        "licenses": [],
        "images": coco_images,
        "annotations": coco_annotations,
        # Derived from train/basketball.yaml, offset by one because COCO ids are
        # 1-based while YOLO class ids are 0-based. See coco_export.py.
        "categories": COCO_CATEGORIES,
    }
    instances_path = annotations_dir / "instances.json"
    instances_path.write_text(json.dumps(instances_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    labels_path = export_dir / "labels.csv"
    label_fields = [
        "coco_annotation_id",
        "coco_image_id",
        "source_annotation_id",
        "prediction_id",
        "video_id",
        "frame_id",
        "track_id",
        "image_filename",
        "jersey_number",
        "label_status",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "coco_x",
        "coco_y",
        "coco_width",
        "coco_height",
    ]
    with labels_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=label_fields)
        writer.writeheader()
        for row in labels_rows:
            writer.writerow(row)

    metadata = {
        "export_id": export_id,
        "video_id": video_id,
        "export_time": datetime.now(timezone.utc).isoformat(),
        "dataset_format": "coco_detection_v1",
        "source": "jersey-active-learning",
        "category_count": 1,
        "image_count": len(coco_images),
        "annotation_count": len(coco_annotations),
        "skipped_unclear_count": skipped_unclear_count,
        "skipped_invalid_bbox_count": skipped_invalid_bbox_count,
        "skipped_missing_image_count": skipped_missing_image_count,
        "skipped_annotations": skipped_annotations,
    }
    metadata_path = export_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    validation = validate_coco_dataset(instances_path)
    if not validation["valid"]:
        metadata["validation_errors"] = validation["errors"]
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        raise HTTPException(status_code=500, detail={"message": "COCO export validation failed", "errors": validation["errors"]})

    return CocoExportResponse(
        export_id=export_id,
        video_id=video_id,
        dataset_format="coco_detection_v1",
        image_count=len(coco_images),
        annotation_count=len(coco_annotations),
        skipped_unclear_count=skipped_unclear_count,
        skipped_invalid_bbox_count=skipped_invalid_bbox_count,
        skipped_missing_image_count=skipped_missing_image_count,
        export_path=str(export_dir.resolve()),
        files_generated=[
            str((export_dir / "images").resolve()),
            str(instances_path.resolve()),
            str(labels_path.resolve()),
            str(metadata_path.resolve()),
        ],
    )


@router.get("/videos/{video_id}/coco-exports", response_model=list[CocoExportRead], include_in_schema=False)
def list_coco_exports(video_id: int, db: Session = Depends(get_db)):
    video = db.query(VideoAsset).filter(VideoAsset.id == video_id).first()
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found")

    exports: list[CocoExportRead] = []
    prefix = f"{video_id}_"
    for path in sorted(_coco_exports_root().glob(f"{prefix}*"), key=lambda item: item.name, reverse=True):
        if not path.is_dir():
            continue
        metadata_path = path / "metadata.json"
        if not metadata_path.exists():
            continue
        metadata = _read_json_file(metadata_path)
        exports.append(
            CocoExportRead(
                export_id=path.name,
                video_id=video_id,
                dataset_format=str(metadata.get("dataset_format", "coco_detection_v1")),
                image_count=_int_metadata_value(metadata, "image_count"),
                annotation_count=_int_metadata_value(metadata, "annotation_count"),
                skipped_unclear_count=_int_metadata_value(metadata, "skipped_unclear_count"),
                skipped_invalid_bbox_count=_int_metadata_value(metadata, "skipped_invalid_bbox_count"),
                skipped_missing_image_count=_int_metadata_value(metadata, "skipped_missing_image_count"),
                export_path=str(path.resolve()),
                created_at=str(metadata.get("export_time", datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat())),
            )
        )
    return exports


@router.get("/coco-exports/{export_id}/download", response_model=ExportDownloadResponse, include_in_schema=False)
def download_coco_export(export_id: str):
    path = (_coco_exports_root() / export_id).resolve()
    if not path.exists() or not path.is_dir():
        raise HTTPException(status_code=404, detail="COCO export not found")
    zip_path = _create_export_zip(path)
    return ExportDownloadResponse(export_id=export_id, export_path=str(zip_path.resolve()))


@router.post("/videos/{video_id}/export-yolo-dataset", response_model=YoloExportResponse, include_in_schema=False)
def export_yolo_dataset(video_id: int, db: Session = Depends(get_db)):
    video = db.query(VideoAsset).filter(VideoAsset.id == video_id).first()
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found")

    annotations = (
        _active_annotation_query(db)
        .filter(ManualAnnotation.video_id == video_id)
        .order_by(ManualAnnotation.frame_id.asc(), ManualAnnotation.track_id.asc(), ManualAnnotation.annotation_id.asc())
        .all()
    )
    if not annotations:
        raise HTTPException(status_code=400, detail="No annotations available for export.")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    export_id = f"{video_id}_{timestamp}"
    export_dir = _yolo_export_dir(export_id)
    images_dir = export_dir / "images"
    labels_dir = export_dir / "labels"

    image_entries: dict[tuple[int, int], dict[str, object]] = {}
    label_lines: dict[str, list[str]] = {}
    annotation_count = 0
    skipped_unclear_count = 0
    skipped_invalid_bbox_count = 0
    skipped_missing_image_count = 0
    skipped_annotations: list[dict[str, object]] = []
    counts_by_class: dict[int, int] = {}
    jersey_numbers: list[dict[str, object]] = []

    for annotation in annotations:
        if annotation.label_status == "unclear":
            skipped_unclear_count += 1
            skipped_annotations.append({"annotation_id": annotation.annotation_id, "frame_id": annotation.frame_id, "reason": "unclear_label_status"})
            continue

        try:
            frame_path = _frame_image_path_from_annotation(annotation)
        except FileNotFoundError:
            skipped_missing_image_count += 1
            skipped_annotations.append({"annotation_id": annotation.annotation_id, "frame_id": annotation.frame_id, "reason": "missing_image"})
            continue

        image_bytes = np.fromfile(str(frame_path), dtype=np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        if image is None:
            skipped_missing_image_count += 1
            skipped_annotations.append({"annotation_id": annotation.annotation_id, "frame_id": annotation.frame_id, "reason": "missing_image"})
            continue

        image_height, image_width = image.shape[:2]
        try:
            raw_bbox = _parse_bbox(annotation.corrected_bbox, field_name="corrected_bbox", prediction_id=annotation.prediction_id)
        except Exception:
            skipped_invalid_bbox_count += 1
            skipped_annotations.append({"annotation_id": annotation.annotation_id, "frame_id": annotation.frame_id, "reason": "invalid_bbox"})
            continue

        corrected_bbox, valid_bbox = _clamp_bbox_xyxy(raw_bbox, image_width, image_height)
        if not valid_bbox:
            skipped_invalid_bbox_count += 1
            skipped_annotations.append({"annotation_id": annotation.annotation_id, "frame_id": annotation.frame_id, "reason": "invalid_bbox"})
            continue

        x1, y1, x2, y2 = corrected_bbox
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        if bbox_width <= 0 or bbox_height <= 0:
            skipped_invalid_bbox_count += 1
            skipped_annotations.append({"annotation_id": annotation.annotation_id, "frame_id": annotation.frame_id, "reason": "invalid_bbox"})
            continue

        x_center = ((x1 + x2) / 2.0) / image_width
        y_center = ((y1 + y2) / 2.0) / image_height
        normalized_width = bbox_width / image_width
        normalized_height = bbox_height / image_height
        normalized_values = [x_center, y_center, normalized_width, normalized_height]
        if any(value < 0.0 or value > 1.0 for value in normalized_values) or normalized_width <= 0 or normalized_height <= 0:
            skipped_invalid_bbox_count += 1
            skipped_annotations.append({"annotation_id": annotation.annotation_id, "frame_id": annotation.frame_id, "reason": "invalid_bbox"})
            continue

        image_key = (annotation.video_id, annotation.frame_id)
        image_filename = f"video_{annotation.video_id}_frame_{annotation.frame_id:06d}{frame_path.suffix or '.jpg'}"
        label_filename = f"video_{annotation.video_id}_frame_{annotation.frame_id:06d}.txt"
        if image_key not in image_entries:
            shutil.copy2(frame_path, images_dir / image_filename)
            image_entries[image_key] = {
                "image_filename": image_filename,
                "label_filename": label_filename,
                "image_width": image_width,
                "image_height": image_height,
            }
            label_lines[label_filename] = []

        annotation_class_id = validate_class_id(annotation.class_id)
        line_index = len(label_lines[label_filename])
        label_lines[label_filename].append(
            f"{annotation_class_id} {x_center:.6f} {y_center:.6f} "
            f"{normalized_width:.6f} {normalized_height:.6f}"
        )
        counts_by_class[annotation_class_id] = counts_by_class.get(annotation_class_id, 0) + 1
        annotation_count += 1

        # The number is NOT a class and never enters a label file. It is written
        # to jersey_numbers.json, keyed to the label line it describes, so it can
        # serve as OCR ground truth without polluting detector training data.
        if annotation_class_id == NUMBER_BEARING_CLASS_ID and annotation.corrected_number:
            normalized_number = clean_jersey_number(annotation.corrected_number)
            if normalized_number is not None:
                jersey_numbers.append(
                    {
                        "label_file": label_filename,
                        "image_file": image_filename,
                        "label_line_index": line_index,
                        "jersey_number": normalized_number,
                        "video_id": int(annotation.video_id),
                        "frame_id": int(annotation.frame_id),
                        "track_id": int(annotation.track_id),
                        "source_annotation_id": int(annotation.annotation_id),
                        "bbox_xyxy": corrected_bbox,
                    }
                )

    if annotation_count == 0:
        raise HTTPException(status_code=400, detail="No valid annotations available for export.")

    for label_filename, lines in label_lines.items():
        (labels_dir / label_filename).write_text("\n".join(lines) + "\n", encoding="utf-8")

    dataset_yaml = export_dir / "dataset.yaml"
    # 'path', 'train' and 'val' describe THIS export and legitimately differ from
    # train/basketball.yaml; nc and names must not, and are rendered from it.
    dataset_yaml.write_text(
        "\n".join(
            [
                f'path: "{str(export_dir.resolve()).replace(chr(92), "/")}"',
                "train: images",
                "val: images",
            ]
            + render_names_block()
            + [""]
        ),
        encoding="utf-8",
    )
    # Refuse to ship a dataset whose class indices disagree with the code that
    # will consume it - the failure is otherwise completely silent.
    assert_matches_basketball_yaml(dataset_yaml)

    classes_txt = export_dir / "classes.txt"
    classes_txt.write_text(classes_txt_content(), encoding="utf-8")

    jersey_numbers_path = export_dir / JERSEY_NUMBERS_FILENAME
    jersey_numbers_path.write_text(
        json.dumps(
            {
                "schema": JERSEY_NUMBERS_SCHEMA,
                "class_id": NUMBER_BEARING_CLASS_ID,
                "class_name": class_name(NUMBER_BEARING_CLASS_ID),
                "normalization": (
                    "src/jersey.py:clean_jersey_number - digits only, max 2, '00' "
                    "rejected, leading zeros stripped ('01' -> '1'), bare '0' kept"
                ),
                "usage": (
                    "OCR ground truth. Join to a YOLO label line via "
                    "(label_file, label_line_index); that line is always a jersey line."
                ),
                "entry_count": len(jersey_numbers),
                "entries": jersey_numbers,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    annotation_counts_by_class = {
        BASKETBALL_CLASS_NAMES[class_id]: count for class_id, count in sorted(counts_by_class.items())
    }
    metadata = {
        "export_id": export_id,
        "video_id": video_id,
        "export_time": datetime.now(timezone.utc).isoformat(),
        "dataset_format": "yolo_detection_v1",
        "source": "jersey-active-learning",
        "target_project": SPORT_CV_TARGET_PROJECT,
        "detected_framework": SPORT_CV_DETECTED_FRAMEWORK,
        "detected_model_version": SPORT_CV_DETECTED_MODEL_VERSION,
        "class_count": CLASS_COUNT,
        "class_mapping": class_mapping_for_metadata(),
        "annotation_counts_by_class": annotation_counts_by_class,
        "classes_with_no_annotations": [
            name for index, name in sorted(BASKETBALL_CLASS_NAMES.items())
            if counts_by_class.get(index, 0) == 0
        ],
        "jersey_number_count": len(jersey_numbers),
        "jersey_numbers_file": JERSEY_NUMBERS_FILENAME,
        "image_count": len(image_entries),
        "annotation_count": annotation_count,
        "label_file_count": len(label_lines),
        "skipped_unclear_count": skipped_unclear_count,
        "skipped_invalid_bbox_count": skipped_invalid_bbox_count,
        "skipped_missing_image_count": skipped_missing_image_count,
        "skipped_annotations": skipped_annotations,
        "train_val_split": False,
        "sport_cv_compatible": True,
        "compatibility_notes": [
            "Export targets Ultralytics YOLO compatibility used by Sport-CV.",
            "Class scheme is read from basketball-v2 train/basketball.yaml and verified "
            "against it field by field before this file is written.",
            "The jersey NUMBER is not a class. It is exported separately in "
            f"{JERSEY_NUMBERS_FILENAME} as OCR ground truth.",
            "THIS EXPORT HAS NO TRAIN/VAL SPLIT: train and val both point at images/, so "
            "training on it measures nothing. Use POST /dataset-versions instead, which "
            "splits by video or track and refuses to let one player's frames land on both "
            "sides.",
        ],
    }
    metadata_path = export_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    validation = validate_yolo_dataset(export_dir)
    if not validation["valid"]:
        metadata["validation_errors"] = validation["errors"]
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        raise HTTPException(status_code=500, detail={"message": "YOLO export validation failed", "errors": validation["errors"]})

    return YoloExportResponse(
        export_id=export_id,
        video_id=video_id,
        dataset_format="yolo_detection_v1",
        target_project=SPORT_CV_TARGET_PROJECT,
        detected_framework=SPORT_CV_DETECTED_FRAMEWORK,
        detected_model_version=SPORT_CV_DETECTED_MODEL_VERSION,
        image_count=len(image_entries),
        annotation_count=annotation_count,
        label_file_count=len(label_lines),
        class_count=CLASS_COUNT,
        annotation_counts_by_class=annotation_counts_by_class,
        jersey_number_count=len(jersey_numbers),
        skipped_unclear_count=skipped_unclear_count,
        skipped_invalid_bbox_count=skipped_invalid_bbox_count,
        skipped_missing_image_count=skipped_missing_image_count,
        sport_cv_compatible=True,
        export_path=str(export_dir.resolve()),
        files_generated=[
            str((export_dir / "images").resolve()),
            str((export_dir / "labels").resolve()),
            str(dataset_yaml.resolve()),
            str(classes_txt.resolve()),
            str(jersey_numbers_path.resolve()),
            str(metadata_path.resolve()),
        ],
    )


@router.get("/videos/{video_id}/yolo-exports", response_model=list[YoloExportRead], include_in_schema=False)
def list_yolo_exports(video_id: int, db: Session = Depends(get_db)):
    video = db.query(VideoAsset).filter(VideoAsset.id == video_id).first()
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found")

    exports: list[YoloExportRead] = []
    prefix = f"{video_id}_"
    for path in sorted(_yolo_exports_root().glob(f"{prefix}*"), key=lambda item: item.name, reverse=True):
        if not path.is_dir():
            continue
        metadata_path = path / "metadata.json"
        if not metadata_path.exists():
            continue
        metadata = _read_json_file(metadata_path)
        exports.append(
            YoloExportRead(
                export_id=path.name,
                video_id=video_id,
                dataset_format=str(metadata.get("dataset_format", "yolo_detection_v1")),
                target_project=str(metadata.get("target_project", SPORT_CV_TARGET_PROJECT)),
                detected_framework=str(metadata.get("detected_framework", SPORT_CV_DETECTED_FRAMEWORK)),
                detected_model_version=str(metadata.get("detected_model_version", SPORT_CV_DETECTED_MODEL_VERSION)),
                image_count=_int_metadata_value(metadata, "image_count"),
                annotation_count=_int_metadata_value(metadata, "annotation_count"),
                label_file_count=_int_metadata_value(metadata, "label_file_count"),
                class_count=_int_metadata_value(metadata, "class_count"),
                annotation_counts_by_class={
                    str(key): int(value)
                    for key, value in (metadata.get("annotation_counts_by_class") or {}).items()
                },
                jersey_number_count=_int_metadata_value(metadata, "jersey_number_count"),
                skipped_unclear_count=_int_metadata_value(metadata, "skipped_unclear_count"),
                skipped_invalid_bbox_count=_int_metadata_value(metadata, "skipped_invalid_bbox_count"),
                skipped_missing_image_count=_int_metadata_value(metadata, "skipped_missing_image_count"),
                sport_cv_compatible=bool(metadata.get("sport_cv_compatible", True)),
                export_path=str(path.resolve()),
                created_at=str(metadata.get("export_time", datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat())),
            )
        )
    return exports


@router.get("/yolo-exports/{export_id}/download", response_model=ExportDownloadResponse, include_in_schema=False)
def download_yolo_export(export_id: str):
    path = (_yolo_exports_root() / export_id).resolve()
    if not path.exists() or not path.is_dir():
        raise HTTPException(status_code=404, detail="YOLO export not found")
    zip_path = _create_export_zip(path)
    return ExportDownloadResponse(export_id=export_id, export_path=str(zip_path.resolve()))


@router.post("/dataset-versions", response_model=DatasetVersionRead, include_in_schema=False)
def create_dataset_version(payload: DatasetVersionCreate, db: Session = Depends(get_db)):
    record = build_dataset_version(
        db,
        version_name=payload.version_name,
        description=payload.description,
        video_ids=payload.video_ids,
        train_ratio=payload.train_ratio,
        val_ratio=payload.val_ratio,
        test_ratio=payload.test_ratio,
        random_seed=payload.random_seed,
        split_strategy=payload.split_strategy,
        temporal_block_size=payload.temporal_block_size,
        strict_split=payload.strict_split,
    )
    return DatasetVersionRead(**record)


@router.get("/dataset-versions", response_model=list[DatasetVersionSummaryRead], include_in_schema=False)
def get_dataset_versions(db: Session = Depends(get_db)):
    return [DatasetVersionSummaryRead(**item) for item in list_dataset_versions(db)]


@router.get("/dataset-versions/{dataset_version_id}", response_model=DatasetVersionRead, include_in_schema=False)
def get_dataset_version(dataset_version_id: int, db: Session = Depends(get_db)):
    row = db.query(DatasetVersion).filter(DatasetVersion.dataset_version_id == dataset_version_id).first()
    if row is None:
        raise HTTPException(status_code=404, detail="Dataset version not found")
    validation_result = None
    if row.validation_result_json:
        try:
            validation_result = json.loads(row.validation_result_json)
        except Exception:
            validation_result = row.validation_result_json
    return DatasetVersionRead(
        dataset_version_id=row.dataset_version_id,
        version_name=row.version_name,
        description=row.description,
        dataset_format=row.dataset_format,
        task_type=row.task_type,
        split_strategy=row.split_strategy,
        random_seed=row.random_seed,
        train_ratio=row.train_ratio,
        val_ratio=row.val_ratio,
        test_ratio=row.test_ratio,
        temporal_block_size=row.temporal_block_size,
        strict_split=bool(row.strict_split),
        source_video_ids=json.loads(row.source_video_ids),
        source_annotation_count=row.source_annotation_count,
        exported_image_count=row.exported_image_count,
        exported_annotation_count=row.exported_annotation_count,
        train_image_count=row.train_image_count,
        val_image_count=row.val_image_count,
        test_image_count=row.test_image_count,
        train_annotation_count=row.train_annotation_count,
        val_annotation_count=row.val_annotation_count,
        test_annotation_count=row.test_annotation_count,
        export_path=row.export_path,
        metadata_path=row.metadata_path,
        status=row.status,
        validation_result_json=validation_result,
        source_data_fingerprint=row.source_data_fingerprint,
        created_at=row.created_at,
        completed_at=row.completed_at,
    )


@router.get("/dataset-versions/{dataset_version_id}/manifest", response_model=DatasetVersionManifestResponse, include_in_schema=False)
def get_dataset_version_manifest(dataset_version_id: int, db: Session = Depends(get_db)):
    row = db.query(DatasetVersion).filter(DatasetVersion.dataset_version_id == dataset_version_id).first()
    if row is None:
        raise HTTPException(status_code=404, detail="Dataset version not found")
    return DatasetVersionManifestResponse(
        dataset_version_id=row.dataset_version_id,
        version_name=row.version_name,
        manifest=load_dataset_manifest(row),
    )


@router.get("/dataset-versions/{dataset_version_id}/split-report", response_model=DatasetVersionSplitReportResponse, include_in_schema=False)
def get_dataset_version_split_report(dataset_version_id: int, db: Session = Depends(get_db)):
    row = db.query(DatasetVersion).filter(DatasetVersion.dataset_version_id == dataset_version_id).first()
    if row is None:
        raise HTTPException(status_code=404, detail="Dataset version not found")
    split_report = json.loads((Path(row.export_path) / "split_report.json").read_text(encoding="utf-8"))
    return DatasetVersionSplitReportResponse(
        dataset_version_id=row.dataset_version_id,
        version_name=row.version_name,
        split_report=split_report,
    )


@router.get("/dataset-versions/{dataset_version_id}/download", response_model=ExportDownloadResponse, include_in_schema=False)
def download_dataset_version(dataset_version_id: int, db: Session = Depends(get_db)):
    row = db.query(DatasetVersion).filter(DatasetVersion.dataset_version_id == dataset_version_id).first()
    if row is None:
        raise HTTPException(status_code=404, detail="Dataset version not found")
    export_root = Path(row.export_path).resolve()
    if not export_root.exists() or not export_root.is_dir():
        raise HTTPException(status_code=404, detail="Dataset version export folder not found")
    zip_path = _create_export_zip(export_root)
    return ExportDownloadResponse(export_id=row.version_name, export_path=str(zip_path.resolve()))


@router.post("/dataset-versions/{dataset_version_id}/rebuild", response_model=DatasetVersionRead, include_in_schema=False)
def rebuild_dataset_version_route(dataset_version_id: int, db: Session = Depends(get_db)):
    row = db.query(DatasetVersion).filter(DatasetVersion.dataset_version_id == dataset_version_id).first()
    if row is None:
        raise HTTPException(status_code=404, detail="Dataset version not found")
    record = rebuild_dataset_version(db, row)
    return DatasetVersionRead(**record)


@router.post(
    "/upload-video",
    response_model=UploadVideoResponse,
    summary="上传视频并自动抽帧",
    description="上传篮球视频后，后端会自动保存视频并按设定间隔抽取视频帧。",
    tags=["视频管理"],
)
async def upload_video(
    file: UploadFile = File(...),
    frame_interval: int = Form(30),
    db: Session = Depends(get_db),
):
    started_at = time.perf_counter()
    ensure_data_dirs()
    filename = file.filename or "uploaded_video.mp4"
    safe_name = f"{uuid4().hex}_{Path(filename).name}"
    target = video_output_path(safe_name)

    LOGGER.info("upload started filename=%s frame_interval=%s", filename, frame_interval)
    with target.open("wb") as buffer:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            buffer.write(chunk)
    await file.close()
    LOGGER.info("video saved path=%s", target.resolve())

    try:
        asset = VideoAsset(filename=filename, path=str(Path(target).resolve()))
        db.add(asset)
        db.flush()

        LOGGER.info("frame extraction started video_id=%s", asset.id)
        frames_dir = frame_output_dir(asset.id)
        extracted = extract_frames(target, frames_dir, frame_interval=frame_interval)
        LOGGER.info("frame extraction finished video_id=%s extracted_frames=%s", asset.id, len(extracted))

        # 抽帧参数留痕。下游（jersey_train 的 mine）需要 frame_interval 来定
        # stride；decode_path 用来判断落盘的帧号数的是原视频还是转码产物。
        # 不记下来的话事后只能靠帧号差分猜 interval，fps 和解码路径无从查起。
        asset.fps = probe_fps(target) or None
        asset.frame_interval = int(frame_interval)
        asset.decode_path = decode_path()

        frame_rows = [
            VideoFrame(
                video_id=asset.id,
                frame_id=frame_id,
                frame_path=str(frame_path.resolve()),
                image_width=image_width,
                image_height=image_height,
            )
            for frame_id, frame_path, image_width, image_height in extracted
        ]
        db.add_all(frame_rows)

        asset.status = "frames_extracted"
        db.commit()
        db.refresh(asset)
        LOGGER.info("database commit finished video_id=%s", asset.id)
    except FrameCountMismatchError as exc:
        # VFR 素材。帧号已经不可信，接受这次上传只会产出一批对不回原视频的帧。
        # 连视频文件一起删掉：留着它只会让人以为「传上去了，重跑一下就好」。
        db.rollback()
        target.unlink(missing_ok=True)
        LOGGER.error("upload rejected, frame count mismatch: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        db.rollback()
        raise

    elapsed = round(time.perf_counter() - started_at, 3)

    return UploadVideoResponse(
        video_id=asset.id,
        original_filename=asset.filename,
        saved_video_path=asset.path,
        frame_count=len(extracted),
        elapsed_seconds=elapsed,
        status=asset.status,
    )


@router.get("/api/videos", response_model=list[VideoAssetRead], include_in_schema=False)
def list_videos_api_alias(db: Session = Depends(get_db)):
    return list_videos(db)


@router.get("/api/videos/{video_id}/frames", response_model=list[VideoFrameRead], include_in_schema=False)
def list_frames_api_alias(video_id: int, db: Session = Depends(get_db)):
    return list_video_frames(video_id, db)


@router.get("/api/videos/{video_id}/predictions", response_model=list[JerseyPredictionRead], include_in_schema=False)
def list_predictions_api_alias(video_id: int, db: Session = Depends(get_db)):
    return list_video_predictions(video_id, db)


@router.get("/api/videos/{video_id}/failure-frames", response_model=FailureFramesResponse, include_in_schema=False)
def list_failure_frames_api_alias(video_id: int, threshold: float = 0.7, db: Session = Depends(get_db)):
    return list_failure_frames(video_id=video_id, threshold=threshold, db=db)


@router.get("/api/videos/{video_id}/inconsistent-frames", response_model=InconsistentFramesResponse, include_in_schema=False)
def list_inconsistent_frames_api_alias(video_id: int, db: Session = Depends(get_db)):
    return list_inconsistent_frames(video_id=video_id, db=db)


@router.get("/api/videos/{video_id}/annotation-candidates", response_model=AnnotationCandidatesResponse, include_in_schema=False)
def list_annotation_candidates_api_alias(video_id: int, threshold: float = 0.7, db: Session = Depends(get_db)):
    return list_annotation_candidates(video_id=video_id, threshold=threshold, db=db)


@router.get("/api/jobs/{job_id}", response_model=ProcessingJobRead, include_in_schema=False)
def get_job_api_alias(job_id: str, db: Session = Depends(get_db)):
    return get_job(job_id=job_id, db=db)


@router.post("/api/annotations", response_model=ManualAnnotationSaveResponse, include_in_schema=False)
def save_annotation_api_alias(payload: ManualAnnotationCreate, db: Session = Depends(get_db)):
    return save_annotation(payload=payload, db=db)


@router.get("/api/annotations/{annotation_id}", response_model=ManualAnnotationRead, include_in_schema=False)
def get_annotation_api_alias(annotation_id: int, db: Session = Depends(get_db)):
    return get_annotation(annotation_id=annotation_id, db=db)


@router.put("/api/annotations/{annotation_id}", response_model=ManualAnnotationSaveResponse, include_in_schema=False)
def update_annotation_api_alias(annotation_id: int, payload: ManualAnnotationUpdate, db: Session = Depends(get_db)):
    return update_annotation(annotation_id=annotation_id, payload=payload, db=db)


@router.delete("/api/annotations/{annotation_id}", response_model=ManualAnnotationSaveResponse, include_in_schema=False)
def delete_annotation_api_alias(annotation_id: int, db: Session = Depends(get_db)):
    return delete_annotation(annotation_id=annotation_id, db=db)


@router.get("/api/videos/{video_id}/annotations", response_model=list[ManualAnnotationRead], include_in_schema=False)
def list_manual_annotations_api_alias(video_id: int, db: Session = Depends(get_db)):
    return list_manual_annotations(video_id=video_id, db=db)


@router.get("/api/videos/{video_id}/annotation-progress", response_model=AnnotationProgressResponse, include_in_schema=False)
def get_annotation_progress_api_alias(video_id: int, db: Session = Depends(get_db)):
    return get_annotation_progress(video_id=video_id, db=db)


@router.post("/api/videos/{video_id}/export-internal-dataset", response_model=InternalExportResponse, include_in_schema=False)
def export_internal_dataset_api_alias(video_id: int, db: Session = Depends(get_db)):
    return export_internal_dataset(video_id=video_id, db=db)


@router.get("/api/videos/{video_id}/exports", response_model=list[VideoExportRead], include_in_schema=False)
def list_video_exports_api_alias(video_id: int, db: Session = Depends(get_db)):
    return list_video_exports(video_id=video_id, db=db)


@router.get("/api/exports/{export_id}/download", response_model=ExportDownloadResponse, include_in_schema=False)
def download_export_api_alias(export_id: str):
    return download_export(export_id=export_id)


@router.post("/api/videos/{video_id}/export-coco-dataset", response_model=CocoExportResponse, include_in_schema=False)
def export_coco_dataset_api_alias(video_id: int, db: Session = Depends(get_db)):
    return export_coco_dataset(video_id=video_id, db=db)


@router.get("/api/videos/{video_id}/coco-exports", response_model=list[CocoExportRead], include_in_schema=False)
def list_coco_exports_api_alias(video_id: int, db: Session = Depends(get_db)):
    return list_coco_exports(video_id=video_id, db=db)


@router.get("/api/coco-exports/{export_id}/download", response_model=ExportDownloadResponse, include_in_schema=False)
def download_coco_export_api_alias(export_id: str):
    return download_coco_export(export_id=export_id)


@router.post("/api/videos/{video_id}/export-yolo-dataset", response_model=YoloExportResponse, include_in_schema=False)
def export_yolo_dataset_api_alias(video_id: int, db: Session = Depends(get_db)):
    return export_yolo_dataset(video_id=video_id, db=db)


@router.get("/api/videos/{video_id}/yolo-exports", response_model=list[YoloExportRead], include_in_schema=False)
def list_yolo_exports_api_alias(video_id: int, db: Session = Depends(get_db)):
    return list_yolo_exports(video_id=video_id, db=db)


@router.get("/api/yolo-exports/{export_id}/download", response_model=ExportDownloadResponse, include_in_schema=False)
def download_yolo_export_api_alias(export_id: str):
    return download_yolo_export(export_id=export_id)


@router.post("/api/dataset-versions", response_model=DatasetVersionRead, include_in_schema=False)
def create_dataset_version_api_alias(payload: DatasetVersionCreate, db: Session = Depends(get_db)):
    return create_dataset_version(payload=payload, db=db)


@router.get("/api/dataset-versions", response_model=list[DatasetVersionSummaryRead], include_in_schema=False)
def get_dataset_versions_api_alias(db: Session = Depends(get_db)):
    return get_dataset_versions(db=db)


@router.get("/api/dataset-versions/{dataset_version_id}", response_model=DatasetVersionRead, include_in_schema=False)
def get_dataset_version_api_alias(dataset_version_id: int, db: Session = Depends(get_db)):
    return get_dataset_version(dataset_version_id=dataset_version_id, db=db)


@router.get("/api/dataset-versions/{dataset_version_id}/manifest", response_model=DatasetVersionManifestResponse, include_in_schema=False)
def get_dataset_version_manifest_api_alias(dataset_version_id: int, db: Session = Depends(get_db)):
    return get_dataset_version_manifest(dataset_version_id=dataset_version_id, db=db)


@router.get("/api/dataset-versions/{dataset_version_id}/split-report", response_model=DatasetVersionSplitReportResponse, include_in_schema=False)
def get_dataset_version_split_report_api_alias(dataset_version_id: int, db: Session = Depends(get_db)):
    return get_dataset_version_split_report(dataset_version_id=dataset_version_id, db=db)


@router.get("/api/dataset-versions/{dataset_version_id}/download", response_model=ExportDownloadResponse, include_in_schema=False)
def download_dataset_version_api_alias(dataset_version_id: int, db: Session = Depends(get_db)):
    return download_dataset_version(dataset_version_id=dataset_version_id, db=db)


@router.post("/api/dataset-versions/{dataset_version_id}/rebuild", response_model=DatasetVersionRead, include_in_schema=False)
def rebuild_dataset_version_api_alias(dataset_version_id: int, db: Session = Depends(get_db)):
    return rebuild_dataset_version_route(dataset_version_id=dataset_version_id, db=db)


# /api/... 的四条出框别名同样摘掉。别名和本名必须同进同退 —— 只摘本名的话
# 前端换个前缀就能把停用的路径又调起来，而 source 列会照样记成 real/yolo/mock。
def run_mock_predictions_api_alias(video_id: int, db: Session = Depends(get_db)):
    """[已停用] 原路由 POST /api/videos/{video_id}/run-mock-jersey-recognition"""
    return run_mock_jersey_recognition(video_id, db)


def run_mock_predictions_async_api_alias(video_id: int, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """[已停用] 原路由 POST /api/videos/{video_id}/run-mock-jersey-recognition-async"""
    return run_mock_jersey_recognition_async(video_id=video_id, background_tasks=background_tasks, db=db)


def run_predictions_api_alias(video_id: int, db: Session = Depends(get_db)):
    """[已停用] 原路由 POST /api/videos/{video_id}/run-jersey-recognition"""
    return run_jersey_recognition(video_id=video_id, db=db)


def run_predictions_async_api_alias(video_id: int, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """[已停用] 原路由 POST /api/videos/{video_id}/run-jersey-recognition-async"""
    return run_jersey_recognition_async(video_id=video_id, background_tasks=background_tasks, db=db)


@router.get("/api/videos/{video_id}/recognition-summary", response_model=RecognitionSummaryResponse, include_in_schema=False)
def recognition_summary_api_alias(video_id: int, threshold: float = 0.7, db: Session = Depends(get_db)):
    return get_recognition_summary(video_id=video_id, threshold=threshold, db=db)


@router.post("/api/videos/upload", response_model=UploadVideoResponse, include_in_schema=False)
async def upload_video_api_alias(
    file: UploadFile = File(...),
    frame_interval: int = Form(30),
    db: Session = Depends(get_db),
):
    return await upload_video(file=file, frame_interval=frame_interval, db=db)
