import json
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.core.config import settings
from app.db.session import SessionLocal
from app.main import app
from app.models.dataset import ManualAnnotation, VideoAsset, VideoFrame
from app.services.coco_export import validate_coco_dataset
from app.services.storage import ensure_data_dirs


def _write_frame(path: Path, width: int = 1920, height: int = 1080) -> None:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = (30, 60, 90)
    cv2.imwrite(str(path), image)


def _create_video_with_annotations(tmp_path, monkeypatch, annotation_specs):
    data_dir = tmp_path / "data"
    monkeypatch.setattr(settings, "data_dir", data_dir)
    ensure_data_dirs()

    db = SessionLocal()
    try:
        video = VideoAsset(filename="test.mp4", path=str((data_dir / "videos" / "test.mp4").resolve()), status="frames_extracted")
        db.add(video)
        db.flush()

        frames_dir = data_dir / "frames" / str(video.id)
        frames_dir.mkdir(parents=True, exist_ok=True)
        frame_path = frames_dir / "frame_000000.jpg"
        _write_frame(frame_path)

        frame_row = VideoFrame(
            video_id=video.id,
            frame_id=0,
            frame_path=str(frame_path.resolve()),
            image_width=1920,
            image_height=1080,
        )
        db.add(frame_row)
        db.flush()

        created_ids = []
        for index, spec in enumerate(annotation_specs, start=1):
            image_path = str(frame_path.resolve())
            if spec.get("missing_image"):
                image_path = str((frames_dir / f"missing_{index}.jpg").resolve())
            annotation = ManualAnnotation(
                prediction_id=spec.get("prediction_id", 1000 + index),
                video_id=video.id,
                frame_id=spec.get("frame_id", 0),
                track_id=spec.get("track_id", index),
                image_path=image_path,
                image_url=f"/static/frames/{video.id}/frame_000000.jpg",
                original_bbox=json.dumps(spec.get("original_bbox", [100, 200, 300, 500])),
                corrected_bbox=json.dumps(spec.get("corrected_bbox", [100, 200, 300, 500])),
                original_number=spec.get("original_number", "23"),
                corrected_number=spec.get("corrected_number", "23"),
                label_status=spec.get("label_status", "correct"),
                deleted_at=spec.get("deleted_at"),
            )
            db.add(annotation)
            db.flush()
            created_ids.append(annotation.annotation_id)

        db.commit()
        return video.id, created_ids
    finally:
        db.close()


def test_coco_export_smoke_bbox_conversion_and_duplicate_frames(tmp_path, monkeypatch):
    video_id, _ = _create_video_with_annotations(
        tmp_path,
        monkeypatch,
        [
            {"prediction_id": 2001, "track_id": 1, "corrected_bbox": [100, 200, 300, 500], "label_status": "correct"},
            {"prediction_id": 2002, "track_id": 2, "corrected_bbox": [400, 100, 600, 250], "label_status": "wrong_bbox"},
        ],
    )
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post(f"/videos/{video_id}/export-coco-dataset")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["image_count"] == 1
    assert payload["annotation_count"] == 2

    export_path = Path(payload["export_path"])
    instances_path = export_path / "annotations" / "instances.json"
    labels_path = export_path / "labels.csv"
    metadata_path = export_path / "metadata.json"
    assert instances_path.exists()
    assert labels_path.exists()
    assert metadata_path.exists()

    instances = json.loads(instances_path.read_text(encoding="utf-8"))
    assert len(instances["images"]) == 1
    assert len(instances["annotations"]) == 2
    assert instances["annotations"][0]["bbox"] == [100, 200, 200, 300]
    assert instances["annotations"][0]["area"] == 60000
    assert validate_coco_dataset(instances_path) == {"valid": True, "errors": []}


def test_coco_export_skips_unclear_invalid_and_missing_images(tmp_path, monkeypatch):
    video_id, _ = _create_video_with_annotations(
        tmp_path,
        monkeypatch,
        [
            {"prediction_id": 3001, "track_id": 1, "corrected_bbox": [100, 200, 300, 500], "label_status": "correct"},
            {"prediction_id": 3002, "track_id": 2, "corrected_bbox": [100, 200, 300, 500], "label_status": "unclear"},
            {"prediction_id": 3003, "track_id": 3, "corrected_bbox": [100, 200, 100, 500], "label_status": "wrong_bbox"},
            {"prediction_id": 3004, "track_id": 4, "frame_id": 999, "corrected_bbox": [100, 200, 300, 500], "label_status": "wrong_number", "missing_image": True},
        ],
    )
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post(f"/videos/{video_id}/export-coco-dataset")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["annotation_count"] == 1
    assert payload["skipped_unclear_count"] == 1
    assert payload["skipped_invalid_bbox_count"] == 1
    assert payload["skipped_missing_image_count"] == 1

    metadata = json.loads((Path(payload["export_path"]) / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["skipped_unclear_count"] == 1
    assert metadata["skipped_invalid_bbox_count"] == 1
    assert metadata["skipped_missing_image_count"] == 1
    assert len(metadata["skipped_annotations"]) == 3


def test_coco_export_rejects_when_no_valid_annotations(tmp_path, monkeypatch):
    video_id, _ = _create_video_with_annotations(
        tmp_path,
        monkeypatch,
        [
            {"prediction_id": 4001, "track_id": 1, "corrected_bbox": [100, 200, 300, 500], "label_status": "unclear"},
            {"prediction_id": 4002, "track_id": 2, "corrected_bbox": [100, 200, 100, 500], "label_status": "wrong_bbox"},
        ],
    )
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post(f"/videos/{video_id}/export-coco-dataset")
    assert response.status_code == 400, response.text
    assert response.json()["detail"] == "No valid annotations available for COCO export."
