import json
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.core.config import settings
from app.db.session import SessionLocal
from app.main import app
from app.models.dataset import ManualAnnotation, VideoAsset, VideoFrame
from app.services.storage import ensure_data_dirs
from app.services.yolo_export import validate_yolo_dataset


def _write_frame(path: Path, width: int = 1000, height: int = 1000) -> None:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = (60, 90, 120)
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

        frame_paths: dict[int, Path] = {}
        for spec in annotation_specs:
            frame_id = spec.get("frame_id", 0)
            if spec.get("missing_image"):
                continue
            if frame_id not in frame_paths:
                frame_path = frames_dir / f"frame_{frame_id:06d}.jpg"
                _write_frame(frame_path)
                frame_paths[frame_id] = frame_path
                db.add(
                    VideoFrame(
                        video_id=video.id,
                        frame_id=frame_id,
                        frame_path=str(frame_path.resolve()),
                        image_width=1000,
                        image_height=1000,
                    )
                )
        db.flush()

        for index, spec in enumerate(annotation_specs, start=1):
            frame_id = spec.get("frame_id", 0)
            frame_path = frame_paths.get(frame_id)
            image_path = str(frame_path.resolve()) if frame_path is not None else str((frames_dir / f"missing_{index}.jpg").resolve())
            if spec.get("missing_image"):
                image_path = str((frames_dir / f"missing_{index}.jpg").resolve())
            annotation = ManualAnnotation(
                prediction_id=spec.get("prediction_id", 5000 + index),
                video_id=video.id,
                frame_id=frame_id,
                track_id=spec.get("track_id", index),
                image_path=image_path,
                image_url=f"/static/frames/{video.id}/frame_{frame_id:06d}.jpg",
                original_bbox=json.dumps(spec.get("original_bbox", [100, 200, 300, 500])),
                corrected_bbox=json.dumps(spec.get("corrected_bbox", [100, 200, 300, 500])),
                original_number=spec.get("original_number", "23"),
                corrected_number=spec.get("corrected_number", "23"),
                label_status=spec.get("label_status", "correct"),
            )
            db.add(annotation)

        db.commit()
        return video.id
    finally:
        db.close()


def test_yolo_export_smoke_and_bbox_conversion(tmp_path, monkeypatch):
    video_id = _create_video_with_annotations(
        tmp_path,
        monkeypatch,
        [{"corrected_bbox": [100, 200, 300, 500], "frame_id": 0, "track_id": 1, "label_status": "correct"}],
    )
    client = TestClient(app, raise_server_exceptions=False)
    response = client.post(f"/videos/{video_id}/export-yolo-dataset")
    assert response.status_code == 200, response.text
    payload = response.json()

    export_root = Path(payload["export_path"])
    assert (export_root / "images").exists()
    assert (export_root / "labels").exists()
    assert (export_root / "dataset.yaml").exists()
    assert (export_root / "classes.txt").exists()
    assert (export_root / "metadata.json").exists()

    label_path = export_root / "labels" / f"video_{video_id}_frame_000000.txt"
    line = label_path.read_text(encoding="utf-8").strip()
    parts = line.split()
    assert parts[0] == "0"
    assert parts[1:] == ["0.200000", "0.350000", "0.200000", "0.300000"]
    assert validate_yolo_dataset(export_root) == {"valid": True, "errors": []}


def test_yolo_export_duplicate_frames_unclear_invalid_missing(tmp_path, monkeypatch):
    video_id = _create_video_with_annotations(
        tmp_path,
        monkeypatch,
        [
            {"prediction_id": 1, "frame_id": 0, "track_id": 1, "corrected_bbox": [100, 200, 300, 500], "label_status": "correct"},
            {"prediction_id": 2, "frame_id": 0, "track_id": 2, "corrected_bbox": [400, 300, 600, 700], "label_status": "wrong_bbox"},
            {"prediction_id": 3, "frame_id": 1, "track_id": 3, "corrected_bbox": [100, 200, 300, 500], "label_status": "unclear"},
            {"prediction_id": 4, "frame_id": 2, "track_id": 4, "corrected_bbox": [100, 200, 100, 500], "label_status": "wrong_number"},
            {"prediction_id": 5, "frame_id": 999, "track_id": 5, "corrected_bbox": [100, 200, 300, 500], "label_status": "correct", "missing_image": True},
        ],
    )
    client = TestClient(app, raise_server_exceptions=False)
    response = client.post(f"/videos/{video_id}/export-yolo-dataset")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["image_count"] == 1
    assert payload["label_file_count"] == 1
    assert payload["annotation_count"] == 2
    assert payload["skipped_unclear_count"] == 1
    assert payload["skipped_invalid_bbox_count"] == 1
    assert payload["skipped_missing_image_count"] == 1

    export_root = Path(payload["export_path"])
    label_path = export_root / "labels" / f"video_{video_id}_frame_000000.txt"
    lines = [line for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 2
    metadata = json.loads((export_root / "metadata.json").read_text(encoding="utf-8"))
    assert len(metadata["skipped_annotations"]) == 3
    assert validate_yolo_dataset(export_root) == {"valid": True, "errors": []}


def test_yolo_export_rejects_when_no_valid_annotations(tmp_path, monkeypatch):
    video_id = _create_video_with_annotations(
        tmp_path,
        monkeypatch,
        [
            {"prediction_id": 10, "frame_id": 0, "track_id": 1, "corrected_bbox": [100, 200, 300, 500], "label_status": "unclear"},
            {"prediction_id": 11, "frame_id": 1, "track_id": 2, "corrected_bbox": [100, 200, 100, 500], "label_status": "wrong_bbox"},
        ],
    )
    client = TestClient(app, raise_server_exceptions=False)
    response = client.post(f"/videos/{video_id}/export-yolo-dataset")
    assert response.status_code == 400, response.text
    assert response.json()["detail"] == "No valid annotations available for export."
