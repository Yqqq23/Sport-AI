import json
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.core.config import settings
from app.db.session import SessionLocal
from app.main import app
from app.models.dataset import ManualAnnotation, VideoAsset, VideoFrame
from app.services.storage import ensure_data_dirs


def _write_frame(path: Path, width: int = 640, height: int = 480) -> None:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.imwrite(str(path), image)


def test_real_recognition_endpoint_returns_predictions_or_clear_model_error(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    monkeypatch.setattr(settings, "data_dir", data_dir)
    ensure_data_dirs()

    db = SessionLocal()
    try:
        video_path = data_dir / "videos" / f"{uuid4().hex}.mp4"
        video_path.write_bytes(b"not-a-real-video")
        video = VideoAsset(filename=video_path.name, path=str(video_path.resolve()), status="frames_extracted")
        db.add(video)
        db.flush()

        frame_dir = data_dir / "frames" / str(video.id)
        frame_dir.mkdir(parents=True, exist_ok=True)
        frame_path = frame_dir / "frame_000000.jpg"
        _write_frame(frame_path)
        db.add(
            VideoFrame(
                video_id=video.id,
                frame_id=0,
                frame_path=str(frame_path.resolve()),
                image_width=640,
                image_height=480,
            )
        )
        db.commit()
        video_id = video.id
    finally:
        db.close()

    client = TestClient(app, raise_server_exceptions=False)
    response = client.post(f"/videos/{video_id}/run-jersey-recognition")
    assert response.status_code in {200, 400}, response.text
    if response.status_code == 400:
        return

    payload = response.json()
    assert payload["prediction_count"] >= 0

    summary_response = client.get(f"/videos/{video_id}/recognition-summary")
    assert summary_response.status_code == 200, summary_response.text


def test_annotation_save_and_internal_export_still_work(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    monkeypatch.setattr(settings, "data_dir", data_dir)
    ensure_data_dirs()

    db = SessionLocal()
    try:
        video_path = data_dir / "videos" / f"{uuid4().hex}.mp4"
        video_path.write_bytes(b"video")
        video = VideoAsset(filename=video_path.name, path=str(video_path.resolve()), status="recognition_ready")
        db.add(video)
        db.flush()

        frame_dir = data_dir / "frames" / str(video.id)
        frame_dir.mkdir(parents=True, exist_ok=True)
        frame_path = frame_dir / "frame_000000.jpg"
        _write_frame(frame_path)
        frame = VideoFrame(video_id=video.id, frame_id=0, frame_path=str(frame_path.resolve()), image_width=640, image_height=480)
        db.add(frame)
        db.flush()

        db.add(
            ManualAnnotation(
                prediction_id=999,
                video_id=video.id,
                frame_id=0,
                track_id=1,
                image_path=str(frame_path.resolve()),
                image_url=f"/static/frames/{video.id}/frame_000000.jpg",
                original_bbox=json.dumps([100, 100, 200, 200]),
                corrected_bbox=json.dumps([100, 100, 200, 200]),
                original_number="23",
                corrected_number="23",
                label_status="correct",
            )
        )
        db.commit()
        video_id = video.id
    finally:
        db.close()

    client = TestClient(app, raise_server_exceptions=False)
    export_response = client.post(f"/videos/{video_id}/export-internal-dataset")
    assert export_response.status_code == 200, export_response.text
    export_id = export_response.json()["export_id"]
    download_response = client.get(f"/exports/{export_id}/download")
    assert download_response.status_code == 200, download_response.text
