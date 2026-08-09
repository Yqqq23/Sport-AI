"""API-level tests for 5-class annotation.

The behaviour change that matters most here is structural: the tool used to key
annotation identity on (prediction_id, video_id) and synthesise ONE id per
frame, which capped it at one hand-drawn box per image. A 5-class dataset needs
several boxes of different classes on the same frame, so hand-drawn boxes now
get one id each.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.db.session import SessionLocal
from app.main import app
from app.models.dataset import ManualAnnotation, VideoAsset, VideoFrame
from app.services.basketball_classes import (
    CLASS_BALL,
    CLASS_JERSEY,
    CLASS_NAMES_IN_INDEX_ORDER,
    CLASS_PLAYER,
    CLASS_REFEREE,
    CLASS_RIM,
)
from app.services.storage import ensure_data_dirs


def _make_video(tmp_path, monkeypatch, frame_ids=(0,)):
    data_dir = tmp_path / "data"
    monkeypatch.setattr(settings, "data_dir", data_dir)
    ensure_data_dirs()

    db = SessionLocal()
    try:
        video = VideoAsset(
            filename="test.mp4",
            path=str((data_dir / "videos" / "test.mp4").resolve()),
            status="frames_extracted",
        )
        db.add(video)
        db.flush()

        frames_dir = data_dir / "frames" / str(video.id)
        frames_dir.mkdir(parents=True, exist_ok=True)
        image = np.zeros((1000, 1000, 3), dtype=np.uint8)
        for frame_id in frame_ids:
            frame_path = frames_dir / f"frame_{frame_id:06d}.jpg"
            cv2.imwrite(str(frame_path), image)
            db.add(
                VideoFrame(
                    video_id=video.id,
                    frame_id=frame_id,
                    frame_path=str(frame_path.resolve()),
                    image_width=1000,
                    image_height=1000,
                )
            )
        db.commit()
        return video.id
    finally:
        db.close()


def _box_payload(video_id, class_id, bbox, *, frame_id=0, number=None, track_id=0):
    return {
        # Negative prediction_id = hand-drawn box; the server allocates a unique id.
        "prediction_id": -(frame_id + 1),
        "video_id": video_id,
        "frame_id": frame_id,
        "track_id": track_id,
        "class_id": class_id,
        "image_path": "",
        "image_url": f"/static/frames/{video_id}/frame_{frame_id:06d}.jpg",
        "original_bbox": bbox,
        "corrected_bbox": bbox,
        "original_number": "",
        "corrected_number": number,
        "label_status": "correct",
        "image_width": 1000,
        "image_height": 1000,
    }


def test_basketball_classes_endpoint_serves_the_scheme():
    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/basketball-classes")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["class_count"] == 5
    assert [item["name"] for item in payload["classes"]] == CLASS_NAMES_IN_INDEX_ORDER
    assert payload["number_bearing_class_name"] == "jersey"
    assert Path(payload["source_of_truth"]).name == "basketball.yaml"


def test_many_boxes_of_different_classes_on_one_frame(tmp_path, monkeypatch):
    """The core structural change: one frame, five boxes, five rows."""
    video_id = _make_video(tmp_path, monkeypatch)
    client = TestClient(app, raise_server_exceptions=False)

    specs = [
        (CLASS_PLAYER, [100, 200, 300, 700], None, 1),
        (CLASS_BALL, [500, 100, 540, 140], None, 0),
        (CLASS_RIM, [600, 80, 700, 120], None, 0),
        (CLASS_JERSEY, [150, 280, 260, 380], "7", 1),
        (CLASS_REFEREE, [800, 200, 900, 600], None, 0),
    ]
    annotation_ids = []
    for class_id, bbox, number, track_id in specs:
        response = client.post(
            "/annotations",
            json=_box_payload(video_id, class_id, bbox, number=number, track_id=track_id),
        )
        assert response.status_code == 200, response.text
        annotation_ids.append(response.json()["annotation_id"])

    # Five distinct rows, not one row overwritten five times.
    assert len(set(annotation_ids)) == 5

    saved = client.get(f"/videos/{video_id}/annotations").json()
    assert len(saved) == 5
    assert sorted(row["class_id"] for row in saved) == [0, 1, 2, 3, 4]
    assert sorted(row["class_name"] for row in saved) == sorted(CLASS_NAMES_IN_INDEX_ORDER)

    # Each hand-drawn box holds a unique prediction_id, which dataset_versioning
    # relies on to detect split leakage.
    assert len({row["prediction_id"] for row in saved}) == 5

    # The review item for that frame exposes every box.
    items = client.get(f"/videos/{video_id}/review-items").json()["items"]
    assert len(items) == 1
    assert len(items[0]["saved_boxes"]) == 5


def test_number_is_rejected_on_a_non_jersey_class(tmp_path, monkeypatch):
    """Silently dropping it would look like the number had been recorded."""
    video_id = _make_video(tmp_path, monkeypatch)
    client = TestClient(app, raise_server_exceptions=False)
    response = client.post(
        "/annotations",
        json=_box_payload(video_id, CLASS_RIM, [600, 80, 700, 120], number="23"),
    )
    assert response.status_code == 422, response.text
    assert "jersey" in response.text


def test_jersey_number_is_normalized_on_save(tmp_path, monkeypatch):
    """'01' is stored as '1', matching what the inference pipeline votes with."""
    video_id = _make_video(tmp_path, monkeypatch)
    client = TestClient(app, raise_server_exceptions=False)
    response = client.post(
        "/annotations",
        json=_box_payload(video_id, CLASS_JERSEY, [100, 200, 300, 500], number="01"),
    )
    assert response.status_code == 200, response.text
    saved = client.get(f"/videos/{video_id}/annotations").json()
    assert saved[0]["corrected_number"] == "1"


@pytest.mark.parametrize("bad_number", ["00", "123", "xx"])
def test_illegal_jersey_numbers_are_rejected(tmp_path, monkeypatch, bad_number):
    video_id = _make_video(tmp_path, monkeypatch)
    client = TestClient(app, raise_server_exceptions=False)
    response = client.post(
        "/annotations",
        json=_box_payload(video_id, CLASS_JERSEY, [100, 200, 300, 500], number=bad_number),
    )
    assert response.status_code == 422, response.text


def test_jersey_number_may_be_blank_when_unclear(tmp_path, monkeypatch):
    """An unreadable number is a legitimate outcome; the box may still be useful."""
    video_id = _make_video(tmp_path, monkeypatch)
    client = TestClient(app, raise_server_exceptions=False)
    payload = _box_payload(video_id, CLASS_JERSEY, [100, 200, 300, 500], number=None)
    payload["label_status"] = "unclear"
    response = client.post("/annotations", json=payload)
    assert response.status_code == 200, response.text


def test_out_of_range_class_id_is_rejected(tmp_path, monkeypatch):
    video_id = _make_video(tmp_path, monkeypatch)
    client = TestClient(app, raise_server_exceptions=False)
    response = client.post(
        "/annotations", json=_box_payload(video_id, 9, [100, 200, 300, 500])
    )
    assert response.status_code == 422, response.text


def test_a_box_can_be_reclassified_without_redrawing_it(tmp_path, monkeypatch):
    """Mislabelling a referee as a player is a common correction."""
    video_id = _make_video(tmp_path, monkeypatch)
    client = TestClient(app, raise_server_exceptions=False)
    created = client.post(
        "/annotations", json=_box_payload(video_id, CLASS_PLAYER, [100, 200, 300, 700], track_id=2)
    )
    annotation_id = created.json()["annotation_id"]

    response = client.put(
        f"/annotations/{annotation_id}",
        json={
            "corrected_bbox": [100, 200, 300, 700],
            "class_id": CLASS_REFEREE,
            "corrected_number": None,
            "label_status": "correct",
            "image_width": 1000,
            "image_height": 1000,
        },
    )
    assert response.status_code == 200, response.text
    saved = client.get(f"/videos/{video_id}/annotations").json()
    assert saved[0]["class_id"] == CLASS_REFEREE
    assert saved[0]["class_name"] == "referee"
