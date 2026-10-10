import json
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.core.config import settings
from app.db.session import SessionLocal
from app.main import app
from app.models.dataset import DatasetVersion, ManualAnnotation, VideoAsset, VideoFrame
from app.services.dataset_versioning import compute_source_data_fingerprint, validate_dataset_version
from app.services.storage import ensure_data_dirs


def _write_frame(path: Path, width: int = 1000, height: int = 1000) -> None:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = (90, 120, 150)
    cv2.imwrite(str(path), image)


def _create_video(db, data_dir: Path, *, filename: str, frame_ids: list[int], annotations: list[dict]):
    video = VideoAsset(filename=filename, path=str((data_dir / "videos" / filename).resolve()), status="frames_extracted")
    db.add(video)
    db.flush()

    frames_dir = data_dir / "frames" / str(video.id)
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_paths: dict[int, Path] = {}
    for frame_id in frame_ids:
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

    for index, spec in enumerate(annotations, start=1):
        frame_id = spec["frame_id"]
        image_path = str(frame_paths[frame_id].resolve())
        if spec.get("missing_image"):
            image_path = str((frames_dir / f"missing_{index}.jpg").resolve())
        db.add(
            ManualAnnotation(
                prediction_id=spec.get("prediction_id", video.id * 1000 + index),
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
                deleted_at=spec.get("deleted_at"),
            )
        )
    db.flush()
    return video.id


def _seed_dataset(tmp_path, monkeypatch, video_payloads):
    data_dir = tmp_path / "data"
    monkeypatch.setattr(settings, "data_dir", data_dir)
    ensure_data_dirs()
    db = SessionLocal()
    try:
        video_ids = []
        for index, payload in enumerate(video_payloads, start=1):
            video_ids.append(
                _create_video(
                    db,
                    data_dir,
                    filename=f"video_{uuid4().hex}_{index}.mp4",
                    frame_ids=payload["frame_ids"],
                    annotations=payload["annotations"],
                )
            )
        db.commit()
        return video_ids
    finally:
        db.close()


def test_dataset_versioning_multi_video_merge_and_video_leakage(tmp_path, monkeypatch):
    video_ids = _seed_dataset(
        tmp_path,
        monkeypatch,
        [
            {"frame_ids": [0, 30], "annotations": [{"frame_id": 0, "track_id": 1}, {"frame_id": 30, "track_id": 1}]},
            {"frame_ids": [0, 30], "annotations": [{"frame_id": 0, "track_id": 2}, {"frame_id": 30, "track_id": 2}]},
            {"frame_ids": [0, 30], "annotations": [{"frame_id": 0, "track_id": 3}, {"frame_id": 30, "track_id": 3}]},
        ],
    )
    client = TestClient(app, raise_server_exceptions=False)
    version_name = f"merge_{uuid4().hex}"
    response = client.post(
        "/dataset-versions",
        json={
            "version_name": version_name,
            "video_ids": video_ids,
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "random_seed": 42,
            "split_strategy": "video",
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["exported_annotation_count"] == 6
    assert payload["exported_image_count"] == 6
    validation = payload["validation_result_json"]
    assert validation["valid"] is True

    split_report = client.get(f"/dataset-versions/{payload['dataset_version_id']}/split-report").json()["split_report"]
    source_video_distribution = split_report["source_video_distribution"]
    assigned_splits = {video_id: split_name for split_name, ids in source_video_distribution.items() for video_id in ids}
    assert len(assigned_splits) == 3
    assert len(set(assigned_splits.values())) >= 1


def test_dataset_versioning_deterministic_and_seed_sensitive(tmp_path, monkeypatch):
    video_ids = _seed_dataset(
        tmp_path,
        monkeypatch,
        [
            {"frame_ids": [0], "annotations": [{"frame_id": 0, "track_id": 1}]},
            {"frame_ids": [0], "annotations": [{"frame_id": 0, "track_id": 2}]},
            {"frame_ids": [0], "annotations": [{"frame_id": 0, "track_id": 3}]},
        ],
    )
    client = TestClient(app, raise_server_exceptions=False)

    def build(name, seed):
        response = client.post(
            "/dataset-versions",
            json={
                "version_name": name,
                "video_ids": video_ids,
                "train_ratio": 0.34,
                "val_ratio": 0.33,
                "test_ratio": 0.33,
                "random_seed": seed,
                "split_strategy": "video",
            },
        )
        assert response.status_code == 200, response.text
        split_report = client.get(f"/dataset-versions/{response.json()['dataset_version_id']}/split-report").json()["split_report"]
        return split_report["source_video_distribution"]

    same_a = build(f"det_a_{uuid4().hex}", 42)
    same_b = build(f"det_b_{uuid4().hex}", 42)
    different = build(f"det_c_{uuid4().hex}", 7)
    assert same_a == same_b
    assert same_a != different


def test_dataset_versioning_track_and_temporal_block_integrity(tmp_path, monkeypatch):
    video_ids = _seed_dataset(
        tmp_path,
        monkeypatch,
        [
            {
                "frame_ids": [0, 1, 2, 3],
                "annotations": [
                    {"frame_id": 0, "track_id": 1},
                    {"frame_id": 0, "track_id": 2},
                    {"frame_id": 1, "track_id": 1},
                    {"frame_id": 2, "track_id": 2},
                    {"frame_id": 3, "track_id": 3},
                ],
            }
        ],
    )
    client = TestClient(app, raise_server_exceptions=False)
    track_response = client.post(
        "/dataset-versions",
        json={
            "version_name": f"track_{uuid4().hex}",
            "video_ids": video_ids,
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "random_seed": 42,
            "split_strategy": "track",
        },
    )
    assert track_response.status_code == 200, track_response.text
    track_payload = track_response.json()
    assert track_payload["validation_result_json"]["valid"] is True
    track_report = client.get(f"/dataset-versions/{track_payload['dataset_version_id']}/split-report").json()["split_report"]
    assert track_report["frame_conflict_resolutions"]

    temporal_response = client.post(
        "/dataset-versions",
        json={
            "version_name": f"temp_{uuid4().hex}",
            "video_ids": video_ids,
            "train_ratio": 0.5,
            "val_ratio": 0.25,
            "test_ratio": 0.25,
            "random_seed": 42,
            "split_strategy": "temporal_block",
            "temporal_block_size": 2,
        },
    )
    assert temporal_response.status_code == 200, temporal_response.text
    temporal_payload = temporal_response.json()
    assert temporal_payload["validation_result_json"]["valid"] is True


def test_dataset_versioning_small_dataset_strict_and_rebuild_fingerprint(tmp_path, monkeypatch):
    video_ids = _seed_dataset(
        tmp_path,
        monkeypatch,
        [{"frame_ids": [0], "annotations": [{"frame_id": 0, "track_id": 1}]}],
    )
    client = TestClient(app, raise_server_exceptions=False)

    strict_response = client.post(
        "/dataset-versions",
        json={
            "version_name": f"strict_{uuid4().hex}",
            "video_ids": video_ids,
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "random_seed": 42,
            "split_strategy": "video",
            "strict_split": True,
        },
    )
    assert strict_response.status_code == 400, strict_response.text

    relaxed_response = client.post(
        "/dataset-versions",
        json={
            "version_name": f"relaxed_{uuid4().hex}",
            "video_ids": video_ids,
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "random_seed": 42,
            "split_strategy": "video",
            "strict_split": False,
        },
    )
    assert relaxed_response.status_code == 200, relaxed_response.text
    payload = relaxed_response.json()
    assert payload["status"] == "completed"
    split_report = client.get(f"/dataset-versions/{payload['dataset_version_id']}/split-report").json()["split_report"]
    assert split_report["warnings"]

    rebuild_ok = client.post(f"/dataset-versions/{payload['dataset_version_id']}/rebuild")
    assert rebuild_ok.status_code == 200, rebuild_ok.text

    db = SessionLocal()
    try:
        annotation = db.query(ManualAnnotation).filter(ManualAnnotation.video_id == video_ids[0]).first()
        annotation.corrected_number = "77"
        db.add(annotation)
        db.commit()
    finally:
        db.close()

    rebuild_fail = client.post(f"/dataset-versions/{payload['dataset_version_id']}/rebuild")
    assert rebuild_fail.status_code == 409, rebuild_fail.text


def test_dataset_versioning_validation_failure_detection(tmp_path, monkeypatch):
    video_ids = _seed_dataset(
        tmp_path,
        monkeypatch,
        [{"frame_ids": [0], "annotations": [{"frame_id": 0, "track_id": 1}]}],
    )
    client = TestClient(app, raise_server_exceptions=False)
    response = client.post(
        "/dataset-versions",
        json={
            "version_name": f"validate_{uuid4().hex}",
            "video_ids": video_ids,
            "train_ratio": 1.0,
            "val_ratio": 0.0,
            "test_ratio": 0.0,
            "random_seed": 42,
            "split_strategy": "video",
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    export_root = Path(payload["export_path"])
    label_files = list((export_root / "labels" / "train").glob("*.txt"))
    assert label_files
    label_files[0].write_text("0 2 2 2 2\n", encoding="utf-8")
    validation = validate_dataset_version(export_root, "video")
    assert validation["valid"] is False


def test_source_data_fingerprint_changes_with_annotation_update(tmp_path, monkeypatch):
    video_ids = _seed_dataset(
        tmp_path,
        monkeypatch,
        [{"frame_ids": [0], "annotations": [{"frame_id": 0, "track_id": 1, "corrected_number": "23"}]}],
    )
    db = SessionLocal()
    try:
        annotations = db.query(ManualAnnotation).filter(ManualAnnotation.video_id == video_ids[0]).all()
        fingerprint_a = compute_source_data_fingerprint(annotations)
        annotations[0].corrected_number = "99"
        db.add(annotations[0])
        db.commit()
        refreshed = db.query(ManualAnnotation).filter(ManualAnnotation.video_id == video_ids[0]).all()
        fingerprint_b = compute_source_data_fingerprint(refreshed)
    finally:
        db.close()
    assert fingerprint_a != fingerprint_b
