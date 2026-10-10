import json
from pathlib import Path

from app.services.fixed7_adapter import Fixed7AdapterError, estimate_jersey_bbox, parse_fixed7_tracking_results, run_fixed7_on_video


def test_parse_fixed7_tracking_results_sample(tmp_path):
    sample = [
        {
            "frame_idx": 12,
            "players": [
                {
                    "track_id": 5,
                    "bbox_xyxy": [100.0, 200.0, 300.0, 500.0],
                    "number_bbox": [150.0, 240.0, 230.0, 320.0],
                    "number_text": "23",
                    "number_conf": 0.91,
                    "player_conf": 0.77,
                },
                {
                    "track_id": 6,
                    "bbox_xyxy": [400.0, 220.0, 520.0, 520.0],
                    "number_text": "",
                    "number_conf": None,
                    "player_conf": 0.63,
                },
            ],
        }
    ]
    results_path = tmp_path / "tracking_pose_results.json"
    results_path.write_text(json.dumps(sample), encoding="utf-8")
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()

    records = parse_fixed7_tracking_results(results_path, frame_dir)
    assert len(records) == 2
    assert records[0]["frame_id"] == 12
    assert records[0]["track_id"] == 5
    assert records[0]["player_bbox"] == [100, 200, 300, 500]
    assert records[0]["jersey_bbox"] == [150, 240, 230, 320]
    assert records[0]["predicted_number"] == "23"
    assert records[0]["confidence"] == 0.91
    assert records[1]["predicted_number"] == ""
    assert records[1]["confidence"] == 0.63


def test_estimate_jersey_bbox_from_player_bbox():
    bbox = estimate_jersey_bbox([100, 200, 300, 500])
    assert bbox[0] >= 100
    assert bbox[1] >= 200
    assert bbox[2] <= 300
    assert bbox[3] <= 500
    assert bbox[0] < bbox[2]
    assert bbox[1] < bbox[3]


def test_missing_weights_returns_clear_error(tmp_path, monkeypatch):
    from app.services import fixed7_adapter

    monkeypatch.setattr(fixed7_adapter, "fixed7_root", lambda: tmp_path / "fixed7")
    monkeypatch.setattr(fixed7_adapter, "fixed7_main_script", lambda: (tmp_path / "fixed7" / "script.py"))
    monkeypatch.setattr(fixed7_adapter, "fixed7_default_detector_weights", lambda: (tmp_path / "fixed7" / "models" / "rf_detr_player_number.pt"))
    (tmp_path / "fixed7").mkdir(parents=True, exist_ok=True)
    (tmp_path / "fixed7" / "script.py").write_text("print('ok')", encoding="utf-8")

    try:
        run_fixed7_on_video(str(tmp_path / "video.mp4"), str(tmp_path / "out"))
        assert False, "expected Fixed7AdapterError"
    except Fixed7AdapterError as exc:
        payload = exc.to_dict()
        assert payload["ok"] is False
        assert payload["error"] == "fixed7 detector weights missing"
        assert payload["missing_path"].endswith("rf_detr_player_number.pt")


def test_no_fake_jersey_numbers_generated(tmp_path):
    sample = [
        {
            "frame_idx": 7,
            "players": [
                {
                    "track_id": 1,
                    "bbox_xyxy": [10.0, 20.0, 110.0, 220.0],
                    "player_conf": 0.55,
                }
            ],
        }
    ]
    results_path = tmp_path / "tracking_pose_results.json"
    results_path.write_text(json.dumps(sample), encoding="utf-8")
    records = parse_fixed7_tracking_results(results_path)
    assert len(records) == 1
    assert records[0]["predicted_number"] == ""
