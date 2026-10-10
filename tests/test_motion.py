"""Motion continuity regressions, including the real render-data assembly path."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from calib import CourtProjector
from identity import PlayerIdentityRegistry
from motion import CourtMotionTracker, MotionConfig
from pipeline import _build_render_data
from render import TrailHistory


def test_crossing_tracks_with_reordered_detections():
    tracker = CourtMotionTracker(MotionConfig(), max_players=2)
    for frame in range(150):
        t = frame / 30
        points = [[3 + 1.2 * t, 7], [7.1 - 1.2 * t, 7]]
        order = [0, 1] if frame % 2 == 0 else [1, 0]
        ids = tracker.update([points[i] for i in order], t)
        assert ids == [i + 1 for i in order]


def test_short_occlusion_prediction_and_reacquisition():
    tracker = CourtMotionTracker(MotionConfig(), max_players=1)
    for frame in range(30):
        assert tracker.update([[3 + frame / 30, 5]], frame / 30) == [1]
    previous_x = tracker.tracks[1].xy[0]
    for frame in range(30, 36):
        tracker.update([], frame / 30)
        predicted = tracker.predictions()
        assert len(predicted) == 1
        assert predicted[0].xy[0] > previous_x
        previous_x = predicted[0].xy[0]
    assert tracker.update([[4.2, 5]], 36 / 30) == [1]
    assert tracker.predictions() == []
    tracker.update([], 1.6)
    assert tracker.predictions() == []
    assert 1 in tracker.tracks  # retained for association, no extrapolated dot


def test_cross_court_outlier_rejected_and_expiry_is_new_identity():
    tracker = CourtMotionTracker(MotionConfig(), max_players=1)
    for frame in range(10):
        tracker.update([[2, 2]], frame / 30)
    assert tracker.update([[14, 13]], 10 / 30) == [0]
    assert np.linalg.norm(np.array(tracker.tracks[1].xy) - [2, 2]) < 0.01
    assert tracker.update([[2.1, 2]], 11 / 30) == [1]
    assert tracker.update([[14, 13]], 2.0) == [2]
    assert tracker.expired_ids == [1]
    assert tracker.tracks[2].slot == 1  # display slot reused, motion ID never reused


def test_invalid_points_do_not_consume_roster():
    tracker = CourtMotionTracker(MotionConfig(), max_players=2)
    assert tracker.update([[np.nan, 2], [200, 3], [5, 5]], 0) == [0, 0, 1]
    assert len(tracker.tracks) == 1
    with pytest.raises(ValueError, match="strictly increasing"):
        tracker.update([], 0)


def test_kalman_reduces_stationary_measurement_noise():
    tracker = CourtMotionTracker(MotionConfig(), max_players=1)
    rng = np.random.default_rng(42)
    raw, filtered = [], []
    for frame in range(180):
        observation = [6, 7] + rng.normal(0, 0.15, 2)
        assert tracker.update([observation], frame / 30) == [1]
        if frame > 30:
            raw.append(observation - [6, 7])
            filtered.append(np.array(tracker.tracks[1].xy) - [6, 7])
    assert np.mean(np.square(filtered)) < np.mean(np.square(raw)) * 0.6


def test_number_confusion_does_not_merge_rendered_paths():
    projector = CourtProjector.from_json(ROOT / "configs/court_calibration_refined.json")
    tracker = CourtMotionTracker(MotionConfig(), max_players=2)
    tracker.update([[3, 3], [12, 11]], 0)
    players = [SimpleNamespace(bbox_xyxy=[0, 0, 10, 10], ground_xy=(950, 850), confidence=0.9)
               for _ in range(2)]
    tracked = SimpleNamespace(players=players, ball=None, rims=[], referees=[])
    registry = PlayerIdentityRegistry(max_players=2)
    numbers = {1: "7", 2: "1"}
    stabilizer = SimpleNamespace(get_stable_number=lambda k: numbers[k], is_locked=lambda k: True)
    trails = TrailHistory()
    snapshots = []
    for frame, number in enumerate(["7", "1", "7"]):
        numbers[1] = number
        data = _build_render_data(
            tracked, stabilizer, registry, projector, frame, frame / 30,
            {id(p): i + 1 for i, p in enumerate(players)}, motion_tracker=tracker,
        )
        assert [p.render_key for p in data.players] == ["motion:1", "motion:2"]
        assert [p.identity.slot for p in data.players] == [1, 2]
        snapshots.append([p.court_xy for p in data.players])
        for player in data.players:
            trails.update(player.render_key, player.court_xy)
    np.testing.assert_allclose(snapshots[0], snapshots[1])
    assert len(trails.get("motion:1")) == len(trails.get("motion:2")) == 3
    assert trails.get("motion:1")[0] != trails.get("motion:2")[0]


def test_pipeline_writes_both_videos_and_uses_motion_ids(tmp_path, monkeypatch):
    import cv2
    import pipeline
    from config import load_config

    source = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1920, 1080))
    assert writer.isOpened()
    for _ in range(12):
        writer.write(np.zeros((1080, 1920, 3), dtype=np.uint8))
    writer.release()
    config = load_config(ROOT / "configs/default.yaml")
    config.detector.weights = tmp_path / "fake.pt"
    config.detector.weights.touch()
    config.output.output_dir = tmp_path / "output"
    config.jersey.enabled = False

    class FakeTracker:
        backend_name = "fake"
        reid_active = False
        class_detection_counts = {0: 8, 1: 1}

        def __init__(self, **kwargs):
            pass

        def update(self, frame, frame_idx, timestamp_sec):
            players = [] if 5 <= frame_idx <= 8 else [SimpleNamespace(
                track_id=99, bbox_xyxy=(900, 700, 1000, 850), ground_xy=(950, 850),
                confidence=0.9,
            )]
            return SimpleNamespace(players=players, jerseys=[], ball=None, rims=[], referees=[])

    shot_inputs = []

    class FakeShots:
        def __init__(self, *args, **kwargs):
            pass

        def update(self, **kwargs):
            shot_inputs.append(kwargs["players"])
            return None

    monkeypatch.setattr(pipeline, "Tracker", FakeTracker)
    monkeypatch.setattr(pipeline, "ShotAnalyzer", FakeShots)
    monkeypatch.setattr(pipeline, "make_ocr_backend", lambda *a, **kw: SimpleNamespace(is_available=lambda: False))
    # Inspect the actual ID passed into shot analysis without relying on its
    # internal player-record representation.
    monkeypatch.setattr(pipeline, "make_player_box", lambda track_id, *a: track_id)
    outputs = pipeline.run_pipeline(source, config)
    import json
    index = json.loads((outputs["player_trajectories"] / "index.json").read_text())
    assert len(index["players"]) == 1
    entry = index["players"][0]
    assert entry["motion_id"] == 1
    assert entry["predicted_frames"] == 4
    assert entry["frames_written"] == 12
    outputs["single_player"] = outputs["player_trajectories"] / entry["video"]
    assert shot_inputs == [[1]] * 5 + [[]] * 4 + [[1]] * 3
    for name, size in (("annotated", (1920, 1080)), ("topdown", (1020, 960)),
                       ("single_player", (1020, 960))):
        reader = cv2.VideoCapture(str(outputs[name]))
        assert reader.isOpened()
        count = 0
        while True:
            ok, frame = reader.read()
            if not ok:
                break
            assert (frame.shape[1], frame.shape[0]) == size
            count += 1
        reader.release()
        assert count == 12
