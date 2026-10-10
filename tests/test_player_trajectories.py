import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from identity import Identity
from player_trajectories import PlayerTrajectoryExporter
from render import CourtRenderConfig, FrameRenderData, PlayerRender, draw_half_court_template


def test_separate_players_numbers_gaps_and_retirement(tmp_path):
    config = CourtRenderConfig(width=360, height=344, margin=20)
    template = draw_half_court_template(config)
    writers = {}

    class Writer:
        def __init__(self):
            self.frames = []
            self.closed = False

        def write(self, frame):
            assert not self.closed
            self.frames.append(frame.copy())

        def release(self):
            self.closed = True

    def open_writer(path, fourcc, fps, size):
        assert size == (360, 344)
        assert path.name not in writers
        writers[path.name] = Writer()
        return writers[path.name]

    exporter = PlayerTrajectoryExporter(tmp_path, template, config, 30, 0, open_writer)

    def player(track_id, slot, point, predicted=False):
        return PlayerRender(track_id, [], court_xy=point, predicted=predicted,
                            identity=Identity(f"motion:{track_id}", "7", slot))

    one = player(1, 1, (70, 110))
    two = player(2, 2, (270, 230))  # same misread jersey, distinct output
    exporter.update(FrameRenderData(10, 10 / 30, [one, two]), {"motion:1", "motion:2"})
    a = writers["motion_000001.mp4"].frames[0]
    b = writers["motion_000002.mp4"].frames[0]
    np.testing.assert_array_equal(a[220:240, 260:280], template[220:240, 260:280])
    np.testing.assert_array_equal(b[100:120, 60:80], template[100:120, 60:80])
    assert not np.array_equal(a[100:120, 60:80], template[100:120, 60:80])
    assert not np.array_equal(b[220:240, 260:280], template[220:240, 260:280])
    one.identity.number = "1"
    one.predicted = True
    exporter.update(FrameRenderData(11, 11 / 30, [one, two]), {"motion:1", "motion:2"})
    exporter.update(FrameRenderData(12, 12 / 30, [two]), {"motion:1", "motion:2"})
    three = player(3, 1, (90, 140))  # reuse slot, never overwrite old motion clip
    exporter.update(FrameRenderData(13, 13 / 30, [two, three]), {"motion:2", "motion:3"})
    assert writers["motion_000001.mp4"].closed
    exporter.close()
    assert all(w.closed for w in writers.values())
    index = json.loads((tmp_path / "index.json").read_text())
    entries = index["players"]
    assert [e["motion_id"] for e in entries] == [1, 2, 3]
    assert entries[0]["jersey_numbers"] == ["7", "1"]
    assert entries[0]["start_frame"] == 10
    assert entries[0]["end_frame"] == 12
    assert entries[0]["frames_written"] == 3
    assert entries[0]["predicted_frames"] == 1
    assert entries[2]["start_frame"] == 13
    assert all((tmp_path / e["trajectory_image"]).is_file() for e in entries)
