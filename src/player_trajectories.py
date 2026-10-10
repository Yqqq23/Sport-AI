"""Independent per-motion-ID top-down clips, using the pipeline's exact canvas."""

import json
from pathlib import Path

import cv2
import numpy as np

from render import FrameRenderData, TrailHistory, render_topdown_frame


class PlayerTrajectoryExporter:
    def __init__(self, directory, template, config, fps, fourcc, open_writer):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.template = template
        self.config = config
        self.fps = fps
        self.fourcc = fourcc
        self.open_writer = open_writer
        self.active = {}
        self.entries = []

    def update(self, data, live_keys):
        """Clips start at first valid position and end when their motion ID retires."""
        for key in list(self.active):
            if key not in live_keys:
                self._finish(key)
        visible = {}
        for player in data.players:
            point = player.court_xy
            if point is None or not np.isfinite(point).all():
                continue
            if not (0 <= point[0] < self.config.width and 0 <= point[1] < self.config.height):
                continue
            key = str(player.render_key)
            visible[key] = player
            if key not in self.active:
                # Track IDs, unlike roster slots and OCR labels, never recycle.
                stem = f"motion_{player.track_id:06d}"
                writer = self.open_writer(self.directory / f"{stem}.mp4", self.fourcc,
                                          self.fps, (self.config.width, self.config.height))
                entry = dict(key=key, motion_id=player.track_id,
                             slot=player.identity.slot, jersey_numbers=[],
                             video=f"{stem}.mp4", trajectory_image=f"{stem}.png",
                             start_frame=data.frame_idx, end_frame=data.frame_idx,
                             start_sec=data.timestamp_sec, frames_written=0,
                             observed_frames=0, predicted_frames=0)
                self.entries.append(entry)
                self.active[key] = dict(writer=writer, entry=entry,
                                        trails=TrailHistory(self.config.trail_length, alpha=1.0),
                                        overview=self.template.copy(), previous=None)

        for key, state in self.active.items():
            player = visible.get(key)
            if player is None:
                state["trails"].discard(key)
                state["previous"] = None
            else:
                entry = state["entry"]
                number = player.identity.number
                if number is not None and number not in entry["jersey_numbers"]:
                    entry["jersey_numbers"].append(number)
                entry["predicted_frames" if player.predicted else "observed_frames"] += 1
                point = tuple(int(round(v)) for v in player.court_xy)
                # Summary shows observed segments only. Predictions stay visible
                # in the video with '~', but are not asserted as measured paths.
                if player.predicted:
                    state["previous"] = None
                else:
                    if state["previous"] is not None:
                        cv2.line(state["overview"], state["previous"], point,
                                 player.color(), 2, cv2.LINE_AA)
                    cv2.circle(state["overview"], point, 2, player.color(), -1)
                    state["previous"] = point
            frame = render_topdown_frame(
                self.template, FrameRenderData(data.frame_idx, data.timestamp_sec,
                                               players=[] if player is None else [player]),
                state["trails"], self.config,
            )
            cv2.putText(frame, f"motion:{state['entry']['motion_id']}  id:{state['entry']['slot']}",
                        (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            state["writer"].write(frame)
            state["entry"]["end_frame"] = data.frame_idx
            state["entry"]["frames_written"] += 1

    def _finish(self, key):
        state = self.active.pop(key)
        state["writer"].release()
        entry = state["entry"]
        cv2.putText(state["overview"], f"motion:{entry['motion_id']}  id:{entry['slot']}",
                    (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        ok, image = cv2.imencode(".png", state["overview"])
        if not ok:
            raise RuntimeError("Cannot encode player trajectory image")
        (self.directory / entry["trajectory_image"]).write_bytes(image.tobytes())

    def close(self):
        try:
            for key in list(self.active):
                self._finish(key)
        finally:
            for state in self.active.values():
                state["writer"].release()
        (self.directory / "index.json").write_text(json.dumps({
            "fps": self.fps, "canvas_size": [self.config.width, self.config.height],
            "court_size_m": [self.config.court_width_m, self.config.court_height_m],
            "note": "One clip per motion ID, starting at start_frame; jersey numbers are labels, not identity.",
            "players": self.entries,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
