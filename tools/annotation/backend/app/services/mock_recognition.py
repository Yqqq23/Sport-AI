from __future__ import annotations

import json
import random
from pathlib import Path

import cv2

from app.models.dataset import JerseyPrediction, VideoAsset, VideoFrame


NUMBERS = ["0", "7", "11", "23", "28"]


def _clamp_bbox(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> list[int]:
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return [x1, y1, x2, y2]


def _player_bbox(track_id: int, frame_id: int, width: int, height: int) -> list[int]:
    lane_width = max(60, width // 10)
    lane_height = max(140, height // 4)
    base_x = int((track_id * width) / 8)
    drift = int(18 * ((frame_id // 5) % 6)) - 36
    x1 = base_x + drift
    y1 = int(height * 0.38) + (track_id % 3) * 20 + int(10 * ((frame_id // 9) % 3))
    return _clamp_bbox(x1, y1, x1 + lane_width, y1 + lane_height, width, height)


def _jersey_bbox(player_bbox: list[int]) -> list[int]:
    x1, y1, x2, y2 = player_bbox
    w = x2 - x1
    h = y2 - y1
    jx1 = x1 + int(w * 0.28)
    jx2 = x2 - int(w * 0.22)
    jy1 = y1 + int(h * 0.18)
    jy2 = y1 + int(h * 0.48)
    return [jx1, jy1, jx2, jy2]


def _predicted_number(track_id: int, frame_index: int) -> str:
    canonical = {1: "23", 2: "11", 3: "7", 4: "0", 5: "28", 6: "23", 7: "11"}
    if track_id == 1 and frame_index == 2:
        return "28"
    return canonical.get(track_id, NUMBERS[(track_id - 1) % len(NUMBERS)])


def generate_mock_predictions(video: VideoAsset, frames: list[VideoFrame]) -> list[JerseyPrediction]:
    rng = random.Random(video.id)
    capture = cv2.VideoCapture(video.path)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
    capture.release()

    predictions: list[JerseyPrediction] = []
    track_pool = [1, 2, 3, 4, 5, 6, 7]
    for frame_index, frame in enumerate(frames):
        active_count = 5 + (frame_index % 3)
        active_tracks = track_pool[:active_count]
        for track_id in active_tracks:
            player_bbox = _player_bbox(track_id, frame.frame_id, width, height)
            jersey_bbox = _jersey_bbox(player_bbox)
            base_conf = 0.55 + 0.35 * rng.random()
            if (frame_index + track_id) % 11 == 0:
                base_conf = 0.31 + 0.08 * rng.random()
            predictions.append(
                JerseyPrediction(
                    video_id=video.id,
                    frame_row_id=frame.id,
                    frame_id=frame.frame_id,
                    track_id=track_id,
                    player_bbox=json.dumps(player_bbox),
                    jersey_bbox=json.dumps(jersey_bbox),
                    predicted_number=_predicted_number(track_id, frame_index),
                    confidence=round(base_conf, 3),
                )
            )
    return predictions
