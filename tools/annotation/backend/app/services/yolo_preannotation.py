from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.dataset import PREDICTION_SOURCE_YOLO, JerseyPrediction, VideoAsset, VideoFrame


class YoloPreannotationError(RuntimeError):
    pass


def _load_yolo_detector():
    try:
        from ultralytics import YOLO
        return YOLO("yolov8n.pt")
    except Exception as exc:
        raise YoloPreannotationError(f"Failed to load YOLO model: {exc}") from exc


def _detect_players_yolo(model, frame: np.ndarray) -> tuple[list[list[int]], list[float]]:
    frame_h, frame_w = frame.shape[:2]
    boxes: list[list[int]] = []
    confs: list[float] = []

    try:
        results = model.predict(frame, conf=max(0.1, settings.detector_confidence), iou=settings.detector_iou, verbose=False)
        if results:
            boxes_xyxy = results[0].boxes.xyxy.detach().cpu().numpy()
            cls_ids = results[0].boxes.cls.detach().cpu().numpy()
            scores = results[0].boxes.conf.detach().cpu().numpy()
            for i in range(len(boxes_xyxy)):
                if int(cls_ids[i]) != 0:
                    continue
                x1 = int(max(0, boxes_xyxy[i][0]))
                y1 = int(max(0, boxes_xyxy[i][1]))
                x2 = int(min(frame_w, boxes_xyxy[i][2]))
                y2 = int(min(frame_h, boxes_xyxy[i][3]))
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                    confs.append(float(scores[i]))
    except Exception as exc:
        raise YoloPreannotationError(f"YOLO predict failed: {exc}") from exc

    if not boxes:
        raise YoloPreannotationError("YOLO detected no players.")
    return boxes, confs


def _estimate_jersey_bbox(player_bbox: list[int], image_width: int, image_height: int) -> list[int]:
    x1, y1, x2, y2 = player_bbox
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    jersey_x1 = x1 + int(0.25 * width)
    jersey_x2 = x2 - int(0.25 * width)
    jersey_y1 = y1 + int(0.15 * height)
    jersey_y2 = y1 + int(0.45 * height)
    jersey_x1 = max(0, min(jersey_x1, image_width - 1))
    jersey_x2 = max(jersey_x1 + 1, min(jersey_x2, image_width))
    jersey_y1 = max(0, min(jersey_y1, image_height - 1))
    jersey_y2 = max(jersey_y1 + 1, min(jersey_y2, image_height))
    return [jersey_x1, jersey_y1, jersey_x2, jersey_y2]


class YoloPreannotator:
    def run(self, video_id: int, db: Session) -> tuple[list[JerseyPrediction], str]:
        video = db.query(VideoAsset).filter(VideoAsset.id == video_id).first()
        if video is None:
            raise YoloPreannotationError("Video not found")
        frames = db.query(VideoFrame).filter(VideoFrame.video_id == video_id).order_by(VideoFrame.frame_id.asc()).all()
        if not frames:
            raise YoloPreannotationError("No frames found")

        model = _load_yolo_detector()
        predictions: list[JerseyPrediction] = []
        next_track_id = 1
        stride = max(1, settings.recognition_frame_stride)

        for frame_row in frames[::stride]:
            frame_path = Path(frame_row.frame_path)
            if not frame_path.exists():
                continue
            image_bytes = np.fromfile(str(frame_path), dtype=np.uint8)
            frame = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
            if frame is None:
                continue
            image_height, image_width = frame.shape[:2]

            player_bboxes, player_confs = _detect_players_yolo(model, frame)
            for index, player_bbox in enumerate(player_bboxes):
                jersey_bbox = _estimate_jersey_bbox(player_bbox, image_width, image_height)
                predictions.append(
                    JerseyPrediction(
                        video_id=video_id,
                        frame_row_id=frame_row.id,
                        frame_id=frame_row.frame_id,
                        track_id=next_track_id + index,
                        player_bbox=json.dumps(player_bbox),
                        jersey_bbox=json.dumps(jersey_bbox),
                        predicted_number="",
                        confidence=round(float(player_confs[index]) if index < len(player_confs) else 0.0, 3),
                        source=PREDICTION_SOURCE_YOLO,
                    )
                )
            next_track_id += len(player_bboxes)

        if not predictions:
            raise YoloPreannotationError("No predictions produced.")
        return predictions, "yolov8n"