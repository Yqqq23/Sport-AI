from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.dataset import PREDICTION_SOURCE_REAL, JerseyPrediction, VideoAsset, VideoFrame


FACE_CLUSTER_THRESHOLD = 0.4


class RealRecognitionConfigurationError(RuntimeError):
    pass


@dataclass
class RecognitionRunMetadata:
    message: str
    ocr_available: bool
    warnings: list[str]


def _load_yolo_detector():
    try:
        from ultralytics import YOLO
        return YOLO("yolov8n.pt")
    except Exception as exc:
        raise RealRecognitionConfigurationError(f"Failed to load YOLO model: {exc}") from exc


class NumberRecognizer:
    def __init__(self) -> None:
        self._tesseract_ok = False
        try:
            import pytesseract
            _ = pytesseract.get_tesseract_version()
            self._tesseract_ok = True
        except Exception:
            self._tesseract_ok = False

    def _prepare_number_crop(self, crop: np.ndarray) -> np.ndarray:
        h, w = crop.shape[:2]
        if h <= 2 or w <= 2:
            return crop
        x1 = int(round(w * 0.18))
        x2 = int(round(w * 0.82))
        y1 = int(round(h * 0.12))
        y2 = int(round(h * 0.68))
        torso = crop[max(0, y1):max(y1 + 1, y2), max(0, x1):max(x1 + 1, x2)].copy()
        if torso.size == 0:
            torso = crop
        scale = max(1, int(round(160 / max(1, torso.shape[0]))))
        if scale > 1:
            torso = cv2.resize(torso, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        return torso

    def _generate_player_crop_views(self, crop: np.ndarray) -> list[np.ndarray]:
        if crop is None or crop.size == 0:
            return []
        base = self._prepare_number_crop(crop)
        views = [base]
        gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bw_bgr = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
        views.append(bw_bgr)
        views.append(cv2.bitwise_not(bw_bgr))
        boosted = cv2.convertScaleAbs(base, alpha=1.35, beta=10)
        views.append(boosted)
        return views

    def _recognize_by_tesseract(self, crop: np.ndarray) -> tuple[str | None, float]:
        if not self._tesseract_ok:
            return None, 0.0
        try:
            import pytesseract
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            text = pytesseract.image_to_string(bw, config="--psm 7 -c tessedit_char_whitelist=0123456789")
            digits = re.findall(r"\d+", text)
            if not digits:
                return None, 0.0
            candidate = max(digits, key=len)
            if len(candidate) > 2:
                candidate = candidate[:2]
            return candidate, 0.7
        except Exception:
            return None, 0.0

    def recognize_jersey_number(self, crop: np.ndarray) -> tuple[str | None, float]:
        if crop is None or crop.size == 0:
            return None, 0.0
        if self._tesseract_ok:
            return self._recognize_by_tesseract(crop)
        return None, 0.0

    def recognize_player_jersey_number(self, crop: np.ndarray) -> tuple[str | None, float]:
        candidates: list[tuple[str, float]] = []
        for view in self._generate_player_crop_views(crop):
            text, conf = self.recognize_jersey_number(view)
            clean = self._clean_jersey_number(text)
            if clean:
                candidates.append((clean, float(conf)))
        if not candidates:
            return None, 0.0
        votes: dict[str, float] = defaultdict(float)
        best_conf: dict[str, float] = defaultdict(float)
        for text, conf in candidates:
            votes[text] += max(0.2, conf)
            best_conf[text] = max(best_conf[text], conf)
        ranked = sorted(votes.items(), key=lambda item: (item[1], best_conf[item[0]], len(item[0])), reverse=True)
        best_text = ranked[0][0]
        return best_text, float(best_conf[best_text])

    @staticmethod
    def _clean_jersey_number(text: str | None) -> str | None:
        if not text:
            return None
        digits = re.sub(r"\D", "", text)[:2]
        return digits if digits else None


class RealJerseyRecognizer:
    def __init__(self) -> None:
        self.last_run_metadata = RecognitionRunMetadata(message="", ocr_available=False, warnings=[])
        self._number_recognizer: NumberRecognizer | None = None
        self._detector: Any = None
        self._face_available = False

    def _get_detector(self):
        if self._detector is None:
            self._detector = _load_yolo_detector()
        return self._detector

    def _get_number_recognizer(self) -> NumberRecognizer:
        if self._number_recognizer is not None:
            return self._number_recognizer
        self._number_recognizer = NumberRecognizer()
        return self._number_recognizer

    def _init_face_clustering(self):
        if self._face_available:
            return True
        try:
            from app.services.face_clustering import extract_face_embedding, cluster_embeddings
            self._face_extract_fn = extract_face_embedding
            self._face_cluster_fn = cluster_embeddings
            self._face_available = True
            return True
        except Exception:
            self._face_available = False
            return False

    def _detect_players(self, frame: np.ndarray) -> tuple[list[list[int]], list[float]]:
        model = self._get_detector()
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
            self.last_run_metadata.warnings.append(f"YOLO predict failed: {exc}")

        if not boxes:
            self.last_run_metadata.warnings.append("No players detected, using full-frame fallback")
            return [[0, 0, frame_w, frame_h]], [0.5]
        return boxes, confs

    @staticmethod
    def _estimate_jersey_bbox(player_bbox: list[int], image_width: int, image_height: int) -> list[int]:
        x1, y1, x2, y2 = player_bbox
        width = x2 - x1
        height = y2 - y1
        jersey = [
            x1 + int(width * 0.22),
            y1 + int(height * 0.16),
            x2 - int(width * 0.18),
            y1 + int(height * 0.52),
        ]
        jersey[0] = max(0, min(jersey[0], image_width - 1))
        jersey[1] = max(0, min(jersey[1], image_height - 1))
        jersey[2] = max(jersey[0] + 1, min(jersey[2], image_width))
        jersey[3] = max(jersey[1] + 1, min(jersey[3], image_height))
        return jersey

    @staticmethod
    def _crop(frame: np.ndarray, bbox: list[int]) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        return frame[y1:y2, x1:x2].copy()

    def run(self, video_id: int, db: Session) -> list[JerseyPrediction]:
        video = db.query(VideoAsset).filter(VideoAsset.id == video_id).first()
        if video is None:
            raise RealRecognitionConfigurationError("Video not found")

        frames = db.query(VideoFrame).filter(VideoFrame.video_id == video_id).order_by(VideoFrame.frame_id.asc()).all()
        if not frames:
            raise RealRecognitionConfigurationError("No extracted frames found for this video")

        number_recognizer = self._get_number_recognizer()
        face_available = self._init_face_clustering()

        predictions: list[JerseyPrediction] = []
        warnings: list[str] = []
        temp_embeddings: dict[int, np.ndarray] = {}
        temp_track_map: dict[int, int] = {}
        next_temp_id = 1

        for frame_row in frames[:: max(1, settings.recognition_frame_stride)]:
            frame_path = Path(frame_row.frame_path)
            if not frame_path.exists():
                warnings.append(f"Missing frame image: {frame_path}")
                continue

            image_bytes = np.fromfile(str(frame_path), dtype=np.uint8)
            frame = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
            if frame is None:
                warnings.append(f"Unreadable frame image: {frame_path}")
                continue

            frame_height, frame_width = frame.shape[:2]
            player_bboxes, player_confs = self._detect_players(frame)

            for index, player_bbox in enumerate(player_bboxes):
                temp_id = next_temp_id
                next_temp_id += 1

                jersey_bbox = self._estimate_jersey_bbox(player_bbox, frame_width, frame_height)
                detector_conf = float(player_confs[index]) if index < len(player_confs) else 0.0

                predicted_number = ""
                ocr_conf = 0.0
                try:
                    crop = self._crop(frame, jersey_bbox)
                    predicted_number, ocr_conf = number_recognizer.recognize_player_jersey_number(crop)
                    predicted_number = predicted_number or ""
                except Exception as exc:
                    warnings.append(f"OCR error frame {frame_row.frame_id}: {exc}")

                confidence = float(ocr_conf if predicted_number else detector_conf)
                predictions.append(
                    JerseyPrediction(
                        video_id=video.id,
                        frame_row_id=frame_row.id,
                        frame_id=frame_row.frame_id,
                        track_id=temp_id,
                        player_bbox=json.dumps(player_bbox),
                        jersey_bbox=json.dumps(jersey_bbox),
                        predicted_number=predicted_number,
                        confidence=round(confidence, 3),
                        source=PREDICTION_SOURCE_REAL,
                    )
                )
                temp_track_map[temp_id] = len(predictions) - 1

                if face_available:
                    try:
                        emb = self._face_extract_fn(frame, player_bbox)
                        if emb is not None:
                            temp_embeddings[temp_id] = emb
                    except Exception:
                        pass

        if not predictions:
            raise RealRecognitionConfigurationError("No predictions produced.")

        if face_available and temp_embeddings:
            face_clusters = self._face_cluster_fn(temp_embeddings, threshold=FACE_CLUSTER_THRESHOLD)
            for temp_id, cluster_id in face_clusters.items():
                pred_idx = temp_track_map[temp_id]
                predictions[pred_idx].track_id = int(cluster_id)
            unique_players = len(set(face_clusters.values()))
            warnings.append(f"Face clustering: {unique_players} unique players")

        message = "Recognition completed"
        if warnings:
            message = "; ".join(dict.fromkeys(warnings))
        self.last_run_metadata = RecognitionRunMetadata(
            message=message,
            ocr_available=number_recognizer._tesseract_ok,
            warnings=warnings,
        )
        return predictions