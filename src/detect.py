"""YOLO26 detector wrapper producing framework-agnostic detections.

This module is the ONLY place in the project allowed to touch Ultralytics
detection APIs or Ultralytics ``Results`` objects. Everything downstream sees
plain ``Detection`` dataclasses. Swapping in a non-Ultralytics detector later
(D-FINE, RT-DETRv3, ...) should require changes inside this file only.

Known and accepted exception: ``src/track.py`` calls ``model.track(...)``
directly, because Ultralytics couples detection and tracking in that call.
That coupling is deliberate and documented there.

There is NO fallback to a COCO checkpoint. If the configured weights are
missing the constructor raises. A generic COCO model would silently produce
"person" boxes under class id 0, which happens to collide with this project's
"player" id, so a silent fallback would look like it worked while every ball,
rim and jersey silently vanished.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Class scheme - must stay in sync with train/basketball.yaml
# --------------------------------------------------------------------------

CLASS_PLAYER = 0
CLASS_BALL = 1
CLASS_RIM = 2
CLASS_JERSEY = 3

BASKETBALL_CLASS_NAMES: dict[int, str] = {
    CLASS_PLAYER: "player",
    CLASS_BALL: "ball",
    CLASS_RIM: "rim",
    CLASS_JERSEY: "jersey",
}


# --------------------------------------------------------------------------
# Framework-agnostic output type
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Detection:
    """One detected object in one frame. Deliberately minimal.

    bbox_xyxy is absolute pixel coordinates in the source frame, ordered
    (x1, y1, x2, y2) with x1 <= x2 and y1 <= y2.
    """

    bbox_xyxy: tuple[float, float, float, float]
    class_id: int
    confidence: float

    @property
    def width(self) -> float:
        return float(self.bbox_xyxy[2] - self.bbox_xyxy[0])

    @property
    def height(self) -> float:
        return float(self.bbox_xyxy[3] - self.bbox_xyxy[1])

    @property
    def center_xy(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def ground_xy(self) -> tuple[float, float]:
        """Bottom-edge midpoint: the foot position used for court projection."""
        x1, _, x2, y2 = self.bbox_xyxy
        return ((x1 + x2) / 2.0, y2)


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


@dataclass
class DetectorConfig:
    """Detector settings. Populated from configs/default.yaml by src/config.py.

    All values below are STARTING POINTS, not values recovered from any tuned
    source. They need calibration once real weights exist.
    """

    weights: Path
    conf: float = 0.25
    iou: float = 0.45
    # 960 matches train/train.py; ball and rim are small objects and lose
    # badly at 640. Inference imgsz should match training imgsz.
    imgsz: int = 960
    device: str = "0"
    half: bool = False
    max_det: int = 60
    # When True, a weights file whose class names do not match the basketball
    # 4-class scheme is a hard error instead of a warning. Left False so the
    # smoke test and tracker comparison can deliberately run official COCO
    # weights before any custom training has happened.
    require_basketball_classes: bool = False


class DetectorWeightsNotFoundError(FileNotFoundError):
    """Raised when the configured weights file does not exist."""


class DetectorClassMismatchError(ValueError):
    """Raised when weights do not expose the basketball 4-class scheme."""


# --------------------------------------------------------------------------
# Detector
# --------------------------------------------------------------------------


class Detector:
    """Ultralytics YOLO26 detector returning ``list[Detection]``."""

    def __init__(self, config: DetectorConfig) -> None:
        self.config = config
        self.weights_path = Path(config.weights)

        if not self.weights_path.exists():
            raise DetectorWeightsNotFoundError(
                f"Detector weights not found: {self.weights_path}\n"
                "This project does NOT fall back to a COCO checkpoint, because COCO "
                "class 0 is 'person' and would masquerade as 'player' while ball, rim "
                "and jersey silently disappear.\n"
                "Fix one of:\n"
                "  1. train weights first:  python train/train.py train --data train/basketball.yaml\n"
                "  2. point detector.weights in configs/default.yaml at an existing .pt\n"
                "  3. for a pre-training smoke test, pass an official checkpoint explicitly "
                "(scripts/smoke_test.py does this on purpose)."
            )

        # Import here, not at module import time, so that this module can be
        # imported for its dataclasses in environments without torch.
        from ultralytics import YOLO

        self.model = YOLO(str(self.weights_path))
        self.names: dict[int, str] = self._read_names(self.model)
        self._check_class_scheme()

    @staticmethod
    def _read_names(model: Any) -> dict[int, str]:
        names = getattr(model, "names", {}) or {}
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        if isinstance(names, (list, tuple)):
            return {int(i): str(v) for i, v in enumerate(names)}
        return {}

    def _check_class_scheme(self) -> None:
        """Compare the checkpoint's classes against the basketball scheme."""
        expected = BASKETBALL_CLASS_NAMES
        actual_lower = {idx: name.lower() for idx, name in self.names.items()}
        matches = all(actual_lower.get(idx) == name for idx, name in expected.items())

        if matches:
            LOGGER.info("Detector loaded with basketball 4-class scheme: %s", self.names)
            return

        message = (
            f"Weights '{self.weights_path.name}' do NOT expose the basketball 4-class "
            f"scheme {expected}. Found {len(self.names)} classes: {self.names}\n"
            "Ball / rim / jersey results will be wrong or empty."
        )
        if self.config.require_basketball_classes:
            raise DetectorClassMismatchError(message)
        LOGGER.warning("%s\n(continuing because require_basketball_classes is False)", message)

    @property
    def is_basketball_model(self) -> bool:
        expected = BASKETBALL_CLASS_NAMES
        actual_lower = {idx: name.lower() for idx, name in self.names.items()}
        return all(actual_lower.get(idx) == name for idx, name in expected.items())

    def class_name(self, class_id: int) -> str:
        return self.names.get(int(class_id), str(class_id))

    def detect(
        self,
        frame: "np.ndarray",
        classes: Optional[Sequence[int]] = None,
    ) -> list[Detection]:
        """Run detection on one BGR frame.

        Args:
            frame: HxWx3 BGR image.
            classes: optional class-id allowlist passed to the model.
        Returns:
            Detections sorted by descending confidence. Never an Ultralytics object.
        """
        results = self.model.predict(
            frame,
            conf=self.config.conf,
            iou=self.config.iou,
            imgsz=self.config.imgsz,
            device=self.config.device,
            half=self.config.half,
            max_det=self.config.max_det,
            classes=list(classes) if classes is not None else None,
            verbose=False,
        )
        return self._to_detections(results)

    @staticmethod
    def _to_detections(results: Any) -> list[Detection]:
        """Convert Ultralytics Results into plain dataclasses.

        This is the containment boundary: nothing Ultralytics-shaped escapes.
        """
        out: list[Detection] = []
        if not results:
            return out

        boxes = getattr(results[0], "boxes", None)
        if boxes is None or boxes.xyxy is None or len(boxes) == 0:
            return out

        xyxy = boxes.xyxy.detach().cpu().numpy()
        cls = (
            boxes.cls.detach().cpu().numpy()
            if getattr(boxes, "cls", None) is not None
            else [-1] * len(xyxy)
        )
        conf = (
            boxes.conf.detach().cpu().numpy()
            if getattr(boxes, "conf", None) is not None
            else [0.0] * len(xyxy)
        )

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = (float(v) for v in xyxy[i].tolist())
            out.append(
                Detection(
                    bbox_xyxy=(x1, y1, x2, y2),
                    class_id=int(cls[i]),
                    confidence=float(conf[i]),
                )
            )
        out.sort(key=lambda d: d.confidence, reverse=True)
        return out


# --------------------------------------------------------------------------
# Helpers usable without a Detector instance
# --------------------------------------------------------------------------


def filter_by_class(detections: Sequence[Detection], class_id: int) -> list[Detection]:
    return [d for d in detections if d.class_id == int(class_id)]


def split_by_class(
    detections: Sequence[Detection],
) -> dict[int, list[Detection]]:
    """Group detections by class id. Missing classes yield empty lists."""
    grouped: dict[int, list[Detection]] = {cid: [] for cid in BASKETBALL_CLASS_NAMES}
    for det in detections:
        grouped.setdefault(det.class_id, []).append(det)
    return grouped


def best_detection(detections: Sequence[Detection], class_id: int) -> Optional[Detection]:
    """Highest-confidence detection of a class, or None.

    Useful for ball and rim, where at most one instance is expected per frame.
    """
    candidates = filter_by_class(detections, class_id)
    if not candidates:
        return None
    return max(candidates, key=lambda d: d.confidence)
