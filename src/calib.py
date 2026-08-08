"""Court calibration: homography solving and image -> court projection.

Salvaged from:
  SRC_A/run_basketball_topdown_projection.py
    - BasketballHalfCourtConfig (subset)  (lines 100-117)
    - CourtCalibrationData                (lines 120-124)
    - _validate_named_points              (lines 463-477)
    - load_court_calibration              (lines 480-529)
    - ViewTransformer                     (lines 532-563)
    - clip_point_xy                       (lines 241-244)
  Cross-checked against SRC_A/0619/projection.py:106-148
    (ViewTransformer / load_calibration / project_bbox_bottom_center)

Design note: the homography is solved ONCE, outside the frame loop. The 0619
inputs were confirmed to be a fixed camera (PlaySight fixed installation), so
per-frame re-estimation would only add cost and jitter.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np


def safe_float(v: Any, default: float = 0.0) -> float:
    """Salvaged from SRC_A/run_basketball_topdown_projection.py (safe_float helper)."""
    try:
        return float(v)
    except Exception:
        return default


@dataclass
class CourtCanvasConfig:
    """Size of the top-down canvas that ``court_points`` are expressed in.

    IMPORTANT ORIENTATION MISMATCH - see the Phase A report.
    SRC_A/court_calibration_manual.json declares ``topdown_template_size:
    [500, 940]`` and its court_points span x<=468, y<=908, i.e. a PORTRAIT
    500x940 canvas. But SRC_A's ``BasketballHalfCourtConfig`` (line 104-105)
    and SRC_A/0619/projection.py's ``CourtConfig`` (line 15-16) both default to
    width=940, height=500, i.e. LANDSCAPE. The two are inconsistent in the
    source. Defaults here follow the calibration JSON (portrait), because that
    is the artifact the homography is actually fitted against. Override via
    config if you re-calibrate against a landscape canvas.
    """

    width: int = 500
    height: int = 940
    margin: int = 32


@dataclass
class CourtCalibrationData:
    """Salvaged verbatim from source lines 120-124."""

    point_names: list[str]
    image_points: np.ndarray
    court_points: np.ndarray


def clip_point_xy(point: tuple[float, float] | list[float], width: int, height: int) -> list[float]:
    """Salvaged verbatim from source lines 241-244."""
    x = float(min(max(0.0, float(point[0])), max(0, width - 1)))
    y = float(min(max(0.0, float(point[1])), max(0, height - 1)))
    return [x, y]


def _validate_named_points(points: Any, field_name: str) -> dict[str, list[float]]:
    """Salvaged verbatim from source lines 463-477."""
    if not isinstance(points, dict):
        raise ValueError(f"Calibration field '{field_name}' must be an object mapping names to [x, y].")
    normalized: dict[str, list[float]] = {}
    for name, value in points.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Calibration field '{field_name}' contains an invalid empty point name.")
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"Calibration point '{name}' in '{field_name}' must be a 2-value array [x, y].")
        x = safe_float(value[0], float("nan"))
        y = safe_float(value[1], float("nan"))
        if not np.isfinite(x) or not np.isfinite(y):
            raise ValueError(f"Calibration point '{name}' in '{field_name}' must contain finite numbers.")
        normalized[name] = [float(x), float(y)]
    return normalized


def load_court_calibration(path: str | Path | None) -> CourtCalibrationData:
    """Load manual calibration JSON.

    Salvaged verbatim from source lines 480-529, except that the ``path`` is
    normalized through ``pathlib.Path`` only (the source already did this).

    Expected structure:
    {
      "image_points": {"corner_tl": [x, y], ...},
      "court_points": {"corner_tl": [X, Y], ...}
    }

    Point names must match exactly. At least 4 pairs are required for homography.
    """
    if path is None or str(path).strip() == "":
        raise ValueError("Calibration path is empty.")

    calibration_path = Path(path)
    if not calibration_path.exists():
        raise FileNotFoundError(f"Calibration JSON not found: {calibration_path}")

    with calibration_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError("Calibration JSON root must be an object.")

    image_points_map = _validate_named_points(payload.get("image_points"), "image_points")
    court_points_map = _validate_named_points(payload.get("court_points"), "court_points")

    image_keys = set(image_points_map.keys())
    court_keys = set(court_points_map.keys())
    if image_keys != court_keys:
        missing_in_image = sorted(court_keys - image_keys)
        missing_in_court = sorted(image_keys - court_keys)
        msg_parts: list[str] = ["Calibration point names must match between image_points and court_points."]
        if missing_in_image:
            msg_parts.append(f"missing in image_points={missing_in_image}")
        if missing_in_court:
            msg_parts.append(f"missing in court_points={missing_in_court}")
        raise ValueError(" ".join(msg_parts))

    point_names = sorted(image_points_map.keys())
    if len(point_names) < 4:
        raise ValueError(
            f"At least 4 calibration point pairs are required for homography, got {len(point_names)}."
        )

    image_points = np.asarray([image_points_map[name] for name in point_names], dtype=np.float32)
    court_points = np.asarray([court_points_map[name] for name in point_names], dtype=np.float32)

    return CourtCalibrationData(point_names=point_names, image_points=image_points, court_points=court_points)


class ViewTransformer:
    """Homography from image pixels to court canvas pixels.

    Salvaged verbatim from source lines 532-563, including ``method=0``
    (plain least-squares, no RANSAC). ``method=0`` is deliberate for a
    hand-clicked point set: every point is trusted, none are outliers.
    """

    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = np.asarray(source, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)
        if source.ndim != 2 or source.shape[1] != 2:
            raise ValueError(f"source must have shape (N, 2), got {source.shape}")
        if target.ndim != 2 or target.shape[1] != 2:
            raise ValueError(f"target must have shape (N, 2), got {target.shape}")
        if source.shape != target.shape:
            raise ValueError(f"source/target shape mismatch: {source.shape} vs {target.shape}")
        if source.shape[0] < 4:
            raise ValueError(f"At least 4 point pairs are required, got {source.shape[0]}")

        homography, inlier_mask = cv2.findHomography(source, target, method=0)
        if homography is None:
            raise RuntimeError("cv2.findHomography failed to compute a valid homography matrix.")

        self.source = source
        self.target = target
        self.homography = homography.astype(np.float32)
        self.inlier_mask = inlier_mask.astype(np.uint8) if inlier_mask is not None else None

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """Salvaged verbatim from source lines 554-563."""
        pts = np.asarray(points, dtype=np.float32)
        if pts.size == 0:
            return np.zeros((0, 2), dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(f"points must have shape (N, 2), got {pts.shape}")
        transformed = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), self.homography)
        if transformed is None:
            raise RuntimeError("cv2.perspectiveTransform returned None.")
        return transformed.reshape(-1, 2)


def bbox_ground_point(bbox_xyxy: list[float]) -> tuple[float, float]:
    """Foot position of a detection: midpoint of the box BOTTOM edge.

    Deliberately NOT the box centre. A player's contact point with the floor is
    what the court homography is valid for; projecting the box centre would put
    the player metres up-court under perspective.

    Source check (requested): neither salvaged implementation used the box
    centre, so no logic correction was needed.
      - SRC_A/0619/projection.py:137-139 already used ``[(x1+x2)/2.0, y2]``.
      - SRC_A/run_basketball_topdown_projection.py:1183-1208
        (``extract_ground_point_from_pose``) preferred the midpoint of the two
        COCO ankle keypoints and only fell back to ``[(x1+x2)/2.0, y2]``.
        Pose is out of scope for this project, so the fallback path becomes the
        only path, which is exactly the 0619 behaviour.
    """
    x1, _, x2, y2 = [float(v) for v in bbox_xyxy]
    return (x1 + x2) / 2.0, y2


class CourtProjector:
    """One-shot calibration loaded outside the frame loop; projects points on demand."""

    def __init__(
        self,
        calibration: CourtCalibrationData,
        canvas: Optional[CourtCanvasConfig] = None,
    ) -> None:
        self.calibration = calibration
        self.canvas = canvas or CourtCanvasConfig()
        self.transformer = ViewTransformer(calibration.image_points, calibration.court_points)

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        canvas: Optional[CourtCanvasConfig] = None,
    ) -> "CourtProjector":
        return cls(load_court_calibration(path), canvas=canvas)

    @property
    def homography(self) -> np.ndarray:
        return self.transformer.homography

    def project_point(self, x: float, y: float) -> tuple[float, float]:
        """Project one image-space point to court-canvas space.

        The result is clipped to the canvas, matching the source's
        ``project_ground_point_to_court`` (lines 1211-1238), which returned
        ``clip_point_xy(...)``.
        """
        projected = self.transformer.transform_points(
            np.asarray([[float(x), float(y)]], dtype=np.float32)
        )
        if projected.shape[0] == 0:
            raise RuntimeError("Projection produced no output point.")
        px = safe_float(projected[0, 0], float("nan"))
        py = safe_float(projected[0, 1], float("nan"))
        if not np.isfinite(px) or not np.isfinite(py):
            raise ValueError(f"Projection produced a non-finite point for input ({x}, {y}).")
        clipped = clip_point_xy((px, py), self.canvas.width, self.canvas.height)
        return clipped[0], clipped[1]

    def project_bbox_ground_point(self, bbox_xyxy: list[float]) -> tuple[float, float]:
        """Project a detection box to the court using its bottom-edge midpoint."""
        gx, gy = bbox_ground_point(bbox_xyxy)
        return self.project_point(gx, gy)
