"""Rendering for output 1 (annotated original view) and output 2 (top-down).

Salvaged from:
  SRC_A/0619/projection.py:51-104        -> draw_half_court_template (+ helpers 29-48)
  SRC_A/0619/render.py:46-88             -> TopdownRenderer trail/point drawing
  SRC_A/run_basketball_topdown_projection.py:1241-1276
                                         -> smooth_point_ema + update_topdown_history
  SRC_A/run_basketball_topdown_projection.py:1411-1462
                                         -> draw_player_analysis_overlay (base for output 1)

Note on the output-1 source: the A3 brief listed only top-down sources, so the
annotated-view drawing was based on ``draw_player_analysis_overlay`` from the
topdown script rather than SRC_A/0619/render.py:13-43 (``build_main_frame``).
Reason: ``build_main_frame`` is written against the ``supervision`` package
(sv.BoxAnnotator / sv.LabelAnnotator), which would add a dependency this
project does not otherwise need; ``draw_player_analysis_overlay`` is pure cv2.
Its colour map, font, scale and thickness are carried over unchanged.

All court geometry constants come from SRC_A/0619/projection.py:60-70 verbatim.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np


# --------------------------------------------------------------------------
# Shared frame data structure (used by BOTH renderers)
# --------------------------------------------------------------------------


@dataclass
class PlayerRender:
    """One tracked player for one frame."""

    track_id: int
    bbox_xyxy: list[float]
    jersey_number: Optional[str] = None
    team_id: Optional[int] = None
    conf: float = 0.0
    # Court-canvas coordinates from calib.CourtProjector (bottom-edge midpoint).
    court_xy: Optional[tuple[float, float]] = None


@dataclass
class BallRender:
    bbox_xyxy: list[float]
    conf: float = 0.0
    track_id: Optional[int] = None
    court_xy: Optional[tuple[float, float]] = None


@dataclass
class RimRender:
    bbox_xyxy: list[float]
    conf: float = 0.0


@dataclass
class FrameRenderData:
    """The single per-frame structure shared by both renderers."""

    frame_idx: int
    timestamp_sec: float
    players: list[PlayerRender] = field(default_factory=list)
    ball: Optional[BallRender] = None
    rims: list[RimRender] = field(default_factory=list)


# --------------------------------------------------------------------------
# Court canvas config + template drawing
# --------------------------------------------------------------------------


@dataclass
class CourtRenderConfig:
    """Top-down canvas appearance.

    Colours/thicknesses salvaged from SRC_A/0619/projection.py:13-27.

    NOTE the orientation caveat documented in calib.CourtCanvasConfig: the
    source defaulted to 940x500 (landscape) while the shipped calibration JSON
    targets 500x940 (portrait). Defaults here follow the calibration JSON so
    the drawn template and the projected points share one coordinate space.
    Keep width/height in sync with calib.CourtCanvasConfig.
    """

    width: int = 500
    height: int = 940
    margin: int = 32
    court_width_m: float = 15.0
    court_height_m: float = 11.0
    court_color: tuple[int, int, int] = (170, 205, 232)
    paint_color: tuple[int, int, int] = (156, 194, 224)
    line_color: tuple[int, int, int] = (220, 120, 30)
    accent_color: tuple[int, int, int] = (235, 210, 110)
    background_color: tuple[int, int, int] = (18, 24, 36)
    line_thickness: int = 2
    player_radius: int = 7
    trail_thickness: int = 2
    trail_length: int = 30


def court_to_pixel(x_m: float, y_m: float, config: CourtRenderConfig) -> list[float]:
    """Salvaged verbatim from SRC_A/0619/projection.py:29-36."""
    drawable_width = max(1.0, float(config.width - 2 * config.margin))
    drawable_height = max(1.0, float(config.height - 2 * config.margin))
    left = float(config.margin)
    bottom = float(config.height - config.margin)
    pixel_x = left + (float(x_m) / config.court_width_m) * drawable_width
    pixel_y = bottom - (float(y_m) / config.court_height_m) * drawable_height
    return [float(pixel_x), float(pixel_y)]


def _arc_polyline_meters(
    cx_m: float,
    cy_m: float,
    radius_m: float,
    start_deg: float,
    end_deg: float,
    config: CourtRenderConfig,
) -> np.ndarray:
    """Salvaged verbatim from SRC_A/0619/projection.py:39-48."""
    angles = np.linspace(start_deg, end_deg, 120, dtype=np.float32)
    pts = []
    for ang_deg in angles.tolist():
        ang = math.radians(ang_deg)
        x_m = cx_m + radius_m * math.cos(ang)
        y_m = cy_m + radius_m * math.sin(ang)
        px, py = court_to_pixel(x_m, y_m, config)
        pts.append([int(round(px)), int(round(py))])
    return np.asarray(pts, dtype=np.int32).reshape(-1, 1, 2)


def draw_half_court_template(config: CourtRenderConfig) -> np.ndarray:
    """Salvaged verbatim from SRC_A/0619/projection.py:51-103.

    All court dimensions (hoop at 7.5m/1.575m, free-throw line 5.80m, lane half
    width 2.45m, three-point radius 6.75m, etc.) are FIBA 3x3 half-court values
    carried over unchanged.
    """
    canvas = np.full((config.height, config.width, 3), config.background_color, dtype=np.uint8)
    gradient = np.linspace(0.0, 1.0, config.height, dtype=np.float32).reshape(config.height, 1, 1)
    court_base = np.full((config.height, config.width, 3), config.court_color, dtype=np.float32)
    court_top = np.full((config.height, config.width, 3), config.paint_color, dtype=np.float32)
    canvas[:] = (court_base * (1.0 - gradient) + court_top * gradient).astype(np.uint8)

    left, bottom = court_to_pixel(0.0, 0.0, config)
    right, top = court_to_pixel(15.0, 11.0, config)
    hoop_x_m = 7.5
    hoop_y_m = 1.575
    free_throw_y_m = 5.80
    lane_half_width_m = 2.45
    three_left_x_m = 0.90
    three_right_x_m = 14.10
    three_radius_m = 6.75
    restricted_radius_m = 1.25
    free_throw_radius_m = 1.80
    backboard_half_width_m = 0.90
    backboard_y_m = 1.20

    cv2.rectangle(
        canvas,
        (int(round(left)), int(round(top))),
        (int(round(right)), int(round(bottom))),
        config.line_color,
        config.line_thickness,
    )

    paint_bl = court_to_pixel(hoop_x_m - lane_half_width_m, 0.0, config)
    paint_tr = court_to_pixel(hoop_x_m + lane_half_width_m, free_throw_y_m, config)
    cv2.rectangle(
        canvas,
        (int(round(paint_bl[0])), int(round(paint_tr[1]))),
        (int(round(paint_tr[0])), int(round(paint_bl[1]))),
        config.line_color,
        config.line_thickness,
    )

    free_throw_left = court_to_pixel(hoop_x_m - lane_half_width_m, free_throw_y_m, config)
    free_throw_right = court_to_pixel(hoop_x_m + lane_half_width_m, free_throw_y_m, config)
    cv2.line(
        canvas,
        (int(round(free_throw_left[0])), int(round(free_throw_left[1]))),
        (int(round(free_throw_right[0])), int(round(free_throw_right[1]))),
        config.line_color,
        config.line_thickness,
        cv2.LINE_AA,
    )
    cv2.polylines(
        canvas,
        [_arc_polyline_meters(hoop_x_m, free_throw_y_m, free_throw_radius_m, 0.0, 180.0, config)],
        False,
        config.line_color,
        config.line_thickness,
        cv2.LINE_AA,
    )

    hoop_px, hoop_py = court_to_pixel(hoop_x_m, hoop_y_m, config)
    cv2.circle(canvas, (int(round(hoop_px)), int(round(hoop_py))), 7, config.accent_color, -1, cv2.LINE_AA)
    cv2.polylines(
        canvas,
        [_arc_polyline_meters(hoop_x_m, hoop_y_m, restricted_radius_m, 0.0, 180.0, config)],
        False,
        config.line_color,
        config.line_thickness,
        cv2.LINE_AA,
    )

    backboard_left = court_to_pixel(hoop_x_m - backboard_half_width_m, backboard_y_m, config)
    backboard_right = court_to_pixel(hoop_x_m + backboard_half_width_m, backboard_y_m, config)
    cv2.line(
        canvas,
        (int(round(backboard_left[0])), int(round(backboard_left[1]))),
        (int(round(backboard_right[0])), int(round(backboard_right[1]))),
        config.line_color,
        config.line_thickness,
        cv2.LINE_AA,
    )

    three_break_y_m = hoop_y_m + math.sqrt(
        max(0.0, three_radius_m * three_radius_m - (hoop_x_m - three_left_x_m) ** 2)
    )
    left_base = court_to_pixel(three_left_x_m, 0.0, config)
    left_break = court_to_pixel(three_left_x_m, three_break_y_m, config)
    right_base = court_to_pixel(three_right_x_m, 0.0, config)
    right_break = court_to_pixel(three_right_x_m, three_break_y_m, config)
    cv2.line(
        canvas,
        (int(round(left_base[0])), int(round(left_base[1]))),
        (int(round(left_break[0])), int(round(left_break[1]))),
        config.line_color,
        config.line_thickness,
        cv2.LINE_AA,
    )
    cv2.line(
        canvas,
        (int(round(right_base[0])), int(round(right_base[1]))),
        (int(round(right_break[0])), int(round(right_break[1]))),
        config.line_color,
        config.line_thickness,
        cv2.LINE_AA,
    )
    left_angle = math.degrees(math.atan2(three_break_y_m - hoop_y_m, three_left_x_m - hoop_x_m))
    right_angle = math.degrees(math.atan2(three_break_y_m - hoop_y_m, three_right_x_m - hoop_x_m))
    cv2.polylines(
        canvas,
        [_arc_polyline_meters(hoop_x_m, hoop_y_m, three_radius_m, left_angle, right_angle, config)],
        False,
        config.line_color,
        config.line_thickness,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        "3x3 HALF COURT",
        (int(round(left + 8)), int(round(top + 28))),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        config.line_color,
        2,
        cv2.LINE_AA,
    )
    return canvas


# --------------------------------------------------------------------------
# Trail history (EMA smoothing + bounded deque)
# --------------------------------------------------------------------------


def smooth_point_ema(
    previous_point: Optional[tuple[float, float]],
    current_point: tuple[float, float],
    alpha: float,
) -> tuple[float, float]:
    """Salvaged verbatim from SRC_A/run_basketball_topdown_projection.py:1241-1251."""
    a = float(min(max(alpha, 0.0), 1.0))
    if previous_point is None:
        return current_point
    x = float(a * current_point[0] + (1.0 - a) * previous_point[0])
    y = float(a * current_point[1] + (1.0 - a) * previous_point[1])
    return (x, y)


class TrailHistory:
    """Per-track smoothed position history for the top-down trail.

    Transcribed from ``update_topdown_history``
    (SRC_A/run_basketball_topdown_projection.py:1254-1276). The EMA-then-append
    order and the deque re-allocation on trail-length change are preserved; the
    two module-level dicts the source threaded through arguments are now
    instance state.

    The default ``alpha`` of 0.5 is NOT a recovered constant: the source read it
    from an argparse value that is not present in the salvaged range. Treat it
    as a starting point (see the Phase A report).
    """

    def __init__(self, trail_length: int = 30, alpha: float = 0.5) -> None:
        self.trail_length = int(trail_length)
        self.alpha = float(alpha)
        self.smoothed_points: dict[int, tuple[float, float]] = {}
        self.track_history_2d: dict[int, deque] = {}

    def update(
        self,
        track_id: int,
        projected_point: Optional[tuple[float, float] | list[float]],
        projection_valid: bool = True,
    ) -> Optional[list[float]]:
        if not projection_valid or projected_point is None:
            return None

        current_tuple = (float(projected_point[0]), float(projected_point[1]))
        prev_smooth = self.smoothed_points.get(track_id)
        smoothed = smooth_point_ema(prev_smooth, current_tuple, self.alpha)
        self.smoothed_points[track_id] = smoothed

        history = self.track_history_2d.get(track_id)
        if history is None or history.maxlen != max(1, self.trail_length):
            history = deque(maxlen=max(1, self.trail_length))
            self.track_history_2d[track_id] = history
        history.append(smoothed)
        return [float(smoothed[0]), float(smoothed[1])]

    def get(self, track_id: int) -> list[tuple[float, float]]:
        return list(self.track_history_2d.get(track_id, ()))


# --------------------------------------------------------------------------
# Output 1: annotated original view
# --------------------------------------------------------------------------

# Salvaged from SRC_A/run_basketball_topdown_projection.py:1419-1423.
TEAM_COLOR: dict[Optional[int], tuple[int, int, int]] = {
    0: (40, 180, 20),
    1: (20, 100, 240),
    None: (180, 180, 180),
}
BALL_COLOR: tuple[int, int, int] = (0, 140, 255)
RIM_COLOR: tuple[int, int, int] = (60, 220, 255)


def render_annotated_frame(
    frame: np.ndarray,
    data: FrameRenderData,
    line_width: int = 2,
    show_timestamp: bool = True,
) -> np.ndarray:
    """Output 1: original camera view with player / ball / rim boxes.

    Player box + label drawing is transcribed from
    ``draw_player_analysis_overlay`` (source lines 1411-1462), minus the pose
    skeleton and mask-contour branches (both out of scope for this project).
    The label now leads with the stabilized jersey number, which is the point of
    output 1; it falls back to the track id when no number has locked yet.

    Ball and rim drawing are NEW here: the salvaged overlay drew players only.
    """
    out = frame.copy()

    for player in data.players:
        bbox = player.bbox_xyxy
        if not bbox:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]
        color = TEAM_COLOR.get(player.team_id, TEAM_COLOR[None])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, line_width)

        number = player.jersey_number
        label = f"#{number}" if number else f"id:{player.track_id}"
        cv2.putText(
            out,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            2,
            cv2.LINE_AA,
        )

    for rim in data.rims:
        x1, y1, x2, y2 = [int(v) for v in rim.bbox_xyxy]
        cv2.rectangle(out, (x1, y1), (x2, y2), RIM_COLOR, line_width)
        cv2.putText(
            out, "rim", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, RIM_COLOR, 2, cv2.LINE_AA
        )

    if data.ball is not None:
        x1, y1, x2, y2 = [int(v) for v in data.ball.bbox_xyxy]
        cv2.rectangle(out, (x1, y1), (x2, y2), BALL_COLOR, line_width)
        cv2.circle(out, (int((x1 + x2) / 2), int((y1 + y2) / 2)), 4, BALL_COLOR, -1)
        cv2.putText(
            out, "ball", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, BALL_COLOR, 2, cv2.LINE_AA
        )

    if show_timestamp:
        cv2.putText(
            out,
            f"frame={data.frame_idx} time={data.timestamp_sec:.2f}s",
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return out


# --------------------------------------------------------------------------
# Output 2: top-down view
# --------------------------------------------------------------------------


def render_topdown_frame(
    court_template: np.ndarray,
    data: FrameRenderData,
    trails: TrailHistory,
    config: CourtRenderConfig,
    ball_trail: Optional[deque] = None,
) -> np.ndarray:
    """Output 2: players as dots on the court template, with motion trails.

    Transcribed from ``TopdownRenderer.render``
    (SRC_A/0619/render.py:53-88): same team colour map, same polyline trail,
    same filled circle, same label offset (+8, -8) / scale 0.42, same ball
    colour and radius, same bottom-left timestamp.

    Difference: trail points come from ``TrailHistory`` (EMA-smoothed, salvaged
    from the topdown script) instead of 0619's raw un-smoothed append. The A3
    brief asked for both sources; EMA smoothing is the strictly better of the
    two and is itself salvaged code, not new logic.
    """
    scene = court_template.copy()
    team_colors = {0: (50, 180, 50), 1: (50, 120, 240), None: (210, 210, 210)}

    for player in data.players:
        point = player.court_xy
        if point is None:
            continue
        smoothed = trails.update(player.track_id, point)
        if smoothed is None:
            continue
        px, py = int(round(smoothed[0])), int(round(smoothed[1]))
        color = team_colors.get(player.team_id, team_colors[None])

        pts = np.asarray([[int(x), int(y)] for x, y in trails.get(player.track_id)], dtype=np.int32)
        if len(pts) >= 2:
            cv2.polylines(scene, [pts.reshape(-1, 1, 2)], False, color, config.trail_thickness, cv2.LINE_AA)
        cv2.circle(scene, (px, py), config.player_radius, color, -1, cv2.LINE_AA)

        number = player.jersey_number
        label = f"#{number}" if number else f"id:{player.track_id}"
        cv2.putText(scene, label, (px + 8, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 2, cv2.LINE_AA)

    if data.ball is not None and data.ball.court_xy is not None:
        bx, by = int(round(data.ball.court_xy[0])), int(round(data.ball.court_xy[1]))
        if ball_trail is not None:
            ball_trail.append((bx, by))
            pts = np.asarray([[int(x), int(y)] for x, y in ball_trail], dtype=np.int32)
            if len(pts) >= 2:
                cv2.polylines(scene, [pts.reshape(-1, 1, 2)], False, BALL_COLOR, 2, cv2.LINE_AA)
        cv2.circle(scene, (bx, by), 6, BALL_COLOR, -1, cv2.LINE_AA)

    cv2.putText(
        scene,
        f"frame={data.frame_idx} time={data.timestamp_sec:.2f}s",
        (18, scene.shape[0] - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    return scene


def make_ball_trail(trail_length: int = 30) -> deque:
    """Matches SRC_A/0619/render.py:51 (``deque(maxlen=trail_length)``)."""
    return deque(maxlen=max(1, int(trail_length)))
