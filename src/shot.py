"""Made-basket detection and shooter attribution.

NOTHING IN THIS MODULE IS SALVAGED. No reference implementation existed in
either source tree, so every rule here is newly written and unvalidated. All
thresholds live in ShotConfig; none are tuned. Read the docstrings for the
known failure modes before trusting any output.

Two independent problems:

1. Made-basket detection (BasketDetector)
   The ball must be seen ABOVE the rim, then INSIDE the rim's horizontal span,
   then BELOW the rim, in that order, within a bounded time window. Downward
   motion is required, which is what separates a shot dropping through from a
   ball travelling upward past the rim.

2. Shooter attribution (PossessionTracker)
   Independently of shots, track who is holding the ball each frame. When a
   basket is confirmed, walk backwards to the last player who held possession
   for at least ``min_possession_frames`` before release.

Streaming design: both components consume one frame at a time and keep only a
bounded deque (``buffer_seconds``). The video is never read in full, so this
works on a live feed.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Sequence

LOGGER = logging.getLogger(__name__)

BBox = tuple[float, float, float, float]


# --------------------------------------------------------------------------
# Configuration - every value below is a GUESS awaiting calibration
# --------------------------------------------------------------------------


@dataclass
class ShotConfig:
    """Thresholds for basket detection and shooter attribution.

    WARNING: NONE OF THESE ARE TUNED VALUES. There was no prior implementation
    to recover constants from. They are first guesses for 1080p / 30fps footage
    and MUST be calibrated on real video before any number here is trusted.
    """

    # --- streaming buffer ------------------------------------------------
    #: Seconds of history retained for both the basket state machine and
    #: possession lookback. Must exceed the longest plausible
    #: shot-release-to-swish flight time.
    buffer_seconds: float = 2.0

    # --- rim geometry ----------------------------------------------------
    #: Horizontal tolerance when testing "inside the rim span", as a fraction
    #: of rim box width. 0.0 means strictly inside the detected box.
    rim_horizontal_margin: float = 0.15
    #: How far above the rim (in rim-box heights) the ball must be seen for the
    #: ABOVE stage to arm. Prevents arming on a ball resting on the rim.
    rim_above_margin: float = 0.5
    #: How far below the rim (in rim-box heights) the ball must fall for the
    #: BELOW stage to confirm. Larger values reject balls that graze the front
    #: edge and bounce away.
    rim_below_margin: float = 0.5
    #: Max frames allowed between the ABOVE observation and the BELOW
    #: confirmation. At 30fps, 45 frames is 1.5s.
    max_frames_above_to_below: int = 45
    #: Ball centre must be descending by at least this many pixels per frame
    #: during the crossing. Rejects upward passes through the rim plane.
    min_downward_velocity: float = 1.0

    #: Frames to suppress further basket events after one fires, so a single
    #: make cannot be counted repeatedly while the ball bounces on the floor.
    cooldown_frames: int = 45

    #: Rim detections are noisy frame to frame. Rather than trusting one frame,
    #: keep a rolling median over this many frames.
    rim_smoothing_frames: int = 15

    # --- possession / shooter attribution --------------------------------
    #: Max distance from ball centre to a player box (pixels, 0 = inside box)
    #: for that player to count as holding the ball.
    possession_max_distance: float = 80.0
    #: A player must hold possession for at least this many consecutive frames
    #: before they can be credited as the shooter. Filters out the ball merely
    #: passing near a defender.
    min_possession_frames: int = 5
    #: How far back (frames) to search for the shooter after a basket.
    #: Must be <= buffer_seconds * fps or the history will already be gone.
    shooter_lookback_frames: int = 60


# --------------------------------------------------------------------------
# Data types
# --------------------------------------------------------------------------


class CrossingStage(Enum):
    """Where the ball is in the rim-crossing sequence."""

    IDLE = "idle"
    ABOVE = "above"
    INSIDE = "inside"


@dataclass(frozen=True)
class BallSample:
    frame_idx: int
    timestamp_sec: float
    center_xy: tuple[float, float]


@dataclass(frozen=True)
class PossessionSample:
    frame_idx: int
    timestamp_sec: float
    track_id: Optional[int]
    #: Foot position of the holder, used as the shot location.
    ground_xy: Optional[tuple[float, float]]


@dataclass
class ShotEvent:
    """A confirmed made basket, with shooter attribution when available."""

    frame_idx: int
    timestamp_sec: float
    made: bool = True
    #: None when no player held possession long enough before the shot.
    shooter_track_id: Optional[int] = None
    #: Filled in by the pipeline from src/jersey.py once the number is stable.
    shooter_jersey_number: Optional[str] = None
    #: Image-space foot position of the shooter at release.
    shot_position_xy: Optional[tuple[float, float]] = None
    #: Court-space position; filled in by the pipeline via src/calib.py.
    court_xy: Optional[tuple[float, float]] = None
    #: Frame at which possession was last held before release.
    release_frame_idx: Optional[int] = None


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------


def point_to_bbox_distance(point: tuple[float, float], bbox: BBox) -> float:
    """Euclidean distance from a point to a box. 0.0 when inside."""
    px, py = point
    x1, y1, x2, y2 = bbox
    dx = max(x1 - px, 0.0, px - x2)
    dy = max(y1 - py, 0.0, py - y2)
    return (dx * dx + dy * dy) ** 0.5


def median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


class RimSmoother:
    """Rolling median of the rim box.

    A single frame's rim detection jitters by several pixels, and the whole
    made/miss decision is measured against this box, so an unsmoothed rim makes
    the margins meaningless.
    """

    def __init__(self, window_frames: int = 15) -> None:
        self.history: deque[BBox] = deque(maxlen=max(1, int(window_frames)))

    def update(self, rim_bbox: Optional[BBox]) -> Optional[BBox]:
        if rim_bbox is not None:
            self.history.append(tuple(float(v) for v in rim_bbox))  # type: ignore[arg-type]
        if not self.history:
            return None
        return (
            median([b[0] for b in self.history]),
            median([b[1] for b in self.history]),
            median([b[2] for b in self.history]),
            median([b[3] for b in self.history]),
        )

    @property
    def has_rim(self) -> bool:
        return bool(self.history)


# --------------------------------------------------------------------------
# 1. Made-basket detection
# --------------------------------------------------------------------------


class BasketDetector:
    """Streaming above -> inside -> below state machine over the rim box.

    KNOWN FAILURE MODES (all unmitigated, listed so they are not a surprise):

    * Ball passing BEHIND the rim from the camera's viewpoint traces the same
      2D path as a ball dropping through it. With a single fixed side camera
      there is no depth cue, so this is a false positive that 2D geometry
      cannot fix. ``rim_below_margin`` only reduces it.
    * A shot that rattles the rim can oscillate across the boundary and fire
      early or twice; ``cooldown_frames`` covers the repeat case only.
    * An airball dropping straight down outside the rim span is rejected by the
      INSIDE stage, but a ball grazing the front edge may still register.
    * If the ball detector drops frames during the fast descent, the BELOW
      stage may never be observed and a real make is missed. This is expected
      to be the dominant error source until the ball class is properly trained.
    * Multiple balls in frame (warmup shots on an adjacent hoop) are not
      disambiguated here; src/track.py yields one ball per frame by
      nearest-neighbour, which may latch onto the wrong one.
    """

    def __init__(self, config: Optional[ShotConfig] = None, fps: float = 30.0) -> None:
        self.config = config or ShotConfig()
        self.fps = float(fps)
        self.rim_smoother = RimSmoother(self.config.rim_smoothing_frames)

        buffer_len = max(1, int(self.config.buffer_seconds * self.fps))
        self.ball_history: deque[BallSample] = deque(maxlen=buffer_len)

        self.stage = CrossingStage.IDLE
        self.stage_frame_idx: Optional[int] = None
        self.cooldown_remaining = 0

    def _reset_stage(self) -> None:
        self.stage = CrossingStage.IDLE
        self.stage_frame_idx = None

    def _downward_velocity(self) -> float:
        """Pixels per frame the ball centre is descending. Negative = rising."""
        if len(self.ball_history) < 2:
            return 0.0
        recent = list(self.ball_history)[-2:]
        dy = recent[1].center_xy[1] - recent[0].center_xy[1]
        df = max(1, recent[1].frame_idx - recent[0].frame_idx)
        return dy / df

    def update(
        self,
        frame_idx: int,
        timestamp_sec: float,
        ball_center: Optional[tuple[float, float]],
        rim_bbox: Optional[BBox],
    ) -> Optional[ShotEvent]:
        """Feed one frame. Returns a ShotEvent on the frame a make is confirmed."""
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1

        rim = self.rim_smoother.update(rim_bbox)
        if rim is None or ball_center is None:
            return None

        self.ball_history.append(
            BallSample(frame_idx=frame_idx, timestamp_sec=timestamp_sec, center_xy=ball_center)
        )

        cfg = self.config
        rx1, ry1, rx2, ry2 = rim
        rim_height = max(1.0, ry2 - ry1)
        rim_width = max(1.0, rx2 - rx1)

        margin_x = cfg.rim_horizontal_margin * rim_width
        inside_span = (rx1 - margin_x) <= ball_center[0] <= (rx2 + margin_x)
        above_line = ry1 - cfg.rim_above_margin * rim_height
        below_line = ry2 + cfg.rim_below_margin * rim_height

        ball_y = ball_center[1]
        is_above = ball_y < above_line
        is_below = ball_y > below_line

        # Expire a stale sequence.
        if self.stage is not CrossingStage.IDLE and self.stage_frame_idx is not None:
            if frame_idx - self.stage_frame_idx > cfg.max_frames_above_to_below:
                self._reset_stage()

        if self.stage is CrossingStage.IDLE:
            if is_above and inside_span:
                # Note: while the ball simply stays above the rim, the timeout
                # above resets to IDLE and this immediately re-arms with a fresh
                # stage_frame_idx. That is intended - a ball hovering above the
                # rim before dropping through is still a make, so the timeout
                # must measure the descent, not how long the ball loitered.
                # The timeout therefore effectively guards the INSIDE stage,
                # where a ball rattling on the rim should eventually be given up on.
                self.stage = CrossingStage.ABOVE
                self.stage_frame_idx = frame_idx
            return None

        if self.stage is CrossingStage.ABOVE:
            if is_below:
                # Skipped the inside observation entirely: at 30fps a fast ball
                # can jump the whole rim between frames. Accept only when it is
                # in the span and clearly descending.
                if inside_span and self._downward_velocity() >= cfg.min_downward_velocity:
                    return self._fire(frame_idx, timestamp_sec)
                self._reset_stage()
                return None
            if inside_span and not is_above:
                self.stage = CrossingStage.INSIDE
                self.stage_frame_idx = frame_idx
            elif not inside_span:
                self._reset_stage()
            return None

        if self.stage is CrossingStage.INSIDE:
            if is_below and inside_span:
                if self._downward_velocity() >= cfg.min_downward_velocity:
                    return self._fire(frame_idx, timestamp_sec)
                self._reset_stage()
                return None
            if is_above:
                # Bounced back up out of the rim: not a make.
                self._reset_stage()
            elif not inside_span:
                self._reset_stage()
            return None

        return None

    def _fire(self, frame_idx: int, timestamp_sec: float) -> Optional[ShotEvent]:
        self._reset_stage()
        if self.cooldown_remaining > 0:
            return None
        self.cooldown_remaining = self.config.cooldown_frames
        LOGGER.info("Made basket detected at frame=%s t=%.2fs", frame_idx, timestamp_sec)
        return ShotEvent(frame_idx=frame_idx, timestamp_sec=timestamp_sec, made=True)


# --------------------------------------------------------------------------
# 2. Shooter attribution
# --------------------------------------------------------------------------


@dataclass
class _PlayerBox:
    track_id: Optional[int]
    bbox_xyxy: BBox
    ground_xy: tuple[float, float]


class PossessionTracker:
    """Records who holds the ball each frame, then answers "who shot it?".

    Possession rule: the player whose box is nearest the ball centre, provided
    that distance is within ``possession_max_distance``. Ties are broken by
    smaller distance only.

    KNOWN FAILURE MODES:

    * During a contested rebound or a screen, two players overlap the ball and
      the nearest-box rule flickers between them. ``min_possession_frames``
      suppresses short flickers but a genuinely contested ball may be credited
      to the wrong player.
    * The rule is 2D. A defender standing between camera and shooter can be
      closer in image space than the actual shooter.
    * After release the ball is nearest to nobody, which is what makes the
      lookback work; but on a quick catch-and-shoot the passer may hold the
      longest run in the window and steal the credit. Only the LAST qualifying
      possession run before the basket is used, which mitigates this.
    * If the shooter's track id switches mid-possession (a tracker ID switch),
      the run is split and may fall below ``min_possession_frames``, yielding
      no attribution rather than a wrong one.
    """

    def __init__(self, config: Optional[ShotConfig] = None, fps: float = 30.0) -> None:
        self.config = config or ShotConfig()
        self.fps = float(fps)
        buffer_len = max(
            int(self.config.buffer_seconds * self.fps),
            int(self.config.shooter_lookback_frames),
            1,
        )
        self.history: deque[PossessionSample] = deque(maxlen=buffer_len)

    def update(
        self,
        frame_idx: int,
        timestamp_sec: float,
        ball_center: Optional[tuple[float, float]],
        players: Sequence[_PlayerBox],
    ) -> Optional[int]:
        """Record possession for one frame. Returns the holder's track id or None."""
        holder: Optional[_PlayerBox] = None

        if ball_center is not None and players:
            best = min(
                players,
                key=lambda p: point_to_bbox_distance(ball_center, p.bbox_xyxy),
            )
            if (
                point_to_bbox_distance(ball_center, best.bbox_xyxy)
                <= self.config.possession_max_distance
            ):
                holder = best

        self.history.append(
            PossessionSample(
                frame_idx=frame_idx,
                timestamp_sec=timestamp_sec,
                track_id=holder.track_id if holder is not None else None,
                ground_xy=holder.ground_xy if holder is not None else None,
            )
        )
        return holder.track_id if holder is not None else None

    def find_shooter(
        self, basket_frame_idx: int
    ) -> tuple[Optional[int], Optional[tuple[float, float]], Optional[int]]:
        """Walk backwards from a basket to the last sustained ball holder.

        Returns (track_id, ground_xy_at_release, release_frame_idx). All None
        when nobody held the ball long enough inside the lookback window.
        """
        cfg = self.config
        window = [
            sample
            for sample in self.history
            if basket_frame_idx - cfg.shooter_lookback_frames
            <= sample.frame_idx
            <= basket_frame_idx
        ]
        if not window:
            return None, None, None

        # Walk backwards, grouping consecutive frames with the same holder.
        # The first group (i.e. latest in time) that is long enough wins.
        run_track_id: Optional[int] = None
        run_length = 0
        run_last_sample: Optional[PossessionSample] = None

        for sample in reversed(window):
            if sample.track_id is None:
                if run_track_id is not None and run_length >= cfg.min_possession_frames:
                    break
                run_track_id = None
                run_length = 0
                run_last_sample = None
                continue

            if sample.track_id == run_track_id:
                run_length += 1
            else:
                if run_track_id is not None and run_length >= cfg.min_possession_frames:
                    break
                # Walking backwards, so the FIRST frame seen for a new holder is
                # the LAST frame they held the ball: their release frame.
                run_track_id = sample.track_id
                run_length = 1
                run_last_sample = sample

        if run_track_id is None or run_length < cfg.min_possession_frames:
            return None, None, None

        ground_xy = run_last_sample.ground_xy if run_last_sample is not None else None
        release_frame = run_last_sample.frame_idx if run_last_sample is not None else None
        return run_track_id, ground_xy, release_frame


# --------------------------------------------------------------------------
# Combined streaming facade
# --------------------------------------------------------------------------


class ShotAnalyzer:
    """Drives BasketDetector and PossessionTracker together, one frame at a time.

    This is what src/pipeline.py calls. It returns fully attributed ShotEvents;
    the pipeline is still responsible for filling ``shooter_jersey_number``
    (from src/jersey.py) and ``court_xy`` (from src/calib.py), because this
    module deliberately knows nothing about OCR or homography.
    """

    def __init__(self, config: Optional[ShotConfig] = None, fps: float = 30.0) -> None:
        self.config = config or ShotConfig()
        self.fps = float(fps)
        self.basket_detector = BasketDetector(self.config, fps=self.fps)
        self.possession_tracker = PossessionTracker(self.config, fps=self.fps)
        self.events: list[ShotEvent] = []

    def update(
        self,
        frame_idx: int,
        timestamp_sec: float,
        ball_center: Optional[tuple[float, float]],
        rim_bbox: Optional[BBox],
        players: Sequence[_PlayerBox],
    ) -> Optional[ShotEvent]:
        """Feed one frame; returns a ShotEvent on the frame a basket is confirmed."""
        self.possession_tracker.update(
            frame_idx=frame_idx,
            timestamp_sec=timestamp_sec,
            ball_center=ball_center,
            players=players,
        )

        event = self.basket_detector.update(
            frame_idx=frame_idx,
            timestamp_sec=timestamp_sec,
            ball_center=ball_center,
            rim_bbox=rim_bbox,
        )
        if event is None:
            return None

        track_id, ground_xy, release_frame = self.possession_tracker.find_shooter(frame_idx)
        event.shooter_track_id = track_id
        event.shot_position_xy = ground_xy
        event.release_frame_idx = release_frame

        if track_id is None:
            LOGGER.warning(
                "Basket at frame=%s has no attributable shooter "
                "(nobody held the ball for >= %s frames within the %s-frame lookback)",
                frame_idx,
                self.config.min_possession_frames,
                self.config.shooter_lookback_frames,
            )

        self.events.append(event)
        return event


def make_player_box(
    track_id: Optional[int], bbox_xyxy: BBox, ground_xy: Optional[tuple[float, float]] = None
) -> _PlayerBox:
    """Build the player record this module expects.

    Keeps src/track.py's TrackedObject out of this module's signature so shot
    logic stays independent of the tracking backend.
    """
    x1, _, x2, y2 = bbox_xyxy
    return _PlayerBox(
        track_id=track_id,
        bbox_xyxy=bbox_xyxy,
        ground_xy=ground_xy if ground_xy is not None else ((x1 + x2) / 2.0, y2),
    )
