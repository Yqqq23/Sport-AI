"""Multi-object tracking via Ultralytics native ``model.track(...)``.

The tracker backend is a CONFIG VALUE. The string "bytetrack" is never
hardcoded as a behavioural default anywhere in this module; it only ever
appears inside the list of selectable backend names.

Accepted coupling: unlike src/detect.py, this module uses the Ultralytics API
directly, because Ultralytics fuses detection and tracking inside
``model.track()`` and exposes no way to feed it external detections. Outputs
are still converted to plain dataclasses before leaving this module.

ReID support is DETECTED, NOT ASSUMED. Verified against the installed package:
``with_reid`` and ``model`` keys exist in botsort.yaml only; bytetrack.yaml has
neither. For the four newer backends the key set is read from the resolved YAML
at runtime, so this module never guesses what a backend it cannot see supports.

PyYAML is used to read/derive tracker configs. It is a hard dependency of
ultralytics, so this adds no new third-party requirement.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Optional, Sequence

from detect import CLASS_BALL, CLASS_JERSEY, CLASS_PLAYER, CLASS_REFEREE, CLASS_RIM

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Selectable backends
# --------------------------------------------------------------------------

#: Backends shipped inside the ultralytics package, with a one-line note each.
#: The four marked "requires ultralytics >= 8.4.63" do not exist in older
#: installs; requesting one there raises a clear upgrade error.
BUNDLED_TRACKERS: dict[str, str] = {
    "botsort.yaml": "default general-purpose tracker, motion plus optional appearance cues",
    "bytetrack.yaml": "lightweight strong baseline, first choice for high frame rates",
    "ocsort.yaml": "observation-centric SORT variant, more robust motion association",
    "deepocsort.yaml": "OC-SORT plus ReID appearance fusion and camera motion compensation",
    "fasttrack.yaml": "occlusion-aware ByteTrack variant with Kalman rollback",
    "tracktrack.yaml": (
        "CVPR 2025, multi-cue iterative association plus Track-Aware Initialization; "
        "best under dense occlusion but the heaviest"
    ),
}

#: The four backends added in ultralytics 8.4.63.
TRACKERS_REQUIRING_NEW_ULTRALYTICS = frozenset(
    {"ocsort.yaml", "deepocsort.yaml", "fasttrack.yaml", "tracktrack.yaml"}
)

MIN_ULTRALYTICS_FOR_ALL_TRACKERS = "8.4.63"


class TrackerNotAvailableError(RuntimeError):
    """Requested tracker config could not be resolved."""


# --------------------------------------------------------------------------
# Framework-agnostic output types
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TrackedObject:
    """One tracked object in one frame.

    ``track_id`` is None when the backend did not assign an id this frame
    (common on the first frames, or for low-confidence detections that the
    tracker has not confirmed yet).
    """

    bbox_xyxy: tuple[float, float, float, float]
    class_id: int
    confidence: float
    track_id: Optional[int] = None

    @property
    def center_xy(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def ground_xy(self) -> tuple[float, float]:
        """Bottom-edge midpoint: foot position, used for court projection."""
        x1, _, x2, y2 = self.bbox_xyxy
        return ((x1 + x2) / 2.0, y2)


@dataclass
class FrameTracks:
    """Everything tracked/detected in one frame, already split by role.

    ``referees`` is RENDER-ONLY and is the single exception to the rule that
    every field here feeds analysis. It exists so referee boxes can be drawn in
    output 1 and visually confirmed to be separated from players. A class whose
    output is never displayed is a class nobody can tell is working, and "the
    detector is quietly filing referees as players" is exactly the failure that
    stays invisible until a referee shows up credited with a basket in output 3.

    NOTHING ELSE MAY CONSUME IT. Referees are excluded from tracking, jersey
    OCR, possession, shot attribution, output 2 and output 3. Those consumers
    all take ``players`` explicitly, so adding this field cannot leak referees
    into them by accident - but do not pass ``referees`` to any of them.
    See Tracker.update and docs/annotation_guide.md.
    """

    frame_idx: int
    timestamp_sec: float
    players: list[TrackedObject] = field(default_factory=list)
    rims: list[TrackedObject] = field(default_factory=list)
    jerseys: list[TrackedObject] = field(default_factory=list)
    #: RENDER-ONLY. Never tracked, never analysed. track_id is always None.
    referees: list[TrackedObject] = field(default_factory=list)
    #: Ball comes from the dedicated nearest-neighbour tracker, never from the
    #: multi-object tracker (see BallTracker).
    ball: Optional[TrackedObject] = None


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


@dataclass
class TrackerConfig:
    """Tracking settings. Populated from configs/default.yaml by src/config.py.

    ``tracker`` accepts either a bundled name from BUNDLED_TRACKERS or a path to
    a custom YAML such as configs/trackers/bytetrack_sportcv.yaml.
    """

    tracker: str = "tracktrack.yaml"

    #: Enable ReID appearance matching. Only takes effect on backends whose
    #: config actually declares ``with_reid`` (verified: botsort does,
    #: bytetrack does not). On a backend without the key this is reported as a
    #: warning and the run continues WITHOUT ReID - never silently ignored.
    #:
    #: UNPROVEN FOR THIS SPORT: basketball team-mates wear near-identical
    #: kits, so appearance embeddings may barely separate them. Treat this as
    #: an experiment to run, not an assumed improvement.
    with_reid: bool = False
    #: "auto" reuses the detector's own features, which is the cheapest option.
    reid_model: str = "auto"

    #: Required for cross-frame id continuity when feeding frames one at a time.
    persist: bool = True

    conf: float = 0.25
    iou: float = 0.45
    imgsz: int = 960
    device: str = "0"
    half: bool = False
    max_det: int = 60

    #: When True, only player boxes are shown to the multi-object tracker and
    #: ball/rim/jersey come from a SECOND forward pass. Strictly isolates the
    #: tracker from non-player classes, at roughly double the inference cost.
    #: When False (default) a single pass tracks everything and non-player
    #: classes are filtered out afterwards - one forward pass, but the backend
    #: sees ball/rim boxes during association.
    #: NOT MEASURED YET on this footage; worth an experiment.
    restrict_tracker_to_players: bool = False

    #: Directory for derived tracker YAMLs written when with_reid is enabled.
    #: Relative to the working directory; never written inside site-packages.
    generated_tracker_dir: Path = Path("outputs/.generated_trackers")

    # --- ball nearest-neighbour continuation ------------------------------
    #: Max centre-to-centre pixel distance to treat a ball detection as the
    #: continuation of the previous one.
    #: NOT A RECOVERED VALUE - needs calibration on real 1080p footage.
    ball_max_distance: float = 120.0
    #: Frames the ball may stay unmatched before its track is considered lost.
    ball_max_lost_frames: int = 15


# --------------------------------------------------------------------------
# Tracker config resolution
# --------------------------------------------------------------------------


def ultralytics_tracker_dir() -> Path:
    """Directory holding the tracker YAMLs bundled with ultralytics."""
    import ultralytics

    return Path(ultralytics.__file__).resolve().parent / "cfg" / "trackers"


def available_bundled_trackers() -> list[str]:
    """Names actually present in the installed ultralytics package."""
    tracker_dir = ultralytics_tracker_dir()
    if not tracker_dir.is_dir():
        return []
    return sorted(p.name for p in tracker_dir.glob("*.yaml"))


def resolve_tracker_path(tracker: str | Path) -> Path:
    """Turn a backend name or path into a concrete YAML path.

    Raises TrackerNotAvailableError with an actionable message, including the
    upgrade hint when a newer backend is requested on an older ultralytics.
    """
    candidate = Path(tracker)
    if candidate.suffix in {".yaml", ".yml"} and candidate.exists():
        return candidate.resolve()

    name = candidate.name
    tracker_dir = ultralytics_tracker_dir()
    bundled = tracker_dir / name
    if bundled.exists():
        return bundled.resolve()

    present = available_bundled_trackers()
    if name in TRACKERS_REQUIRING_NEW_ULTRALYTICS:
        import ultralytics

        raise TrackerNotAvailableError(
            f"Tracker '{name}' is not in this ultralytics install "
            f"(version {getattr(ultralytics, '__version__', 'unknown')}).\n"
            f"It requires ultralytics >= {MIN_ULTRALYTICS_FOR_ALL_TRACKERS}.\n"
            f"Fix:  pip install -U 'ultralytics>={MIN_ULTRALYTICS_FOR_ALL_TRACKERS}'\n"
            f"Bundled trackers currently available: {present}"
        )

    raise TrackerNotAvailableError(
        f"Could not resolve tracker '{tracker}'.\n"
        f"Give either a path to an existing YAML (e.g. configs/trackers/bytetrack_sportcv.yaml)\n"
        f"or one of the bundled names: {sorted(BUNDLED_TRACKERS)}\n"
        f"Bundled trackers currently available in this install: {present}"
    )


def read_tracker_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TrackerNotAvailableError(f"Tracker config is not a mapping: {path}")
    return data


def backend_supports_reid(tracker_path: Path) -> bool:
    """True when the resolved backend config declares a ``with_reid`` key.

    Read from the file rather than assumed, so backends this codebase has never
    seen are judged on their actual contents.
    """
    return "with_reid" in read_tracker_yaml(tracker_path)


def prepare_tracker_config(config: TrackerConfig) -> Path:
    """Resolve the backend and, if ReID is requested and supported, derive a copy.

    Never mutates files inside site-packages: when an override is needed a new
    YAML is written under ``config.generated_tracker_dir``.
    """
    resolved = resolve_tracker_path(config.tracker)

    if not config.with_reid:
        return resolved

    if not backend_supports_reid(resolved):
        LOGGER.warning(
            "with_reid=True was requested but backend '%s' does not declare a "
            "'with_reid' key, so ReID CANNOT be enabled for it. Running WITHOUT ReID. "
            "Verified support: botsort.yaml declares with_reid/model; bytetrack.yaml "
            "does not. Choose a ReID-capable backend (e.g. botsort.yaml or "
            "deepocsort.yaml) or set with_reid=False to silence this.",
            resolved.name,
        )
        return resolved

    import yaml

    data = read_tracker_yaml(resolved)
    data["with_reid"] = True
    data["model"] = config.reid_model

    out_dir = Path(config.generated_tracker_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    derived = out_dir / f"{resolved.stem}_reid.yaml"
    with derived.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)

    LOGGER.info(
        "ReID enabled on '%s' (model=%s). Derived config written to %s",
        resolved.name,
        config.reid_model,
        derived,
    )
    return derived


# --------------------------------------------------------------------------
# Ball: dedicated nearest-neighbour continuation
# --------------------------------------------------------------------------


class BallTracker:
    """Single-object nearest-neighbour continuation for the ball.

    Kept out of the multi-object tracker on purpose: there is only ever one
    ball, it moves far between frames, and it is frequently occluded, all of
    which make a general MOT backend a poor fit.
    """

    def __init__(self, max_distance: float = 120.0, max_lost_frames: int = 15) -> None:
        self.max_distance = float(max_distance)
        self.max_lost_frames = int(max_lost_frames)
        self.previous_center: Optional[tuple[float, float]] = None
        self.lost_frames = 0
        self.track_id = 1

    def update(self, candidates: Sequence[TrackedObject]) -> Optional[TrackedObject]:
        """Pick this frame's ball from candidate ball detections."""
        if not candidates:
            self.lost_frames += 1
            if self.lost_frames > self.max_lost_frames:
                self.previous_center = None
            return None

        if self.previous_center is None:
            chosen = max(candidates, key=lambda d: d.confidence)
        else:
            px, py = self.previous_center

            def distance(obj: TrackedObject) -> float:
                cx, cy = obj.center_xy
                return ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5

            nearest = min(candidates, key=distance)
            if distance(nearest) <= self.max_distance:
                chosen = nearest
            else:
                # Too far to be continuous; fall back to the strongest detection
                # rather than dropping the ball entirely.
                chosen = max(candidates, key=lambda d: d.confidence)

        self.previous_center = chosen.center_xy
        self.lost_frames = 0
        return TrackedObject(
            bbox_xyxy=chosen.bbox_xyxy,
            class_id=CLASS_BALL,
            confidence=chosen.confidence,
            track_id=self.track_id,
        )


# --------------------------------------------------------------------------
# Tracker
# --------------------------------------------------------------------------


def _to_tracked_objects(results: Any) -> list[TrackedObject]:
    """Convert Ultralytics tracking Results into plain dataclasses."""
    out: list[TrackedObject] = []
    if not results:
        return out

    boxes = getattr(results[0], "boxes", None)
    if boxes is None or getattr(boxes, "xyxy", None) is None or len(boxes) == 0:
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
    ids = boxes.id.detach().cpu().numpy() if getattr(boxes, "id", None) is not None else None

    for i in range(len(xyxy)):
        x1, y1, x2, y2 = (float(v) for v in xyxy[i].tolist())
        out.append(
            TrackedObject(
                bbox_xyxy=(x1, y1, x2, y2),
                class_id=int(cls[i]),
                confidence=float(conf[i]),
                track_id=int(ids[i]) if ids is not None else None,
            )
        )
    return out


class Tracker:
    """Wraps ``model.track`` with a configurable backend and a separate ball track."""

    def __init__(self, weights: Path, config: Optional[TrackerConfig] = None) -> None:
        self.config = config or TrackerConfig()
        self.weights = Path(weights)
        if not self.weights.exists():
            raise FileNotFoundError(
                f"Tracking weights not found: {self.weights}\n"
                "Pass an existing .pt. See src/detect.py for why there is no COCO fallback."
            )

        self.tracker_path = prepare_tracker_config(self.config)
        self.backend_name = self.tracker_path.name
        self.reid_active = self.config.with_reid and backend_supports_reid(self.tracker_path)

        from ultralytics import YOLO

        self.model = YOLO(str(self.weights))
        self.ball_tracker = BallTracker(
            max_distance=self.config.ball_max_distance,
            max_lost_frames=self.config.ball_max_lost_frames,
        )
        LOGGER.info(
            "Tracker ready: backend=%s reid=%s weights=%s",
            self.backend_name,
            self.reid_active,
            self.weights.name,
        )

    def _track_call(self, frame: "np.ndarray", classes: Optional[Sequence[int]]) -> list[TrackedObject]:
        results = self.model.track(
            frame,
            persist=self.config.persist,
            tracker=str(self.tracker_path),
            conf=self.config.conf,
            iou=self.config.iou,
            imgsz=self.config.imgsz,
            device=self.config.device,
            half=self.config.half,
            max_det=self.config.max_det,
            classes=list(classes) if classes is not None else None,
            verbose=False,
        )
        return _to_tracked_objects(results)

    def _predict_call(self, frame: "np.ndarray", classes: Sequence[int]) -> list[TrackedObject]:
        results = self.model.predict(
            frame,
            conf=self.config.conf,
            iou=self.config.iou,
            imgsz=self.config.imgsz,
            device=self.config.device,
            half=self.config.half,
            max_det=self.config.max_det,
            classes=list(classes),
            verbose=False,
        )
        return _to_tracked_objects(results)

    def update(self, frame: "np.ndarray", frame_idx: int, timestamp_sec: float) -> FrameTracks:
        """Track one frame and return roles already split out."""
        if self.config.restrict_tracker_to_players:
            tracked = self._track_call(frame, classes=[CLASS_PLAYER])
            # CLASS_REFEREE is DETECTED here but never TRACKED. It is included in
            # this second pass purely so referees remain visible for verification:
            # a class whose output nobody can see is a class nobody can tell is
            # working, and "the detector is quietly calling referees players" is
            # invisible unless referee boxes are actually produced. It costs one
            # extra class in a pass that is already running.
            others = self._predict_call(
                frame, classes=[CLASS_BALL, CLASS_RIM, CLASS_JERSEY, CLASS_REFEREE]
            )
            objects = tracked + others
        else:
            objects = self._track_call(frame, classes=None)

        # Role split by exact class id. CLASS_REFEREE (4) matches none of these
        # four branches, so referees are DROPPED HERE BY DESIGN and never reach
        # tracking, jersey OCR, possession, shot attribution or any output.
        #
        # That exclusion IS the class's job - it is not an oversight and not a
        # TODO. The detector needs a label for "person who is not a player";
        # without one it files referees under `player`, which puts a non-player
        # on the top-down court, into the possession contest, and potentially
        # credited with a basket. Do not add a `referees = [...]` line here
        # expecting a downstream consumer; there is none, on purpose.
        players = [o for o in objects if o.class_id == CLASS_PLAYER]
        rims = [o for o in objects if o.class_id == CLASS_RIM]
        jerseys = [o for o in objects if o.class_id == CLASS_JERSEY]
        ball_candidates = [o for o in objects if o.class_id == CLASS_BALL]
        # RENDER ONLY. Collected so referee boxes can be drawn in output 1 and
        # visually confirmed to be separated from players - see FrameTracks.
        # Referees are excluded from `players` above and from every consumer of
        # it, which is the whole job of the class. Do not feed this list to
        # tracking, jersey OCR, possession, shot attribution, or output 2/3.
        referees = [o for o in objects if o.class_id == CLASS_REFEREE]

        return FrameTracks(
            frame_idx=frame_idx,
            timestamp_sec=timestamp_sec,
            players=players,
            rims=rims,
            jerseys=jerseys,
            referees=referees,
            ball=self.ball_tracker.update(ball_candidates),
        )

    def reset(self) -> None:
        """Drop tracker state between videos."""
        predictor = getattr(self.model, "predictor", None)
        trackers = getattr(predictor, "trackers", None) if predictor is not None else None
        if trackers:
            for backend in trackers:
                if hasattr(backend, "reset"):
                    backend.reset()
        self.ball_tracker = BallTracker(
            max_distance=self.config.ball_max_distance,
            max_lost_frames=self.config.ball_max_lost_frames,
        )


# --------------------------------------------------------------------------
# Whole-video entry point (reused by scripts/compare_trackers.py)
# --------------------------------------------------------------------------


@dataclass
class VideoInfo:
    fps: float
    width: int
    height: int
    frame_count: int

    @property
    def duration_sec(self) -> float:
        return float(self.frame_count) / self.fps if self.fps > 0 else 0.0


def probe_video(video: Path) -> VideoInfo:
    import cv2

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    try:
        return VideoInfo(
            fps=float(capture.get(cv2.CAP_PROP_FPS) or 30.0),
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
            frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        )
    finally:
        capture.release()


def track_video(
    video: str | Path,
    tracker_name: str,
    weights: str | Path,
    config: Optional[TrackerConfig] = None,
    max_frames: Optional[int] = None,
    frame_callback: Optional[Any] = None,
) -> Iterator[FrameTracks]:
    """Track a whole video with a named backend, yielding one FrameTracks per frame.

    This is THE tracking entry point for external scripts (notably
    scripts/compare_trackers.py). Callers must not reimplement tracking.

    Args:
        video: input video path
        tracker_name: bundled backend name or path to a custom YAML; overrides
            ``config.tracker`` so callers can sweep backends over one config
        weights: detector/tracker weights
        config: base settings; a copy is used so the caller's object is untouched
        max_frames: stop early (smoke tests)
        frame_callback: optional ``fn(frame_bgr, FrameTracks)`` invoked per frame,
            used by compare_trackers to write an annotated video without a
            second decode pass
    """
    import copy

    import cv2

    video_path = Path(video)
    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    run_config = copy.deepcopy(config) if config is not None else TrackerConfig()
    run_config.tracker = tracker_name

    tracker = Tracker(weights=Path(weights), config=run_config)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)

    frame_idx = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if max_frames is not None and frame_idx >= max_frames:
                break
            frame_tracks = tracker.update(
                frame, frame_idx=frame_idx, timestamp_sec=frame_idx / max(fps, 1e-6)
            )
            if frame_callback is not None:
                frame_callback(frame, frame_tracks)
            yield frame_tracks
            frame_idx += 1
    finally:
        capture.release()


# --------------------------------------------------------------------------
# Summary statistics (used by scripts/compare_trackers.py)
# --------------------------------------------------------------------------


@dataclass
class TrackingSummary:
    """Backend comparison metrics.

    ``unique_track_ids`` is the headline number: the on-court player count is
    fixed, so a total far above it means the backend is churning identities.
    """

    tracker_name: str
    frames_processed: int
    unique_track_ids: int
    mean_track_lifetime_frames: float
    median_track_lifetime_frames: float
    mean_players_per_frame: float
    frames_with_ball: int
    elapsed_sec: float
    video_duration_sec: float

    @property
    def realtime_ratio(self) -> float:
        """Processing time divided by video duration. <= 1.0 is real time."""
        if self.video_duration_sec <= 0:
            return float("inf")
        return self.elapsed_sec / self.video_duration_sec


def summarize_tracks(
    frames: Sequence[FrameTracks],
    tracker_name: str,
    elapsed_sec: float,
    video_duration_sec: float,
) -> TrackingSummary:
    """Reduce per-frame results to comparison metrics."""
    lifetimes: dict[int, int] = {}
    player_counts: list[int] = []
    frames_with_ball = 0

    for frame in frames:
        player_counts.append(len(frame.players))
        if frame.ball is not None:
            frames_with_ball += 1
        for player in frame.players:
            if player.track_id is not None:
                lifetimes[player.track_id] = lifetimes.get(player.track_id, 0) + 1

    values = sorted(lifetimes.values())
    if values:
        mean_lifetime = sum(values) / len(values)
        mid = len(values) // 2
        median_lifetime = (
            float(values[mid])
            if len(values) % 2 == 1
            else (values[mid - 1] + values[mid]) / 2.0
        )
    else:
        mean_lifetime = 0.0
        median_lifetime = 0.0

    return TrackingSummary(
        tracker_name=tracker_name,
        frames_processed=len(frames),
        unique_track_ids=len(lifetimes),
        mean_track_lifetime_frames=float(mean_lifetime),
        median_track_lifetime_frames=float(median_lifetime),
        mean_players_per_frame=(
            sum(player_counts) / len(player_counts) if player_counts else 0.0
        ),
        frames_with_ball=frames_with_ball,
        elapsed_sec=float(elapsed_sec),
        video_duration_sec=float(video_duration_sec),
    )


def run_tracker_benchmark(
    video: str | Path,
    tracker_name: str,
    weights: str | Path,
    config: Optional[TrackerConfig] = None,
    max_frames: Optional[int] = None,
    frame_callback: Optional[Any] = None,
) -> tuple[list[FrameTracks], TrackingSummary]:
    """Track a video end to end and return both the frames and the summary."""
    info = probe_video(Path(video))
    effective_frames = (
        info.frame_count if max_frames is None else min(max_frames, info.frame_count or max_frames)
    )
    duration = effective_frames / info.fps if info.fps > 0 else 0.0

    started = time.perf_counter()
    frames = list(
        track_video(
            video=video,
            tracker_name=tracker_name,
            weights=weights,
            config=config,
            max_frames=max_frames,
            frame_callback=frame_callback,
        )
    )
    elapsed = time.perf_counter() - started

    return frames, summarize_tracks(
        frames=frames,
        tracker_name=tracker_name,
        elapsed_sec=elapsed,
        video_duration_sec=duration,
    )
