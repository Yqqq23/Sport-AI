"""Configuration loading: YAML -> typed dataclasses.

Single source of truth for every threshold, path and switch. Modules never read
YAML themselves; the pipeline builds their config objects from here.

Path handling: every path in the YAML is resolved RELATIVE TO THE CONFIG FILE,
not the process working directory. That keeps configs portable between the
Windows editing machine and the Linux GPU server without absolute paths.

PyYAML is a hard dependency of ultralytics, so this adds no new requirement.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional, TypeVar, get_type_hints

from calib import CourtCanvasConfig
from detect import DetectorConfig
from identity import IdentityConfig
from jersey import (
    JerseyStabilizerConfig,
    LocalClassifierConfig,
    OCRGateConfig,
    PARSeqConfig,
    PerDigitClassifierConfig,
    RoboflowConfig,
)
from render import CourtRenderConfig
from shot import ShotConfig
from track import TrackerConfig

LOGGER = logging.getLogger(__name__)

T = TypeVar("T")

DEFAULT_CONFIG_PATH = Path("configs/default.yaml")

#: Keys whose values are paths and must be resolved against the config file.
_PATH_KEYS = {
    "weights",
    "calibration_json",
    "output_dir",
    "generated_tracker_dir",
    "tesseract_cmd",
    "debug_crop_dir",
}


@dataclass
class OutputConfig:
    """Output 1 / 2 / 3 destinations and toggles."""

    output_dir: Path = Path("outputs/run")
    annotated_video_name: str = "annotated.mp4"
    topdown_video_name: str = "topdown.mp4"
    shots_stem: str = "shots"
    write_annotated: bool = True
    write_topdown: bool = True
    #: mp4v is available in stock OpenCV builds. avc1/H.264 usually is not
    #: unless OpenCV was built against libx264.
    fourcc: str = "mp4v"


@dataclass
class JerseyPipelineConfig:
    """How often jersey OCR runs, plus the stabilizer and gate settings."""

    enabled: bool = True
    #: Run OCR every N frames. Salvaged default from the source pipeline
    #: (--number_interval, default 5).
    ocr_interval: int = 5
    #: "auto" | "parseq" | "local" | "roboflow" | "paddleocr" | "easyocr"
    #: | "pytesseract".
    #: Default is "auto": the locally-trained classifier when its weights exist,
    #: then PARSeq (no training needed), then paddleocr/easyocr, and only
    #: pytesseract with a warning. The previous hard default of "pytesseract"
    #: silently pinned the pipeline to the weakest recognizer even on machines
    #: with a good one available.
    backend: str = "auto"
    #: Optional explicit path to the tesseract binary.
    tesseract_cmd: Optional[Path] = None
    #: Run the OCR backend on GPU where the backend supports it.
    use_gpu: bool = True
    #: How far a jersey box centre may sit OUTSIDE a player box and still be
    #: attributed to it, as a fraction of that player's box height. The old
    #: code required the centre to be strictly inside (distance == 0.0), so a
    #: jersey box a few pixels proud of a tight player box was dropped without
    #: even being counted as a skip.
    owner_max_distance_ratio: float = 0.25
    #: When the detector produces NO jersey boxes, derive the number region
    #: geometrically from each player box instead of doing nothing.
    #:
    #: Weights trained without a `jersey` class otherwise make the whole OCR
    #: loop a no-op - no crops, no calls, no skips - and nothing in the summary
    #: explains the silence. This keeps numbers working on such weights, at
    #: lower accuracy than a real jersey detection.
    fallback_to_torso: bool = True
    #: Torso band as fractions of the player box: vertical span and the
    #: fraction of width kept around the centre. Defaults put the band between
    #: 15% and 55% of height (below the head, above the waist), middle 60% wide.
    torso_top: float = 0.15
    torso_bottom: float = 0.55
    torso_width: float = 0.60
    #: Directory to dump the exact crops handed to OCR, for eyeballing what the
    #: recognizer actually sees. None disables. This is the first thing to turn
    #: on when numbers are not appearing - it separates "crop is unreadable" from
    #: "recognizer is failing on a readable crop", which no metric can.
    debug_crop_dir: Optional[Path] = None
    #: Stop calling OCR for a track once its number has LOCKED. Essential with
    #: a network backend (Roboflow serverless measures 2.6-6.8 s per crop, so a
    #: full game is otherwise days of wall clock) and a free win on every other
    #: backend. Turn it off only when deliberately measuring raw OCR accuracy.
    skip_locked_tracks: bool = True
    #: Hard ceiling on OCR calls per run. 0 = unlimited. A safety net for
    #: metered backends: it bounds the bill of a misconfigured overnight run.
    max_ocr_calls: int = 0
    stabilizer: JerseyStabilizerConfig = field(default_factory=JerseyStabilizerConfig)
    gate: OCRGateConfig = field(default_factory=OCRGateConfig)
    roboflow: RoboflowConfig = field(default_factory=RoboflowConfig)
    local: LocalClassifierConfig = field(default_factory=LocalClassifierConfig)
    parseq: PARSeqConfig = field(default_factory=PARSeqConfig)
    perdigit: PerDigitClassifierConfig = field(default_factory=PerDigitClassifierConfig)


@dataclass
class RenderPipelineConfig:
    court: CourtRenderConfig = field(default_factory=CourtRenderConfig)
    line_width: int = 2
    #: EMA smoothing factor for top-down trails. NOT a recovered value.
    trail_alpha: float = 0.5


@dataclass
class AppConfig:
    """Everything the pipeline needs."""

    detector: DetectorConfig
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    shot: ShotConfig = field(default_factory=ShotConfig)
    jersey: JerseyPipelineConfig = field(default_factory=JerseyPipelineConfig)
    identity: IdentityConfig = field(default_factory=IdentityConfig)
    render: RenderPipelineConfig = field(default_factory=RenderPipelineConfig)
    canvas: CourtCanvasConfig = field(default_factory=CourtCanvasConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    calibration_json: Optional[Path] = None
    max_frames: Optional[int] = None

    def validate(self) -> list[str]:
        """Return human-readable problems; empty list means usable."""
        problems: list[str] = []
        if not Path(self.detector.weights).exists():
            problems.append(f"detector.weights does not exist: {self.detector.weights}")
        if self.calibration_json is not None and not Path(self.calibration_json).exists():
            problems.append(f"calibration_json does not exist: {self.calibration_json}")
        if self.canvas.width != self.render.court.width or self.canvas.height != self.render.court.height:
            problems.append(
                "canvas size and render.court size disagree "
                f"({self.canvas.width}x{self.canvas.height} vs "
                f"{self.render.court.width}x{self.render.court.height}). "
                "Projected points would land in a different coordinate space than the "
                "drawn court."
            )
        return problems


# --------------------------------------------------------------------------
# YAML -> dataclass
# --------------------------------------------------------------------------


def _resolve_path(value: Any, base_dir: Path) -> Path:
    """Resolve a path from the YAML against the config file's directory."""
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (base_dir / path)


def _build(cls: type[T], data: Optional[dict[str, Any]], base_dir: Path) -> T:
    """Instantiate a dataclass from a mapping, recursing into nested dataclasses.

    Unknown keys are an error rather than being ignored: a typo in a threshold
    name would otherwise silently leave the default in place, which is exactly
    the class of bug that is impossible to notice in output.
    """
    if data is None:
        return cls()  # type: ignore[call-arg]
    if not isinstance(data, dict):
        raise TypeError(f"Expected a mapping for {cls.__name__}, got {type(data).__name__}")

    known = {f.name: f for f in fields(cls)}  # type: ignore[arg-type]
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(
            f"Unknown key(s) for {cls.__name__}: {sorted(unknown)}. "
            f"Valid keys: {sorted(known)}"
        )

    # This module uses `from __future__ import annotations`, so Field.type is a
    # STRING, not the class. Resolving hints is what makes nested dataclass
    # detection below actually fire.
    hints = get_type_hints(cls)

    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        field_type = hints.get(name)
        if is_dataclass(field_type) and isinstance(value, dict):
            kwargs[name] = _build(field_type, value, base_dir)  # type: ignore[arg-type]
        elif name in _PATH_KEYS and value is not None:
            kwargs[name] = _resolve_path(value, base_dir)
        else:
            kwargs[name] = value
    return cls(**kwargs)  # type: ignore[call-arg]


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    """Load and type-check configs/default.yaml (or another config file)."""
    import yaml

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Copy configs/default.yaml and pass it with --config."
        )

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")

    base_dir = config_path.resolve().parent

    detector_data = raw.get("detector")
    if not detector_data or "weights" not in detector_data:
        raise ValueError(
            f"{config_path}: 'detector.weights' is required. "
            "There is no default weights path, because a wrong-but-present default "
            "would silently produce meaningless detections."
        )

    unknown_top = set(raw) - {f.name for f in fields(AppConfig)}
    if unknown_top:
        raise ValueError(
            f"Unknown top-level config key(s): {sorted(unknown_top)}. "
            f"Valid keys: {sorted(f.name for f in fields(AppConfig))}"
        )

    config = AppConfig(
        detector=_build(DetectorConfig, detector_data, base_dir),
        tracker=_build(TrackerConfig, raw.get("tracker"), base_dir),
        shot=_build(ShotConfig, raw.get("shot"), base_dir),
        jersey=_build(JerseyPipelineConfig, raw.get("jersey"), base_dir),
        identity=_build(IdentityConfig, raw.get("identity"), base_dir),
        render=_build(RenderPipelineConfig, raw.get("render"), base_dir),
        canvas=_build(CourtCanvasConfig, raw.get("canvas"), base_dir),
        output=_build(OutputConfig, raw.get("output"), base_dir),
        calibration_json=(
            _resolve_path(raw["calibration_json"], base_dir)
            if raw.get("calibration_json")
            else None
        ),
        max_frames=raw.get("max_frames"),
    )

    LOGGER.info("Loaded config from %s", config_path)
    return config
