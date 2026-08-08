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
from jersey import JerseyStabilizerConfig, OCRGateConfig
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
    backend: str = "pytesseract"
    #: Optional explicit path to the tesseract binary.
    tesseract_cmd: Optional[Path] = None
    stabilizer: JerseyStabilizerConfig = field(default_factory=JerseyStabilizerConfig)
    gate: OCRGateConfig = field(default_factory=OCRGateConfig)


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
