"""The bridge between this annotation tool and the basketball-v2 project.

This module is the ONLY place the tool learns what the classes are and how a
jersey number is normalized. Everything else imports from here.

Two things are deliberately NOT redefined locally:

1. The class scheme is READ AT IMPORT TIME from ``train/basketball.yaml``, which
   that project declares as its single source of truth. A hand-maintained copy
   here would drift the moment someone appends a sixth class, and the drift
   would be silent: the exporter would happily write labels under indices the
   detector maps to something else. Reading the file makes drift impossible.

2. ``clean_jersey_number`` is IMPORTED from ``src/jersey.py``, the exact
   function the inference pipeline votes with. Reimplementing it here would give
   ground truth and predictions two subtly different notions of what "01" means,
   and every OCR accuracy number measured against this dataset would be wrong in
   a direction nobody could see.

The scheme is additionally cross-checked against ``src/detect.py`` at import.
Those two files are supposed to mirror each other; if they ever disagree, this
tool refuses to start rather than pick a winner.

Layout assumption: this file lives at
``<repo>/tools/annotation/backend/app/services/`` and the repo root is located
by walking up until both marker files are found. No absolute paths anywhere -
the tool moves with the repo.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any, Callable

LOGGER = logging.getLogger(__name__)

#: Files that identify the basketball-v2 repository root.
_MARKERS = (Path("train") / "basketball.yaml", Path("src") / "jersey.py")


class BasketballIntegrationError(RuntimeError):
    """The tool could not bind to the basketball-v2 project.

    Always fatal at import time. A tool that cannot see the class scheme would
    otherwise fall back to its own idea of the classes and export a dataset that
    looks fine and trains the wrong thing.
    """


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if all((candidate / marker).is_file() for marker in _MARKERS):
            return candidate
    raise BasketballIntegrationError(
        "Could not locate the basketball-v2 repository root above "
        f"{here}.\nExpected to find both "
        f"'{_MARKERS[0].as_posix()}' and '{_MARKERS[1].as_posix()}' in some parent "
        "directory.\nThis tool is designed to live at "
        "<repo>/tools/annotation/. If you moved it out of the repository, move it "
        "back: it cannot invent the class scheme on its own."
    )


REPO_ROOT: Path = _find_repo_root()
BASKETBALL_YAML: Path = REPO_ROOT / "train" / "basketball.yaml"
JERSEY_MODULE_PATH: Path = REPO_ROOT / "src" / "jersey.py"
DETECT_MODULE_PATH: Path = REPO_ROOT / "src" / "detect.py"


# --------------------------------------------------------------------------
# Class scheme, read from train/basketball.yaml
# --------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency of ultralytics
        raise BasketballIntegrationError(
            f"PyYAML is required to read {path} ({exc}). Install it: pip install PyYAML"
        ) from exc
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise BasketballIntegrationError(f"{path} must contain a mapping at the top level")
    return data


def _parse_class_names(data: dict[str, Any], source: Path) -> dict[int, str]:
    raw_names = data.get("names")
    if raw_names is None:
        raise BasketballIntegrationError(f"{source} has no 'names' block")

    if isinstance(raw_names, dict):
        names = {int(key): str(value) for key, value in raw_names.items()}
    elif isinstance(raw_names, (list, tuple)):
        names = {int(index): str(value) for index, value in enumerate(raw_names)}
    else:
        raise BasketballIntegrationError(
            f"{source}: 'names' must be a mapping or a list, got {type(raw_names).__name__}"
        )

    if not names:
        raise BasketballIntegrationError(f"{source}: 'names' is empty")

    expected_indices = list(range(len(names)))
    if sorted(names) != expected_indices:
        raise BasketballIntegrationError(
            f"{source}: class indices must be a gap-free range starting at 0, got "
            f"{sorted(names)}. YOLO label files address classes positionally, so a gap "
            "would shift every class after it."
        )

    declared_count = data.get("nc")
    if declared_count is not None and int(declared_count) != len(names):
        raise BasketballIntegrationError(
            f"{source}: nc is {declared_count} but 'names' lists {len(names)} classes. "
            "Fix the file; this tool will not guess which one is right."
        )
    return names


def _load_module_from_path(name: str, path: Path):
    """Import a single .py file without putting its directory on sys.path.

    Keeps the tool's own module names (app.services.*) from colliding with the
    project's flat module names (jersey, detect, config, ...), which is why the
    modules are registered under prefixed names rather than their own.

    The module MUST be in ``sys.modules`` before ``exec_module`` runs: the
    ``@dataclass`` decorator resolves annotations by looking its own module up
    there, and both jersey.py and detect.py define dataclasses. Registering
    afterwards raises AttributeError on a None module.
    """
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise BasketballIntegrationError(f"Cannot load {path} as a Python module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        # Do not leave a half-initialised module behind for the next caller.
        sys.modules.pop(name, None)
        raise
    return module


_yaml_data = _load_yaml(BASKETBALL_YAML)

#: index -> class name, exactly as train/basketball.yaml declares it.
#: Ordering is by index, never alphabetical: YOLO label files address classes by
#: position, so sorting these names would silently relabel every annotation.
BASKETBALL_CLASS_NAMES: dict[int, str] = _parse_class_names(_yaml_data, BASKETBALL_YAML)
CLASS_COUNT: int = len(BASKETBALL_CLASS_NAMES)

#: Names in index order. Use this anywhere a list is needed (classes.txt, COCO
#: categories, UI dropdowns) so index and position always agree.
CLASS_NAMES_IN_INDEX_ORDER: list[str] = [
    BASKETBALL_CLASS_NAMES[index] for index in range(CLASS_COUNT)
]

_NAME_TO_ID: dict[str, int] = {name: index for index, name in BASKETBALL_CLASS_NAMES.items()}


def _require_class(name: str) -> int:
    if name not in _NAME_TO_ID:
        raise BasketballIntegrationError(
            f"{BASKETBALL_YAML} does not define a '{name}' class, but this tool needs it. "
            f"Declared classes: {CLASS_NAMES_IN_INDEX_ORDER}"
        )
    return _NAME_TO_ID[name]


#: Resolved by NAME, not hardcoded as an integer, so appending classes upstream
#: cannot silently repoint them.
CLASS_PLAYER: int = _require_class("player")
CLASS_BALL: int = _require_class("ball")
CLASS_RIM: int = _require_class("rim")
CLASS_JERSEY: int = _require_class("jersey")
CLASS_REFEREE: int = _require_class("referee")

#: The only class that carries a number. See NUMBER_BEARING_CLASS_ID below.
NUMBER_BEARING_CLASS_ID: int = CLASS_JERSEY

DEFAULT_CLASS_ID: int = CLASS_JERSEY


def _cross_check_against_detect_py() -> None:
    """Verify src/detect.py agrees with train/basketball.yaml.

    They are two hand-maintained copies of one fact. If they disagree, exporting
    against either one produces a dataset that mismatches the runtime, so this
    refuses to start instead of choosing.
    """
    if not DETECT_MODULE_PATH.is_file():
        LOGGER.warning(
            "Cannot cross-check the class scheme: %s not found. Proceeding with "
            "train/basketball.yaml alone.",
            DETECT_MODULE_PATH,
        )
        return
    try:
        detect = _load_module_from_path("basketball_v2_detect", DETECT_MODULE_PATH)
        runtime_names = {int(k): str(v) for k, v in detect.BASKETBALL_CLASS_NAMES.items()}
    except BasketballIntegrationError:
        raise
    except Exception as exc:
        LOGGER.warning(
            "Cannot cross-check the class scheme against %s (%s). Proceeding with "
            "train/basketball.yaml alone.",
            DETECT_MODULE_PATH,
            exc,
        )
        return

    if runtime_names != BASKETBALL_CLASS_NAMES:
        raise BasketballIntegrationError(
            "Class scheme mismatch between the two files that are supposed to mirror "
            "each other:\n"
            f"  {BASKETBALL_YAML}\n    {BASKETBALL_CLASS_NAMES}\n"
            f"  {DETECT_MODULE_PATH}\n    {runtime_names}\n"
            "Exporting against either one would produce a dataset whose class indices "
            "disagree with the detector that consumes it. Reconcile them first; "
            "train/basketball.yaml is the declared source of truth."
        )
    LOGGER.info("Class scheme verified against %s: %s", DETECT_MODULE_PATH.name, BASKETBALL_CLASS_NAMES)


_cross_check_against_detect_py()


# --------------------------------------------------------------------------
# Jersey number normalization, imported from src/jersey.py
# --------------------------------------------------------------------------


def _load_clean_jersey_number() -> Callable[[str | None], str | None]:
    """Import the project's own normalizer.

    src/jersey.py imports only the standard library at module scope (cv2 and
    pytesseract are imported lazily inside the OCR backend), so loading it here
    is cheap and needs no vision dependencies.
    """
    if not JERSEY_MODULE_PATH.is_file():
        raise BasketballIntegrationError(f"Not found: {JERSEY_MODULE_PATH}")
    module = _load_module_from_path("basketball_v2_jersey", JERSEY_MODULE_PATH)
    function = getattr(module, "clean_jersey_number", None)
    if not callable(function):
        raise BasketballIntegrationError(
            f"{JERSEY_MODULE_PATH} does not export a callable 'clean_jersey_number'"
        )
    return function


#: THE project's normalizer, not a copy: digits only, at most two, "00"
#: rejected, leading zeros stripped so "01" and "1" are the same number, and a
#: bare "0" stays legal. Ground truth stored here is normalized with the exact
#: function the inference pipeline votes with.
clean_jersey_number = _load_clean_jersey_number()

JERSEY_NUMBER_RULE = (
    "digits only, at most 2, '00' rejected, leading zeros stripped "
    "('01' -> '1'), bare '0' kept. Applied via src/jersey.py:clean_jersey_number."
)


def normalize_jersey_number(raw_text: str | None) -> str | None:
    """Normalize an operator-typed jersey number, or return None when blank.

    Raises ValueError when text was supplied but is not a legal number, so the
    operator is told immediately instead of discovering it at evaluation time.
    """
    text = (raw_text or "").strip()
    if not text:
        return None
    cleaned = clean_jersey_number(text)
    if cleaned is None:
        raise ValueError(
            f"'{text}' is not a valid jersey number. Rule: {JERSEY_NUMBER_RULE}"
        )
    return cleaned


# --------------------------------------------------------------------------
# Class id validation and dataset.yaml generation / comparison
# --------------------------------------------------------------------------


def validate_class_id(value: object) -> int:
    """Coerce and range-check a class id coming from the API."""
    try:
        class_id = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"class_id must be an integer, got {value!r}") from exc
    if class_id not in BASKETBALL_CLASS_NAMES:
        raise ValueError(
            f"class_id {class_id} is not one of the {CLASS_COUNT} basketball classes: "
            + ", ".join(f"{index}={name}" for index, name in sorted(BASKETBALL_CLASS_NAMES.items()))
        )
    return class_id


def class_name(class_id: int) -> str:
    return BASKETBALL_CLASS_NAMES[validate_class_id(class_id)]


def render_names_block() -> list[str]:
    """The 'nc' and 'names' lines for a generated dataset.yaml, in index order."""
    lines = [f"nc: {CLASS_COUNT}", "names:"]
    lines.extend(f"  {index}: {BASKETBALL_CLASS_NAMES[index]}" for index in range(CLASS_COUNT))
    return lines


def classes_txt_content() -> str:
    """classes.txt: one class name per line, in index order."""
    return "".join(f"{name}\n" for name in CLASS_NAMES_IN_INDEX_ORDER)


def class_mapping_for_metadata() -> dict[str, str]:
    """The {"0": "player", ...} form used inside metadata.json."""
    return {str(index): name for index, name in sorted(BASKETBALL_CLASS_NAMES.items())}


#: Fields compared between a generated dataset.yaml and train/basketball.yaml.
#: 'path', 'train', 'val' and 'test' are deliberately EXCLUDED: an export has its
#: own root directory and its own split layout, and those legitimately differ.
#: What must never differ is the class scheme itself.
COMPARED_FIELDS = ("nc", "names")


def compare_with_basketball_yaml(generated_yaml_path: str | Path) -> list[str]:
    """Diff a generated dataset.yaml against train/basketball.yaml.

    Returns a list of human-readable differences; empty means identical on every
    compared field. Callers must treat a non-empty result as a hard failure.
    """
    path = Path(generated_yaml_path)
    if not path.is_file():
        return [f"generated dataset.yaml not found: {path}"]

    try:
        generated = _load_yaml(path)
    except BasketballIntegrationError as exc:
        return [str(exc)]

    differences: list[str] = []

    generated_nc = generated.get("nc")
    if generated_nc is None:
        differences.append("generated dataset.yaml has no 'nc' field")
    elif int(generated_nc) != CLASS_COUNT:
        differences.append(
            f"nc: generated={generated_nc} but {BASKETBALL_YAML.name}={CLASS_COUNT}"
        )

    try:
        generated_names = _parse_class_names(generated, path)
    except BasketballIntegrationError as exc:
        differences.append(str(exc))
        return differences

    if generated_names != BASKETBALL_CLASS_NAMES:
        for index in sorted(set(generated_names) | set(BASKETBALL_CLASS_NAMES)):
            produced = generated_names.get(index)
            expected = BASKETBALL_CLASS_NAMES.get(index)
            if produced != expected:
                differences.append(
                    f"names[{index}]: generated={produced!r} but "
                    f"{BASKETBALL_YAML.name}={expected!r}"
                )
    return differences


def assert_matches_basketball_yaml(generated_yaml_path: str | Path) -> None:
    """Hard-fail an export whose dataset.yaml disagrees with the source of truth.

    This is the check that makes the whole arrangement trustworthy. Without it a
    class-scheme change upstream would produce a dataset that looks correct,
    imports cleanly into Ultralytics, and trains every class under the wrong
    label. Silence is the dangerous outcome here, so this raises.
    """
    differences = compare_with_basketball_yaml(generated_yaml_path)
    if not differences:
        return
    raise BasketballIntegrationError(
        "Generated dataset.yaml does not match the project class scheme.\n"
        f"  source of truth : {BASKETBALL_YAML}\n"
        f"  generated       : {Path(generated_yaml_path)}\n"
        f"  compared fields : {', '.join(COMPARED_FIELDS)} "
        "(path/train/val/test are expected to differ)\n"
        "Differences:\n  - " + "\n  - ".join(differences)
    )


def describe() -> dict[str, Any]:
    """Summary for logs, metadata files and the /basketball-classes endpoint."""
    return {
        "repo_root": str(REPO_ROOT),
        "source_of_truth": str(BASKETBALL_YAML),
        "class_count": CLASS_COUNT,
        "classes": [
            {"id": index, "name": BASKETBALL_CLASS_NAMES[index]} for index in range(CLASS_COUNT)
        ],
        "number_bearing_class_id": NUMBER_BEARING_CLASS_ID,
        "number_bearing_class_name": BASKETBALL_CLASS_NAMES[NUMBER_BEARING_CLASS_ID],
        "jersey_number_rule": JERSEY_NUMBER_RULE,
        "jersey_number_source": str(JERSEY_MODULE_PATH),
    }


LOGGER.info(
    "Bound to basketball-v2 at %s: %s classes %s",
    REPO_ROOT,
    CLASS_COUNT,
    CLASS_NAMES_IN_INDEX_ORDER,
)
