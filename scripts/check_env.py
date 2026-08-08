"""First script to run after uploading to the GPU server.

Verifies every external dependency this project needs and prints a concrete fix
for each failure. Exits non-zero if any REQUIRED check fails.

    python scripts/check_env.py [--config configs/default.yaml]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

MIN_ULTRALYTICS = (8, 4, 63)
MIN_ULTRALYTICS_STR = ".".join(str(v) for v in MIN_ULTRALYTICS)

REQUIRED_TRACKERS = (
    "botsort.yaml",
    "bytetrack.yaml",
    "ocsort.yaml",
    "deepocsort.yaml",
    "fasttrack.yaml",
    "tracktrack.yaml",
)

PASS = "[ OK ]"
FAIL = "[FAIL]"
WARN = "[WARN]"

failures: list[str] = []
warnings: list[str] = []


def report_pass(message: str) -> None:
    print(f"{PASS} {message}")


def report_fail(message: str, fix: str) -> None:
    print(f"{FAIL} {message}")
    print(f"       fix: {fix}")
    failures.append(message)


def report_warn(message: str, fix: str) -> None:
    print(f"{WARN} {message}")
    print(f"       fix: {fix}")
    warnings.append(message)


def section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def parse_version(text: str) -> tuple[int, ...]:
    parts: list[int] = []
    for chunk in str(text).split(".")[:3]:
        digits = "".join(c for c in chunk if c.isdigit())
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


# --------------------------------------------------------------------------


def check_torch() -> None:
    section("1. PyTorch and GPU")
    try:
        import torch
    except Exception as exc:
        report_fail(
            f"cannot import torch ({exc})",
            "install a build matching the server CUDA version, e.g.\n"
            "            pip install torch torchvision --index-url "
            "https://download.pytorch.org/whl/cu121",
        )
        return

    report_pass(f"torch {torch.__version__}")

    if not torch.cuda.is_available():
        report_fail(
            "torch.cuda.is_available() is False - inference would run on CPU",
            "check `nvidia-smi` works, then reinstall the CUDA build of torch. "
            "A CPU-only wheel is the usual cause.",
        )
        return

    report_pass(f"CUDA available, built against CUDA {torch.version.cuda}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / (1024**3)
        report_pass(f"device {i}: {props.name}, {memory_gb:.1f} GB")


def check_ultralytics() -> None:
    section("2. Ultralytics and tracker backends")
    try:
        import ultralytics
    except Exception as exc:
        report_fail(
            f"cannot import ultralytics ({exc})",
            f"pip install 'ultralytics>={MIN_ULTRALYTICS_STR}'",
        )
        return

    version = getattr(ultralytics, "__version__", "0.0.0")
    if parse_version(version) < MIN_ULTRALYTICS:
        report_fail(
            f"ultralytics {version} is older than the required {MIN_ULTRALYTICS_STR}",
            f"pip install -U 'ultralytics>={MIN_ULTRALYTICS_STR}'\n"
            "            ocsort, deepocsort, fasttrack and tracktrack were added in "
            f"{MIN_ULTRALYTICS_STR}; tracktrack.yaml is this project's DEFAULT backend, "
            "so the default config cannot run on an older version.",
        )
    else:
        report_pass(f"ultralytics {version} (>= {MIN_ULTRALYTICS_STR})")

    tracker_dir = Path(ultralytics.__file__).resolve().parent / "cfg" / "trackers"
    if not tracker_dir.is_dir():
        report_fail(
            f"tracker config directory not found: {tracker_dir}",
            "the ultralytics install looks damaged; reinstall it",
        )
        return

    present = sorted(p.name for p in tracker_dir.glob("*.yaml"))
    for name in REQUIRED_TRACKERS:
        if name in present:
            report_pass(f"tracker config found: {name}")
        else:
            report_fail(
                f"tracker config MISSING: {name}",
                f"pip install -U 'ultralytics>={MIN_ULTRALYTICS_STR}' "
                f"(this install has: {present})",
            )


def check_model_load() -> None:
    section("3. Model loading")
    try:
        from ultralytics import YOLO
    except Exception as exc:
        report_fail(f"cannot import YOLO ({exc})", "fix the ultralytics install first")
        return

    try:
        # Downloads on first run. This is the only network access check_env does.
        model = YOLO("yolo26n.pt")
        names = getattr(model, "names", {})
        report_pass(f"loaded yolo26n.pt ({len(names)} classes)")
    except Exception as exc:
        report_fail(
            f"cannot load yolo26n.pt ({exc})",
            "check outbound network access, or download the weights manually and "
            "place them in the working directory",
        )


def check_tesseract() -> None:
    section("4. Jersey OCR (tesseract)")
    try:
        import pytesseract
    except Exception as exc:
        report_warn(
            f"cannot import pytesseract ({exc})",
            "pip install pytesseract  (jersey numbers stay unknown without it)",
        )
        return
    report_pass("pytesseract python package importable")

    binary = shutil.which("tesseract")
    if binary is None:
        report_warn(
            "tesseract BINARY not found on PATH",
            "sudo apt-get install -y tesseract-ocr\n"
            "            The pip package is only a binding. Without the binary the "
            "pipeline still runs, but every jersey number stays unknown and output 3 "
            "falls back to track ids.",
        )
        return

    try:
        version = subprocess.run(
            [binary, "--version"], capture_output=True, text=True, timeout=15
        ).stdout.splitlines()[0]
        report_pass(f"tesseract binary: {version.strip()} ({binary})")
    except Exception as exc:
        report_warn(f"tesseract found at {binary} but did not run ({exc})", "reinstall tesseract-ocr")


def check_other_imports() -> None:
    section("5. Other Python dependencies")
    for module, fix in (
        ("cv2", "pip install opencv-python"),
        ("numpy", "pip install numpy"),
        ("yaml", "pip install PyYAML"),
    ):
        try:
            imported = __import__(module)
            report_pass(f"{module} {getattr(imported, '__version__', '')}".strip())
        except Exception as exc:
            report_fail(f"cannot import {module} ({exc})", fix)


def check_config(config_path: Path) -> None:
    section("6. Project configuration and paths")
    if not config_path.exists():
        report_fail(
            f"config not found: {config_path}",
            "pass --config with the right path, or restore configs/default.yaml",
        )
        return
    report_pass(f"config file present: {config_path}")

    try:
        from config import load_config
    except Exception as exc:
        report_fail(f"cannot import src/config.py ({exc})", "run this script from the repo root")
        return

    try:
        cfg = load_config(config_path)
    except Exception as exc:
        report_fail(f"config failed to load: {exc}", "fix the YAML error reported above")
        return
    report_pass("config parsed and type-checked")

    for problem in cfg.validate():
        if "detector.weights" in problem:
            report_warn(
                problem,
                "train first:  python train/train.py train --data train/basketball.yaml\n"
                "            Until then use scripts/smoke_test.py and "
                "scripts/compare_trackers.py, which accept official weights explicitly.",
            )
        elif "calibration_json" in problem:
            report_warn(
                problem,
                "cp configs/court_calibration.template.json configs/court_calibration.json\n"
                "            then replace every image_points placeholder with real clicked "
                "pixel coordinates. Without it there is no top-down video and no court_x/court_y.",
            )
        else:
            report_fail(problem, "fix the config")

    tracker_ref = Path(cfg.tracker.tracker)
    if tracker_ref.suffix in {".yaml", ".yml"} and tracker_ref.name not in REQUIRED_TRACKERS:
        candidate = tracker_ref if tracker_ref.is_absolute() else config_path.parent / tracker_ref
        if candidate.exists():
            report_pass(f"custom tracker config found: {candidate}")
        else:
            report_fail(
                f"configured tracker file not found: {candidate}",
                "point tracker.tracker at an existing YAML or a bundled backend name",
            )
    else:
        report_pass(f"tracker backend selected: {cfg.tracker.tracker}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the runtime environment")
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "configs" / "default.yaml"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip the yolo26n.pt load check (avoids network access)",
    )
    args = parser.parse_args()

    print("=" * 62)
    print("basketball-v2 environment check")
    print("=" * 62)

    check_torch()
    check_ultralytics()
    if not args.skip_download:
        check_model_load()
    else:
        section("3. Model loading")
        print(f"{WARN} skipped by --skip-download")
    check_tesseract()
    check_other_imports()
    check_config(args.config)

    print()
    print("=" * 62)
    if failures:
        print(f"{len(failures)} REQUIRED check(s) failed:")
        for item in failures:
            print(f"  - {item}")
    if warnings:
        print(f"{len(warnings)} optional check(s) need attention:")
        for item in warnings:
            print(f"  - {item}")
    if not failures and not warnings:
        print("All checks passed. The environment is ready.")
    elif not failures:
        print("No blocking failures. The pipeline will run, with the caveats above.")
    print("=" * 62)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
