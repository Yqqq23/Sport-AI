"""Launch the existing Sport-CV calibration GUI without hardcoded Unicode paths."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


POINT_NAMES = ",".join(
    [
        "corner_tl",
        "corner_tr",
        "corner_br",
        "corner_bl",
        "paint_tl",
        "paint_tr",
        "paint_br",
        "paint_bl",
        "free_throw_center",
        "three_left_base",
        "three_right_base",
        "three_left_break",
        "three_right_break",
    ]
)


def find_tool() -> Path:
    patterns = [
        "25-26*/*/Sport-CV/generate_court_calibration_json.py",
        "**/Sport-CV/generate_court_calibration_json.py",
    ]
    matches: list[Path] = []
    for pattern in patterns:
        matches = sorted(path.resolve() for path in Path("D:/").glob(pattern) if path.is_file())
        if matches:
            break
    if not matches:
        raise FileNotFoundError(
            "Could not find Sport-CV/generate_court_calibration_json.py under D:/"
        )
    if len(matches) > 1:
        paths = "\n  ".join(str(path) for path in matches)
        raise RuntimeError(f"Multiple calibration tools found:\n  {paths}")
    return matches[0]


def select_python(sport_cv_root: Path) -> Path:
    candidates = [
        sport_cv_root / ".venv" / "Scripts" / "python.exe",
        sport_cv_root / ".venv_run" / "Scripts" / "python.exe",
        Path(sys.executable),
    ]
    checked: list[str] = []
    for path in candidates:
        if not path.is_file():
            continue
        probe = subprocess.run(
            [
                str(path),
                "-c",
                "import cv2, numpy; "
                "assert 'GUI:                           NONE' not in cv2.getBuildInformation()",
            ],
            capture_output=True,
            text=True,
        )
        if probe.returncode == 0:
            return path
        detail = (probe.stderr or probe.stdout).strip().splitlines()
        checked.append(f"{path}: {detail[-1] if detail else 'dependency probe failed'}")
    raise RuntimeError(
        "No Python interpreter with numpy and GUI-enabled OpenCV was found:\n  "
        + "\n  ".join(checked)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch the existing court calibration GUI")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--frame-idx", type=int, default=100)
    args = parser.parse_args()

    video = args.video.resolve()
    if not video.is_file():
        raise FileNotFoundError(f"Input video not found: {video}")

    tool = find_tool()
    python = select_python(tool.parent)
    output = Path(__file__).resolve().parent.parent / "configs" / "court_calibration.json"

    command = [
        str(python),
        str(tool),
        "--input",
        str(video),
        "--output_json",
        str(output),
        "--frame_idx",
        str(args.frame_idx),
        "--topdown_width",
        "500",
        "--topdown_height",
        "940",
        "--point_names",
        POINT_NAMES,
    ]

    print(f"Tool:   {tool}")
    print(f"Python: {python}")
    print(f"Video:  {video}")
    print(f"Frame:  {args.frame_idx}")
    print(f"Output: {output}")
    print()
    print("Controls: left click=set point, u=undo, k=skip, s=save, q/ESC=cancel")
    print("Only floor-plane landmarks are included. Hoop and backboard points are excluded.")
    print()

    result = subprocess.run(command, cwd=tool.parent)
    if result.returncode != 0:
        return result.returncode
    if not output.is_file():
        raise RuntimeError(f"Calibration finished but output was not created: {output}")

    print()
    print(f"Calibration saved successfully: {output}")
    print("Upload it to:")
    print("  /home/s1143757/basketball-v2/configs/court_calibration.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
