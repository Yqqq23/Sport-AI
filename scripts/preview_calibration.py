"""Preview the actual floor warp without loading detector or jersey weights."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from calib import CourtProjector
from config import load_config
from render import draw_half_court_template


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--frame", type=int, default=6000)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/calibration_preview"))
    args = parser.parse_args()
    config = load_config(args.config)
    projector = CourtProjector.from_json(config.calibration_json, config.canvas)
    capture = cv2.VideoCapture(str(args.input))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {args.input}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Cannot read frame {args.frame}")
    finally:
        capture.release()
    projector.check_frame(frame.shape[1], frame.shape[0])
    court = config.render.court
    warped = cv2.warpPerspective(frame, projector.homography, (court.width, court.height))
    template = draw_half_court_template(court)
    # Overlay only model lines, so the photographed floor remains readable.
    lines = np.all(template == court.line_color, axis=2)
    overlay = warped.copy()
    overlay[lines] = (0, 255, 255)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, image in (("court", template), ("warped", warped), ("overlay", overlay)):
        path = args.output_dir / f"{name}.png"
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise RuntimeError(f"Cannot encode {path}")
        path.write_bytes(encoded.tobytes())
        print(path)


if __name__ == "__main__":
    main()
