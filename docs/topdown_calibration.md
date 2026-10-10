# Top-down calibration for the Side A video

The default pipeline now uses `configs/court_calibration_refined.json`, imported
from `Computer Vision/calibration/court_3x3.json` (line-refined 2026-08-16).
The original 13-click `court_calibration.json` contains incorrect landmark
correspondences and is no longer selected by the default configuration.

This footage uses a **15 m wide × 14 m deep FIBA half court**, rather than a
dedicated 15 × 11 m 3x3 court. The saved homography maps **1920 × 1080 video
pixels to metres**. It is used directly, without re-fitting the original clicks
or applying the reference project's still-image-to-video conversion again.

Rendering and projection share one metric-to-canvas transform: X points right,
the baseline is below, and +Y points towards the halfway line. At the default
1020 × 960 output size, the 60-pixel margin leaves a 900 × 840 court, or exactly
60 pixels/metre along both axes. Other canvas sizes are fitted with equal scale
and centred, never stretched independently along X and Y.

Player projection uses the detection box's bottom-edge midpoint. Points outside
the canvas are omitted and their display trails reset, rather than clipped to
the border. Existing `court_x/court_y` shot fields remain **canvas pixels**.

The calibration is specific to this fixed camera and image geometry. Different
resolutions are rejected; cropped or different-camera videos need recalibration
even if their resolution is also 1920 × 1080.

## Run on the server

After synchronising the updated source and configs:

```bash
cd /home/s1143757/basketball-v2
python src/pipeline.py --input videos/test.mp4 --config configs/default.yaml --output-dir outputs/run_topdown_fixed --max-frames 1800
```

Outputs include `annotated.mp4`, `topdown.mp4`, `shots.csv` and `shots.json`.
The calibration change requires re-running the pipeline.

To inspect floor-line alignment without loading any model weights:

```bash
python scripts/preview_calibration.py --input videos/test.mp4 --frame 6000
```

This writes `court.png`, `warped.png`, and `overlay.png` under
`outputs/calibration_preview`. Yellow model lines in `overlay.png` should align
with the photographed floor lines.

## Jersey checkpoint

`jersey.backend: auto` prefers the per-digit classifier when it can load
`jersey.perdigit.weights`. The default path resolves to
`checkpoints/stage3_soccernet.pt` in the project root, with the accompanying
`stage3_soccernet.json` required in the same directory. This is distinct from
the detector's `best.pt`. The success log now reports the full checkpoint path.

Use `--jersey-backend perdigit` to select this recognizer explicitly. The older
`--jersey-weights` option overrides `jersey.local.weights` only; to change the
per-digit checkpoint, edit `jersey.perdigit.weights` in the YAML.
