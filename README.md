# Sport-AI

**Computer Vision for Basketball Video Analysis, Player Tracking, and Court-Space Analytics**

Sport-AI is a Python-based basketball video analysis project that brings together object detection, multi-object tracking, jersey-number recognition, court calibration, and player trajectory visualisation. It supports both command-line processing and a web interface for reviewing annotated game footage and top-down player movement.

The repository combines the core `basketball-v2` analysis pipeline with additional modules for court calibration, metric-space trajectories, annotation, and jersey-recognition training. It is designed primarily for fixed-camera basketball footage and computer-vision experimentation.

## Key Features

| Component | Capabilities |
| --- | --- |
| Object detection | Detect players, basketballs, rims, jersey regions, and referees with an Ultralytics YOLO-based detector. |
| Multi-object tracking | Configurable tracking backends, including ByteTrack and BoT-SORT. |
| Motion identity | Maintain motion tracks separately from recognised jersey labels; use Kalman filtering and gated Hungarian matching for court-space association. |
| Jersey recognition | Per-digit classification, PARSeq, and optional OCR backends, with temporal evidence aggregation. |
| Court calibration | Homography-based projection from image coordinates to a top-down court view. |
| Player analytics | Visualise individual trajectories, movement heatmaps, animations, and metric-coordinate data. |
| Shot-event analysis | Heuristic basket-event detection from ball/rim motion, exported as CSV and JSON. |
| Web application | Upload footage, monitor processing, review annotated/top-down videos, and generate jersey-specific views. |
| Training and annotation | Utilities for preparing detection datasets, annotating five object classes, and training jersey-recognition models. |

## System Overview

```text
Basketball video
      |
      v
YOLO object detection
      |
      +--> Players --> Multi-object tracking --> Foot-point projection
      |                                          |
      |                                          v
      |                                 Court-space motion tracks
      |                                          |
      +--> Jersey regions --> Number recognition + temporal voting
      |                                          |
      |                                          v
      |                            Annotated video / top-down view
      |                            Trajectories / heatmaps / CSV
      |
      +--> Ball + rim --> Shot-event heuristics --> CSV / JSON
```

**Tracking design:** Motion IDs represent trajectories; jersey numbers are separately estimated labels. A shared number alone does not merge two motion tracks. Brief occlusions can be bridged with limited prediction, while longer interruptions may remain discontinuous.

## Technology Stack

- **Machine learning:** PyTorch, Torchvision, Ultralytics YOLO
- **Tracking and numerical processing:** ByteTrack, BoT-SORT, Kalman filtering, SciPy, NumPy, Pandas
- **Vision and video:** OpenCV, FFmpeg
- **Jersey recognition:** ResNet18-based per-digit classifier, PARSeq, optional PaddleOCR/EasyOCR
- **Backend:** FastAPI, Uvicorn
- **Frontend:** HTML, CSS, JavaScript
- **Configuration and exports:** YAML, JSON, JSONL, CSV

## Repository Structure

```text
Sport-AI/
├── src/                        # Core detection, tracking, identity, rendering and analytics
│   ├── pipeline.py             # Command-line analysis entry point
│   ├── detect.py
│   ├── track.py
│   ├── motion.py
│   ├── identity.py
│   ├── jersey.py
│   ├── calib.py
│   ├── shot.py
│   └── render.py
├── ComputerVision/
│   ├── calibration/            # Court-point selection and homography calibration
│   ├── trajectory/             # Metric trajectories, heatmaps and visualisations
│   └── jersey_train/          # Local jersey-classifier training
├── webapp/                     # FastAPI application and frontend
├── configs/                    # Model, tracking and calibration configuration
├── scripts/                    # Data preparation and validation utilities
├── train/                      # Detection training workflows
├── jersey-train-standalone/    # Per-digit jersey model training
├── jersey-active-learning/     # Jersey labelling tools
├── tools/annotation/           # Detection annotation utilities
├── docs/                       # Technical documentation
├── tests/                      # Tests
└── requirements.txt
```

> Some data directories, checkpoints, or generated outputs may be unavailable in a public checkout. Verify paths against the current repository version.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Yqqq23/Sport-AI.git
cd Sport-AI
```

### 2. Create a Python environment

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

Install a PyTorch/Torchvision build appropriate for your CPU or CUDA environment using the [official installation instructions](https://pytorch.org/get-started/locally/), then:

```bash
python -m pip install -r requirements.txt
python -m pip install -r webapp/requirements.txt
```

For the standalone trajectory module:

```bash
python -m pip install -r ComputerVision/trajectory/requirements.txt
```

Install **FFmpeg** and ensure `ffmpeg` is available on your `PATH` if you intend to use web video encoding.

### 4. Configure models and calibration

Review `configs/default.yaml` and set paths and devices for your environment, particularly:

- `detector.weights`: basketball-specific detector checkpoint
- `detector.device` and `tracker.device`: inference device
- `jersey.backend`: selected number-recognition backend
- `jersey.perdigit.weights`: per-digit checkpoint, when enabled
- `calibration_json`: calibration corresponding to the camera view

For per-digit recognition, the checkpoint and its matching JSON metadata must be available. Pretrained model files and datasets may need to be supplied separately.

## Run Video Analysis

Run from the repository root:

```bash
python src/pipeline.py --input "path/to/game.mp4" --config configs/default.yaml --output-dir outputs/game1
```

For a shorter test:

```bash
python src/pipeline.py --input "path/to/game.mp4" --config configs/default.yaml --output-dir outputs/demo --max-frames 300
```

To export trajectory data:

```bash
python src/pipeline.py --input "path/to/game.mp4" --config configs/default.yaml --output-dir outputs/game1 --jersey-backend perdigit --export-trajectories
```

### Typical outputs

| File | Description |
| --- | --- |
| `annotated.mp4` | Video with detections, motion IDs and available jersey labels |
| `topdown.mp4` | Top-down court representation |
| `shots.csv`, `shots.json` | Heuristic basket-event records |
| `players/motion_*.mp4` | Individual motion-track videos |
| `players/motion_*.png` | Accumulated trajectory images |
| `players/index.json` | Motion-track metadata |
| `trajectories.jsonl` | Per-frame trajectory records when trajectory export is enabled |

**Coordinate note:** The core pipeline's `court_x` and `court_y` shot-export fields refer to top-down canvas pixels. For coordinates in metres, use the `ComputerVision/trajectory` workflow.

## Web Interface

Start the FastAPI server from the repository root:

```bash
python -m uvicorn webapp.app:app --host 0.0.0.0 --port 8000 --workers 1
```

Open `http://localhost:8000` for the interface or `http://localhost:8000/docs` for API documentation.

The interface supports uploading footage, checking analysis and encoding progress, reviewing processed videos, and generating views for selected jersey numbers. Video playback depends on FFmpeg and correctly configured model checkpoints.

See [Web Application Documentation](docs/webapp.md).

## Court Calibration and Trajectory Analysis

The `ComputerVision/calibration` module supports manual point selection, homography estimation, refinement, validation, and top-down projection:

```bash
cd ComputerVision
python -m calibration.pick_points
python -m calibration.calibrate
python -m calibration.refine
python -m calibration.validate
python -m calibration.topdown
```

The documented calibration example uses fixed-camera **1920 × 1080** footage and a **15 m × 14 m** half-court representation. Recalibrate when the camera geometry, crop, or court changes. Player locations are projected using the bottom-centre of their bounding boxes as an approximation of the foot point.

To extract metric trajectories from the `ComputerVision` directory:

```bash
python -m trajectory.cli check --repo ..
python -m trajectory.cli extract --repo .. --video "path/to/game.mp4" --out outputs/run1 --duration-sec 60
python -m trajectory.cli players --csv outputs/run1/tracks.csv
python -m trajectory.cli plot --csv outputs/run1/tracks.csv --jersey 7
```

The exported `tracks.csv` includes `court_x_m` and `court_y_m`. Trajectory processing includes jump filtering, smoothing, and limited interpolation across short gaps.

See [Calibration](ComputerVision/calibration/README.md) and [Trajectory Analysis](ComputerVision/trajectory/README.md).

## Model Training and Annotation

The detection dataset defines five classes:

| ID | Class |
| --- | --- |
| 0 | `player` |
| 1 | `ball` |
| 2 | `rim` |
| 3 | `jersey` |
| 4 | `referee` |

Dataset preparation and detector training entry points:

```bash
python scripts/prepare_dataset.py
python train/train.py train --data train/datasets/basketball/data.yaml
```

The repository also contains a standalone per-digit jersey-recognition workflow with a shared ResNet18 backbone and separate heads for number length/readability and individual digits. Its documented stages include SVHN pretraining, synthetic jersey data, and SoccerNet-related fine-tuning. A separate local jersey classifier is available in `ComputerVision/jersey_train`.

See [Annotation Guide](docs/annotation_guide.md) and [Standalone Jersey Training](jersey-train-standalone/README.md).

## Testing

```bash
python -m pip install pytest
python -m pytest tests/test_motion.py tests/test_calibration.py -q
python -m pytest tests/test_webapp.py -q
```

End-to-end validation requires appropriate weights, representative footage, and the relevant video-processing dependencies.

## Current Limitations

- Calibration is specific to the source camera geometry.
- Small, blurred, or occluded jersey numbers may be unreadable.
- Overlapping players and long occlusions can cause identity switches or fragmented tracks.
- Jersey-based web views may include multiple players sharing the same number across teams.
- Shot-event detection is heuristic and based on single-view 2D geometry; it is not a complete shot-attempt or shooting-percentage estimator.
- Runtime performance varies with hardware, video resolution, selected models, and OCR configuration.

## Further Documentation

- [Web App](docs/webapp.md)
- [Motion Tracking](docs/motion_tracking.md)
- [Top-Down Calibration](docs/topdown_calibration.md)
- [Annotation Guide](docs/annotation_guide.md)
- [Court Calibration](ComputerVision/calibration/README.md)
- [Trajectory Module](ComputerVision/trajectory/README.md)
- [Local Jersey Classifier](ComputerVision/jersey_train/README.md)

---

*Sport-AI is an experimental computer-vision project. Its predictions and statistics should be validated against annotated footage before being used for performance evaluation.*
