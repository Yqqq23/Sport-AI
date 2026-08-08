# Sport-CV YOLO Compatibility Analysis

## Summary

- Detected framework: Ultralytics YOLO is the only clearly integrated YOLO detector backend in active Sport-CV code.
- Detected YOLO version: YOLO11-family weights are referenced (`yolo11n.pt`, `yolo11n-pose.pt`) and the MVP backend environment contains `ultralytics 8.4.39`.
- Relevant pipeline choice for Step 12: the Ultralytics-based detector/tracker path is the only realistic compatibility target for a YOLO export.

## Files Inspected

- `D:\25-26香港教育大学\继续卷output。。。\Sport-CV\run_basketball_topdown_projection.py`
- `D:\25-26香港教育大学\继续卷output。。。\Sport-CV\run_basketball_pose_tracking.py`
- `D:\25-26香港教育大学\继续卷output。。。\Sport-CV\0619\README.txt`
- `D:\25-26香港教育大学\继续卷output。。。\Sport-CV\0619\detection.py`
- `D:\25-26香港教育大学\继续卷output。。。\Sport-CV\tracker_configs\bytetrack_sportcv.yaml`
- `D:\25-26香港教育大学\继续卷output。。。\Sport-CV\basketball-analysis-mvp\README.md`
- `D:\25-26香港教育大学\继续卷output。。。\Sport-CV\basketball-analysis-mvp\backend\app\pipeline\real_components.py`
- `D:\25-26香港教育大学\继续卷output。。。\Sport-CV\basketball-analysis-mvp\backend\app\core\config.py`

## Detector Implementations Found

### 1. Root Sport-CV scripts

- `run_basketball_topdown_projection.py`
- `run_basketball_pose_tracking.py`
- `fixed7_jersey_tracking*/*.py`

These scripts use a detector wrapper that prefers:

1. `inference.get_model` (Roboflow inference / RF-DETR style)
2. `ultralytics.RTDETR`
3. `ultralytics.YOLO` fallback

The root scripts are multi-object sports analysis pipelines, not pure YOLO training code.

### 2. `0619` pipeline

- `0619/detection.py` uses `inference_sdk.InferenceHTTPClient` with Roboflow-hosted models.
- It defines semantic classes such as player, jersey number, and ball by class name matching.

### 3. `basketball-analysis-mvp`

- `basketball-analysis-mvp/backend/app/pipeline/real_components.py`
- Uses `from ultralytics import YOLO`
- Loads `settings.yolo_model_path`
- Runs `model.track(...)` with ByteTrack

This is the clearest active Ultralytics detector path in Sport-CV.

## Framework Decision

### A. Actual detection framework used

- Primary active YOLO-compatible framework: `Ultralytics YOLO`
- Also present: Roboflow inference / RF-DETR-style wrappers, RT-DETR fallback, SAM2, MMPose

### B. YOLO version / model family

- Model names referenced:
  - `yolo11n.pt`
  - `yolo11n-pose.pt`
  - `yolov8x.pt` (ball fallback in older scripts)
- Practical conclusion: Sport-CV currently expects the Ultralytics 8.x API while using YOLO11-family weights.

## Dataset / Class Expectations Found

### C. Expected dataset directory structure

- No existing YOLO training dataset directory or `dataset.yaml` was found in Sport-CV source.
- No explicit train/val/test export convention was implemented in Sport-CV.
- Therefore Step 12 should generate a standard Ultralytics-compatible detection dataset layout.

### D. Expected `dataset.yaml` / `data.yaml`

- No project-specific `dataset.yaml` was found.
- Ultralytics API usage implies standard Ultralytics detection YAML is the safest compatible choice.

### E. Existing class IDs and class names

Observed class semantics in Sport-CV code:

- Player-like classes:
  - `player`
  - `person`
  - `player-in-possession`
  - `player-jump-shot`
  - `player-layup-dunk`
  - `player-shot-block`
- Number-like classes:
  - `number`
  - `jersey-number`
  - `jersey number`
  - `digit`
- Ball-like classes:
  - `ball`
  - `sports ball`
  - `ball-in-basket`

Important MVP-specific finding:

- `basketball-analysis-mvp/backend/app/pipeline/real_components.py` uses `classes=[0]` and maps class `0` to `player`.

### F. Does `jersey` already exist as a detection class?

- No direct detection class named `jersey` was found.
- Sport-CV currently reasons about jersey numbers, not jersey-bbox-as-a-class.

### G. Expected bbox format

- Runtime detector outputs are consumed in `xyxy` pixel coordinates.
- Ultralytics training compatibility still requires standard YOLO normalized labels:
  - `<class_id> <x_center> <y_center> <width> <height>`

## Entry Points Found

### H. Training entry point

- No dedicated YOLO training script was found.
- No project-owned `train.py`, `dataset.yaml`, or `model.train(...)` call was found.
- Recommendation: exported Step 12 dataset should target standard Ultralytics training usage external to Sport-CV.

### I. Evaluation entry point

- No dedicated detection evaluation script for YOLO datasets was found.

### J. Inference entry point

- `Sport-CV/basketball-analysis-mvp/backend/app/pipeline/real_components.py`
- `Sport-CV/run_basketball_topdown_projection.py`
- `Sport-CV/run_basketball_pose_tracking.py`

## Detector Scope

### K. Existing detector scope

- Sport-CV is currently a multi-class sports analysis stack.
- It is not a dedicated jersey detector.
- The strongest YOLO path in MVP is actually player detection/tracking, not jersey detection.
- The root pipeline also uses number-like detections in a combined player/number detector.

### L. Recommendation for new jersey dataset

- Default Step 12 mode should train a separate one-class jersey detector dataset.
- Do not silently extend Sport-CV's existing class IDs.
- Future multi-class merge should be handled later by explicitly reconciling class mappings with the player/number detector design.

## Recommended Step 12 Compatibility Design

- Export a standard Ultralytics-compatible one-class YOLO dataset:
  - class `0 = jersey`
  - `images/`
  - `labels/`
  - `dataset.yaml`
  - `classes.txt`
- Mark it as Sport-CV compatible at the framework level because Sport-CV uses Ultralytics YOLO APIs.
- Also document that it is not a drop-in replacement for the current MVP player detector because that detector assumes class `0 = player`.

## Suggested Training Command

No Sport-CV-owned training command exists. The compatible external Ultralytics command is:

```bash
yolo detect train data=dataset.yaml model=yolo11n.pt
```

## Suggested Inference Command Shape

Sport-CV-compatible inference backend uses the Ultralytics Python API, for example:

```python
from ultralytics import YOLO
model = YOLO("yolo11n.pt")
result = model.predict(frame, conf=0.25, verbose=False)
```

For MVP tracking mode:

```python
result = model.track(source=frame, tracker="bytetrack.yaml", classes=[0], persist=True, verbose=False)
```

## Compatibility Risks

- Sport-CV has no built-in YOLO training dataset loader to validate against directly.
- `basketball-analysis-mvp` currently assumes YOLO class `0` means `player`, not `jersey`.
- Root scripts rely heavily on RF-DETR / Roboflow class semantics for players and jersey numbers.
- A one-class jersey detector export is framework-compatible with Ultralytics, but not semantically interchangeable with the current player detector.

## Final Recommendation

- Implement Step 12 as a separate one-class Ultralytics YOLO detection export for `jersey`.
- Keep it isolated from Sport-CV's current player detector mapping.
- Document that future integration into a multi-class Sport-CV detector requires an explicit class remapping and retraining plan.
