# fixed7_jersey_tracking_ocr7 Audit

## Audited folder

- `D:\25-26香港教育大学\继续卷output。。。\Sport-CV\fixed7_jersey_tracking_ocr7`

## Files inspected

- `D:\25-26香港教育大学\继续卷output。。。\Sport-CV\fixed7_jersey_tracking_ocr7\run_basketball_pose_tracking_fixed7_ocr7_stable.py`
- `D:\25-26香港教育大学\继续卷output。。。\Sport-CV\fixed7_jersey_tracking_ocr7\run_basketball_pose_tracking_fixed7_ocr7.py`

Observed folder contents are minimal:

- no README
- no local `requirements.txt`
- no local config files in this subfolder
- no local weights/checkpoints in this subfolder

## Detected pipeline structure

The stable script declares this pipeline directly:

`RF-DETR detection -> tracking -> pose + jersey number OCR -> fixed 7-player IDs -> annotated video + player-level JSON`

Main phases found in code:

1. detector creation via `build_detector(...)`
2. tracker creation via `build_tracker(...)`
3. optional pose via `build_pose_model(...)`
4. jersey number OCR via `NumberRecognizer(...)`
5. player-team clustering via `TeamClusterEngine(...)`
6. per-frame result writing into:
   - `annotated_video.mp4`
   - `tracking_pose_results.json`
   - `run_summary.json`

## Model framework

Primary framework mix:

- Ultralytics YOLO / RT-DETR fallback support
- Roboflow `inference` package support
- optional MMPose or Ultralytics pose
- optional `pytesseract`
- optional Roboflow workflow OCR via `InferenceHTTPClient`

## Detector model

Detector wrapper:

- class: `RFDETRDetector`
- file: `run_basketball_pose_tracking_fixed7_ocr7_stable.py`

Priority order implemented:

1. `inference.get_model(model_id=weights)`
2. `ultralytics.RTDETR(weights)`
3. `ultralytics.YOLO(weights)` fallback

Default CLI detector arguments:

- `--detector_backend rf_detr`
- `--detector_weights models/rf_detr_player_number.pt`

Important finding:

- the default detector weight is **not** a standard pretrained Ultralytics filename
- it looks like a custom local RF-DETR player+number detector artifact

## Tracker method

Two tracker modes exist:

1. `bytetrack`
   - actually implemented as local `ByteTrackLikeTracker`
   - CPU-friendly custom IoU association tracker
2. `sam2`
   - requires SAM2 checkpoint + config

Default tracker:

- `--tracker_backend bytetrack`

So it can track without SAM2 if detections are available.

## OCR method

OCR wrapper:

- class: `NumberRecognizer`

Supported OCR backends:

1. `roboflow_workflow` / `roboflow`
   - uses `inference_sdk.InferenceHTTPClient`
   - default API key-like string is embedded as `number_model_path`
   - calls Roboflow workflow `basketball-jersey-numbers-ocr`
2. `tesseract` / `ocr`
   - uses local `pytesseract`

Not found in this folder:

- PaddleOCR
- EasyOCR
- TrOCR

## Pretrained model names / auto-download behavior

Found defaults:

- `yolo11n-pose.pt`
- `yolov8x.pt`

These are standard Ultralytics weight names and may auto-download **if** Ultralytics + Torch are working in the environment.

However, the primary detector default is:

- `models/rf_detr_player_number.pt`

This file is not present in the subfolder and is not a standard auto-download Ultralytics name.

## Output files

The main script writes:

- `annotated_video.mp4`
- `tracking_pose_results.json`
- `run_summary.json`

These are created under `--output_dir`.

## Importability / lightweight tests performed

### Module import

Test result:

- importing `run_basketball_pose_tracking_fixed7_ocr7_stable.py` succeeds if loaded carefully via `importlib`

### OCR recognizer init

Test result:

- `NumberRecognizer('roboflow_workflow', '')` initializes a Roboflow client successfully in this environment
- `NumberRecognizer('tesseract', '')` reports `tesseract_ok = False`

Interpretation:

- Roboflow OCR path is at least importable
- local Tesseract is not ready in the current environment

### Pose model init

Test result:

- `build_pose_model('ultralytics', 'yolo11n-pose.pt', device)` returned `None`
- warning showed Torch / DLL loading failure in current environment

Interpretation:

- Ultralytics pose is not currently usable here without fixing Torch/runtime dependencies

### Detector init

Tests attempted:

1. `build_detector('yolo', 'yolo11n.pt', cpu)`
2. `build_detector('rf_detr', 'basketball-player-detection-3-ycjdo/4', cpu)`

Both failed in the current environment.

Observed behavior:

- `inference` package exists but detector init still fails for the tested model strings
- Ultralytics fallback also fails in current environment
- final error:
  - `Failed to build RF-DETR detector. Install inference or ultralytics, and provide valid --detector_weights.`

Interpretation:

- this folder is **not directly runnable in the current environment as-is**

## Can it produce the fields needed by jersey-active-learning?

### `track_id`

Yes.

- produced in `player_rows` and JSON output

### `player_bbox`

Yes.

- exported as `bbox_xyxy`
- raw detector bbox also preserved as `raw_bbox_xyxy`

### `predicted_number`

Yes, partially.

- exported as `number_text`

### `confidence`

Yes, partially.

- detector confidence available as `player_conf`
- OCR confidence available as `number_conf`

### `jersey_bbox`

Not directly as a dedicated field.

- closest existing field is `number_bbox`
- if no number detection is matched, only player bbox / player crop exists

### `source_frame_path`

Not directly emitted in JSON.

- output JSON contains `frame_idx` and per-frame player rows
- source frame path would need to be reconstructed or added

## Gaps relative to `JerseyPrediction`

`JerseyPrediction` needs:

- `video_id`
- `frame_id`
- `track_id`
- `player_bbox`
- `jersey_bbox`
- `predicted_number`
- `confidence`
- `frame_row_id`

Current fixed7 output gaps:

1. no direct `video_id`
2. no direct `frame_row_id`
3. no direct `source_frame_path`
4. no explicit `jersey_bbox` field
5. confidence is split into `player_conf` and `number_conf`, so a fusion rule is needed

## Does it need custom weights?

### A. Can it run with pretrained YOLO auto-download?

Partially.

- pose and ball components reference standard pretrained names (`yolo11n-pose.pt`, `yolov8x.pt`)
- but the **main detector default does not**

### B. Does it require custom jersey detector weights?

Yes, for the default detector path.

- the main detector default is `models/rf_detr_player_number.pt`
- that file is not present in the folder
- this strongly suggests a custom player+number detector is expected

### C. Does it require OCR model weights?

Not necessarily local weights, but it does require a working OCR backend.

- Roboflow workflow OCR uses remote service access, not local model files in this folder
- Tesseract path requires local Tesseract installation

### D. Does it use an OCR library that downloads models automatically?

No clear automatic local OCR model download path was found.

- Roboflow workflow path uses remote inference service
- Tesseract path uses local executable setup

### E. Does it need GPU, or can it run on CPU?

It is designed to support CPU.

- `select_device('auto')` falls back to CPU
- `bytetrack` tracker path is CPU-friendly

However:

- current environment showed Torch / Ultralytics runtime issues, so practical CPU run still depends on fixing local dependencies

## Main entry point

Main entry point:

- `run_basketball_pose_tracking_fixed7_ocr7_stable.py`

Exact CLI shape:

```bash
python run_basketball_pose_tracking_fixed7_ocr7_stable.py --input <video_path> --output_dir <output_dir>
```

Important defaults:

- detector backend: `rf_detr`
- detector weights: `models/rf_detr_player_number.pt`
- tracker backend: `bytetrack`
- number backend: `roboflow_workflow`
- pose backend: `ultralytics`

## Expected outputs

Under output dir:

- `annotated_video.mp4`
- `tracking_pose_results.json`
- `run_summary.json`

`tracking_pose_results.json` contains per-frame player rows with:

- `track_id`
- `fixed_slot_id`
- `bbox_xyxy`
- `raw_bbox_xyxy`
- `number_bbox`
- `number_text`
- `number_conf`
- `player_conf`
- etc.

## Can it run directly?

Answer: **partially**.

Reasons:

- code structure is usable and already outputs structured JSON
- but default primary detector appears to depend on missing custom weights
- current environment also shows Torch / Ultralytics runtime problems
- OCR is tied either to Roboflow workflow access or local Tesseract

## Recommended integration approach

Do not directly shell out to the full pipeline first.

Better approach:

1. reuse the stable script as a reference implementation
2. wrap/import only the pieces needed for:
   - detector
   - tracker
   - OCR
3. intercept `player_rows` just before JSON writing in `run_pipeline(...)`
4. convert each row into `JerseyPrediction` records
5. define fallback mapping:
   - `player_bbox` <- `bbox_xyxy`
   - `jersey_bbox` <- `number_bbox` if present, else estimated upper-body crop from `bbox_xyxy`
   - `predicted_number` <- `number_text`
   - `confidence` <- `number_conf` if number exists, else `player_conf`
   - `track_id` <- `track_id`

## Recommended wrapper shape

If integration proceeds later, the wrapper should look like:

```python
run_fixed7_on_video(video_path, output_dir) -> list[dict]
```

Each dict should contain:

- `frame_id`
- `track_id`
- `player_bbox`
- `jersey_bbox`
- `predicted_number`
- `confidence`
- `source_frame_path` if available

## Final audit conclusion

- Can `fixed7_jersey_tracking_ocr7` be used directly? **partially**
- Does it need custom model weights? **yes** for the default detector path
- Does it use pretrained models? **yes, partially** for pose/ball references, but **not enough** for the main detector path
