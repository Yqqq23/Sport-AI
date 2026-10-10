# fixed7_jersey_tracking_ocr7 Integration Plan

## Goal

Reuse the strongest parts of:

- `Sport-CV/fixed7_jersey_tracking_ocr7/run_basketball_pose_tracking_fixed7_ocr7_stable.py`

to feed `jersey-active-learning` with real `JerseyPrediction` rows.

## Adapter status

- `backend/app/services/fixed7_adapter.py` now exists as a safe exploratory adapter.
- It can:
  - inspect fixed7 availability
  - parse `tracking_pose_results.json`
  - estimate fallback `jersey_bbox` from player bbox
  - run a guarded dry-run only when required detector weights exist
- It does **not** replace the main recognizer yet.

## Exact files / functions to reuse

Primary source file:

- `D:\25-26香港教育大学\继续卷output。。。\Sport-CV\fixed7_jersey_tracking_ocr7\run_basketball_pose_tracking_fixed7_ocr7_stable.py`

Most relevant functions/classes:

- `build_detector(...)`
- `run_detector_on_frame(...)`
- `build_tracker(...)`
- `NumberRecognizer`
- `recognize_numbers_for_players(...)`
- `pick_fixed_player_slots(...)`
- `run_pipeline(...)`

Most useful intercept point:

- inside `run_pipeline(...)`
- where `player_rows` are assembled before writing `tracking_pose_results.json`

That is the best place to convert the script's internal output into `JerseyPrediction`-compatible records.

## Proposed backend wrapper

Proposed wrapper file:

- `backend/app/services/fixed7_adapter.py`

Proposed function:

```python
run_fixed7_on_video(video_path, output_dir) -> list[dict]
```

Expected wrapper output dict:

- `frame_id`
- `track_id`
- `player_bbox`
- `jersey_bbox`
- `predicted_number`
- `confidence`
- `source_frame_path`

## Mapping to `JerseyPrediction`

Target database fields:

- `video_id`
- `frame_row_id`
- `frame_id`
- `track_id`
- `player_bbox`
- `jersey_bbox`
- `predicted_number`
- `confidence`

Recommended conversion:

- `player_bbox` <- `player_row['bbox_xyxy']`
- `track_id` <- `player_row['track_id']`
- `predicted_number` <- `player_row['number_text'] or ""`
- `confidence` <- `number_conf` if `number_text` exists else `player_conf`

`jersey_bbox` fallback order:

1. `number_bbox` if present
2. estimated upper-body crop from `bbox_xyxy`

`frame_id`:

- use `frame_idx` from the JSON or direct in-memory frame loop

`frame_row_id`:

- resolve by `(video_id, frame_id)` against `VideoFrame`

## Required dependencies

Likely required:

- `opencv-python`
- `numpy`
- `ultralytics`
- `torch` compatible with the environment
- optionally `inference` / `inference_sdk`
- optionally `pytesseract`
- optionally `transformers` for SigLIP team clustering if that path is kept

For minimal integration, team clustering and ball detection should be skipped.

## Required environment variables

Recommended new config for a future adapter:

```env
SPORT_CV_ROOT=D:\25-26香港教育大学\继续卷output。。。\Sport-CV
FIXED7_SCRIPT=D:\25-26香港教育大学\继续卷output。。。\Sport-CV\fixed7_jersey_tracking_ocr7\run_basketball_pose_tracking_fixed7_ocr7_stable.py
FIXED7_DETECTOR_BACKEND=rf_detr
FIXED7_DETECTOR_WEIGHTS=<custom detector model or valid remote model id>
FIXED7_TRACKER_BACKEND=bytetrack
FIXED7_NUMBER_BACKEND=roboflow_workflow
FIXED7_NUMBER_MODEL_PATH=<roboflow api key or OCR config>
FIXED7_POSE_ENABLED=false
```

## Expected runtime speed

Likely moderate to slow compared with the current lightweight recognizer.

Reason:

- per-frame detection
- tracking
- OCR every N frames
- optional pose and clustering overhead

Practical recommendation:

- disable pose
- disable ball detection
- keep `bytetrack`
- keep OCR interval sparse

## Risks

1. default detector weight path appears missing
2. current environment showed Torch / Ultralytics runtime DLL failure
3. OCR depends on Roboflow workflow connectivity or local Tesseract
4. script is designed as a full pipeline executable, not a clean importable library
5. no dedicated `jersey_bbox` field is produced directly

## Minimal implementation steps

1. create a lightweight adapter module that imports the stable script safely
2. bypass video writer, ball detection, and team clustering for annotation use
3. run detector + tracker + OCR only
4. intercept per-frame `player_rows`
5. map rows into `JerseyPrediction`
6. write structured logs for missing detector/OCR dependencies
7. expose adapter behind a new backend recognizer mode

## Safe next implementation step

The exact next step should be:

1. build `backend/app/services/fixed7_adapter.py`
2. make it import `run_basketball_pose_tracking_fixed7_ocr7_stable.py`
3. test detector initialization using an explicitly configured detector weight/model id
4. disable pose and ball logic initially
5. return structured per-frame prediction rows before wiring it into the production recognition endpoint

That is safer than replacing the current recognition service immediately.

## Required missing weights / config

Current blocker observed during audit:

- default detector path expected by fixed7:
  - `models/rf_detr_player_number.pt`
- this file is missing from the fixed7 folder

Exact next step to move forward:

1. obtain the original `rf_detr_player_number.pt`, or
2. identify a valid detector model id/path that `build_detector(...)` accepts in the current environment, and
3. update the adapter/recognizer configuration to use that explicit detector path instead of relying on the missing default file.
