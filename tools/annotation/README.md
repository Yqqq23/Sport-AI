# jersey-active-learning

Full-stack starter for active learning on jersey number data.

## Current Real Recognition Workflow

Primary RA workflow:

1. Upload a basketball video
2. Select the video
3. Click `Run Jersey Recognition`
4. Click `Load Review Candidates`
5. Correct bbox / jersey number
6. Save annotation
7. Click `Download Corrected Dataset`

Important docs:

- `docs/real_recognition_integration.md`
- `docs/current_recognition_pipeline_audit.md`
- `docs/sport_cv_yolo_compatibility.md`
- `docs/dataset_versioning_design.md`

Required backend configuration:

```env
REAL_RECOGNITION_ENABLED=true
SPORT_CV_ROOT=D:\25-26香港教育大学\继续卷output。。。\Sport-CV
PLAYER_DETECTOR_WEIGHTS=D:\path\to\player_detector_weights.pt
JERSEY_OCR_WEIGHTS=<optional-ocr-access-token>
DETECTOR_CONFIDENCE=0.2
DETECTOR_IOU=0.45
RECOGNITION_FRAME_STRIDE=5
```

If `PLAYER_DETECTOR_WEIGHTS` is missing, the backend returns a clear model-missing error and does not fall back to synthetic predictions.

## RA 使用流程

1. 启动后端和前端
2. 上传视频
3. 点击 `Run YOLO Pre-annotation`
4. 点击 `Load Frames for Annotation`
5. 修正 bbox 和 jersey number
6. 保存标注
7. 下载 `Corrected Dataset`

说明：

- YOLO 预标注只是辅助工具，人工标注结果才是最终真实数据。
- 即使 OCR、跟踪或自定义权重不可用，系统也应允许直接对抽帧图片进行手工标注。

## 服务器手动上传与启动

如果你准备把项目手动上传到 EdUHK Linux GPU server，请先阅读：

- `docs/eduhk_gpu_manual_deployment.md`
- `docs/manual_upload_checklist.md`

服务端环境示例文件：

- `backend/.env.server.example`
- `frontend/.env.server.example`

辅助脚本：

- `scripts/check_server_env.sh`
- `scripts/start_server_backend.sh`
- `scripts/start_server_frontend.sh`

## Swagger 使用说明

- Swagger 中只保留了核心流程接口；COCO / YOLO / Dataset Version 等高级接口仍然保留在后端，但默认不显示。
- 普通使用者建议按照以下顺序调用接口：上传视频 → 运行识别 → 查看候选帧 → 保存标注 → 下载数据集。

## Stack

- Backend: FastAPI + Python
- Frontend: React + Vite
- Database: SQLite
- Data folders: `data/videos`, `data/frames`, `data/crops`, `data/annotations`, `data/exports`

## Project Structure

```text
jersey-active-learning/
  backend/
  frontend/
  data/
```

## Quick Start

### 1) Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Backend runs at `http://127.0.0.1:8000`.

### 2) Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at `http://127.0.0.1:5173`.

### 3) API Docs

- Swagger UI: `http://127.0.0.1:8000/docs`

### 4) Debug Upload

With the backend running, execute:

```bash
cd backend
python debug_upload.py
```

This sends the sample 7-second video to `POST /upload-video` with `frame_interval=30` and prints status, response, and elapsed time.

## Environment

Create `backend/.env` if needed:

```env
APP_NAME=jersey-active-learning
APP_ENV=development
DATABASE_URL=sqlite:///./jersey_active_learning.db
CORS_ORIGINS=http://127.0.0.1:5173,http://localhost:5173
DATA_DIR=../data
```

## Notes

- SQLite database file is created in `backend/jersey_active_learning.db`.
- Upload and dataset paths are managed from `data/`.
- This is a starter scaffold, ready to extend with video ingestion, frame extraction, crop review, and annotation workflows.

## Implemented Video APIs

- `POST /upload-video`
  - multipart form fields:
    - `file`: video file
    - `frame_interval`: integer, defaults to `30`
- `GET /videos`
- `GET /videos/{video_id}/frames`
- `POST /videos/{video_id}/run-jersey-recognition`
- `POST /videos/{video_id}/run-jersey-recognition-async`
- `GET /videos/{video_id}/recognition-summary`
- `GET /videos/{video_id}/predictions`
- `GET /videos/{video_id}/failure-frames?threshold=0.7`
- `GET /videos/{video_id}/inconsistent-frames`
- `GET /videos/{video_id}/annotation-candidates?threshold=0.7`

Upload behavior:

- saves video into `data/videos/`
- extracts frames every `N` frames using OpenCV
- stores extracted frames in `data/frames/{video_id}/`
- writes `video_id`, `frame_id`, and `frame_path` into SQLite

## Recognition And Candidate Review

Normal workflow:

1. `POST /videos/{video_id}/run-jersey-recognition` or `POST /videos/{video_id}/run-jersey-recognition-async`
2. `GET /videos/{video_id}/recognition-summary`
3. `GET /videos/{video_id}/annotation-candidates?threshold=0.7`

Candidate generation behavior:

- uses real `JerseyPrediction` rows produced by the recognition service
- selects candidates from:
  - low confidence predictions
  - temporal inconsistency patterns such as `23 -> 28 -> 23`
- merges reasons by `prediction_id`
- sorts candidates by `frame_id`, then `track_id`, then `prediction_id`

Frontend review behavior:

- fetch uploaded videos from `GET /videos`
- select one video
- run jersey recognition
- load review candidates manually
- show one candidate at a time
- render the frame image and draw the predicted `jersey_bbox`
- allow manual bbox correction and jersey number correction
- save/update/delete annotation

Developer note:

- legacy mock endpoints may exist only for tests/debugging

## Step 7 Frontend Annotation Candidate Viewer

Backend changes:

- static frames are exposed from `/static/frames`
- annotation candidate responses include `image_url`
- frontend should use `http://127.0.0.1:8000${image_url}`

Viewer behavior:

- fetch uploaded videos from `GET /videos`
- select one video
- fetch candidates from `GET /videos/{video_id}/annotation-candidates?threshold=0.7`
- show one candidate at a time
- render the frame image and draw the `jersey_bbox`
- show `predicted_number`, `confidence`, `track_id`, `frame_id`, and `reasons`
- provide `Previous` and `Next` navigation

Testing:

1. Start backend:
   - `cd backend`
   - `uvicorn app.main:app --reload`
2. Start frontend:
   - `cd frontend`
   - `npm install`
   - `npm run dev`
3. Upload a video
4. Run jersey recognition for that video
5. Open the viewer in the frontend and verify:
   - image loads using `image_url`
   - jersey box is drawn on top of the frame
   - Previous/Next works
   - loading and error states display correctly

## Step 7.5 Async Job Queue

Backend:

- `processing_jobs` SQLite table tracks long-running tasks
- `GET /jobs/{job_id}` returns job status and progress
- `POST /videos/{video_id}/run-jersey-recognition-async` queues a background task
- synchronous `POST /videos/{video_id}/run-jersey-recognition` is available

Async flow:

1. Frontend calls `POST /videos/{video_id}/run-jersey-recognition-async`
2. Backend returns:

```json
{
  "job_id": "abc123...",
  "status": "queued",
  "poll_url": "/jobs/abc123..."
}
```

3. Frontend polls `GET /jobs/{job_id}` every second
4. Job progresses through:
   - `queued`
   - `running`
   - `completed` or `failed`
5. When completed, annotation candidates are refreshed automatically

Testing:

1. Start backend
2. Start frontend
3. Select a video
4. Click `Run Jersey Recognition`
5. Confirm job goes `queued -> running -> completed`
6. Confirm candidates refresh after completion

## Step 8 Manual Annotation Editing

Backend:

- `manual_annotations` SQLite table stores bbox and number corrections
- `POST /annotations` saves or updates one annotation for a prediction
- `GET /videos/{video_id}/annotations` returns saved annotations for the selected video

Frontend workflow:

1. Select a video in the frontend
2. Open an annotation candidate
3. Drag the green edited bbox to reposition it
4. Drag the bottom-right green handle to resize it
5. Enter a corrected jersey number if needed
6. Choose one of:
   - `correct`
   - `wrong_bbox`
   - `wrong_number`
   - `unclear`
7. Click `Save Annotation`

Behavior:

- the editable bbox starts from the predicted `jersey_bbox`
- corrected number defaults to the predicted jersey number
- saved annotations keep you on the same candidate
- saved candidates are marked as `Saved` in the candidate metadata

API example:

```json
{
  "prediction_id": 123,
  "video_id": 2,
  "frame_id": 60,
  "track_id": 1,
  "image_path": "data/frames/2/frame_000060.jpg",
  "image_url": "/static/frames/2/frame_000060.jpg",
  "original_bbox": [257, 478, 354, 559],
  "corrected_bbox": [260, 480, 350, 560],
  "original_number": "28",
  "corrected_number": "23",
  "label_status": "wrong_number",
  "image_width": 1920,
  "image_height": 1080
}
```

View saved annotations:

- `GET /videos/{video_id}/annotations`
- or use Swagger at `http://127.0.0.1:8000/docs`

Step 9 verification:

1. Load candidates for a video
2. Edit one candidate and click `Save Annotation`
3. Confirm `GET /videos/{video_id}/annotations` returns one row for that `prediction_id`
4. Change the bbox, number, or status and click `Update Annotation`
5. Confirm `GET /annotations/{annotation_id}` reflects the latest corrected fields
6. Click `Delete Annotation`
7. Confirm the deleted annotation no longer appears in `GET /videos/{video_id}/annotations`
8. Check `GET /videos/{video_id}/annotation-progress` for:
   - `annotated_count`
   - `remaining_count`
   - `progress_percent`
9. Re-save the same `prediction_id` + `video_id` and verify no duplicate active annotations are created

## Step 10 Internal Dataset Export

Purpose:

- `Internal Dataset Export` keeps the project-native review format with copied frames, crops, tabular annotation files, and metadata for auditing.
- `COCO Export` is for jersey bbox detection training and uses one detection category: `jersey`.

Export endpoint:

- `POST /videos/{video_id}/export-internal-dataset`

List/download endpoints:

- `GET /videos/{video_id}/exports`
- `GET /exports/{export_id}/download`

Export output folder:

- `data/exports/internal/{video_id}_{timestamp}/`
- contains:
  - `images/`
  - `crops/`
  - `annotations.csv`
  - `annotations.json`
  - `metadata.json`

Testing:

1. Save at least one manual annotation
2. Trigger `Export Internal Dataset` from the frontend or call `POST /videos/{video_id}/export-internal-dataset`
3. Confirm the export folder is created under `data/exports/internal/`
4. Check `images/` for copied source frames
5. Check `crops/` for corrected bbox crops
6. Open `annotations.csv`
7. Open `annotations.json`
8. Open `metadata.json`
9. Confirm `GET /videos/{video_id}/exports` lists the export

## Step 11 COCO Detection Dataset Export v1

Purpose:

- exports active manual annotations as a COCO detection dataset for jersey bbox detection
- uses corrected bbox coordinates as the detection target
- keeps corrected jersey numbers in `labels.csv` and custom COCO annotation metadata for later number-recognition work

COCO design:

- exactly one category is exported:

```json
{
  "id": 1,
  "name": "jersey",
  "supercategory": "sports_uniform"
}
```

- `corrected_bbox` is converted from `[x1, y1, x2, y2]` to COCO `[x, y, width, height]`
- example: `[100, 200, 300, 500] -> [100, 200, 200, 300]`
- `corrected_number` is not a COCO category; it is exported to `labels.csv` and stored as custom JSON metadata
- annotations with `label_status = unclear` are skipped

COCO output folder:

- `data/exports/coco/{video_id}_{timestamp}/`
- contains:
  - `images/`
  - `annotations/instances.json`
  - `labels.csv`
  - `metadata.json`

API endpoints:

- `POST /videos/{video_id}/export-coco-dataset`
- `GET /videos/{video_id}/coco-exports`
- `GET /coco-exports/{export_id}/download`

Validation:

- backend runs `validate_coco_dataset(instances.json)` automatically after export
- it verifies:
  - `instances.json` exists and is valid JSON
  - `images`, `annotations`, and `categories` are lists
  - exactly one jersey category exists
  - referenced image files exist
  - bbox values are valid and positive
  - image IDs and annotation IDs are unique

Frontend testing:

1. Save at least one manual annotation with status `correct`, `wrong_bbox`, or `wrong_number`
2. Click `Export COCO Dataset`
3. Confirm the result panel shows:
   - `export_id`
   - `dataset_format`
   - `image_count`
   - `annotation_count`
   - skipped counters
   - `export_path`
   - `download_path`
4. Confirm available COCO exports refresh for the selected video

Backend/API testing:

1. Call `POST /videos/{video_id}/export-coco-dataset`
2. Check `data/exports/coco/{video_id}_{timestamp}/`
3. Open `annotations/instances.json`
4. Open `labels.csv`
5. Open `metadata.json`
6. Call `GET /videos/{video_id}/coco-exports`
7. Call `GET /coco-exports/{export_id}/download`
8. Confirm duplicate-frame annotations share one COCO image entry
9. Confirm `unclear` annotations are skipped and counted in metadata

## Step 12 Sport-CV-Compatible YOLO Dataset Export

Compatibility analysis:

- detailed inspection report: `docs/sport_cv_yolo_compatibility.md`
- detected framework: `Ultralytics YOLO`
- detected model/version: `YOLO11-family weights via ultralytics 8.x`
- important limitation: current Sport-CV MVP inference assumes YOLO class `0 = player`, so this export is a separate one-class jersey detector dataset rather than a drop-in replacement for the player detector

Purpose:

- `Internal Export` is for project-native review and auditing
- `COCO Export` is for jersey bbox detection in COCO format
- `YOLO Export` is for Sport-CV-compatible Ultralytics-style jersey bbox detection training data

YOLO bbox conversion:

- internal bbox: `[x1, y1, x2, y2]`
- YOLO label: `<class_id> <x_center> <y_center> <width> <height>`
- normalized by image width and height

YOLO output folder:

- `data/exports/yolo/{video_id}_{timestamp}/`
- contains:
  - `images/`
  - `labels/`
  - `dataset.yaml`
  - `classes.txt`
  - `metadata.json`

Behavior:

- exactly one class is exported by default:
  - `0 = jersey`
- multiple annotated jerseys from the same frame produce:
  - one copied image
  - one `.txt` label file
  - multiple YOLO label lines inside that file
- `unclear` annotations are skipped
- invalid bboxes and missing images are skipped and recorded in metadata
- no train/val/test split is implemented yet
- `dataset.yaml` points both `train` and `val` to `images/` for temporary Ultralytics compatibility only

Validation:

- backend runs `validate_yolo_dataset(export_root)` automatically after export
- it checks:
  - required files/directories exist
  - image/label correspondence is complete
  - every label line has exactly 5 values
  - normalized coordinates are valid
  - metadata counts match exported files
  - generated config matches the Sport-CV compatibility mode

API endpoints:

- `POST /videos/{video_id}/export-yolo-dataset`
- `GET /videos/{video_id}/yolo-exports`
- `GET /yolo-exports/{export_id}/download`

Frontend testing:

1. Save at least one non-`unclear` annotation
2. Click `Export YOLO Dataset`
3. Confirm the panel shows:
   - `export_id`
   - `dataset_format`
   - `target_project`
   - `detected_framework`
   - `detected_model_version`
   - image/annotation/label counts
   - skipped counters
   - `sport_cv_compatible`
   - `export_path`
   - `download_path`
4. Confirm the available YOLO export list refreshes

Backend testing:

1. Run `POST /videos/{video_id}/export-yolo-dataset`
2. Open the generated folder in `data/exports/yolo/`
3. Check `images/` and `labels/`
4. Open `dataset.yaml`
5. Open `classes.txt`
6. Open `metadata.json`
7. Call `GET /videos/{video_id}/yolo-exports`
8. Call `GET /yolo-exports/{export_id}/download`

Sport-CV usage note:

- the export is framework-compatible with Sport-CV's Ultralytics detector stack
- future integration into Sport-CV's current multi-class/player pipeline requires explicit class-mapping and retraining work

## Step 13 Multi-Video Dataset Versioning + Group-Aware Split

Purpose:

- build immutable multi-video dataset snapshots directly from database annotations
- prevent leakage between train/val/test for video data
- make dataset builds deterministic and reproducible
- preserve full provenance and split diagnostics

Design docs:

- `docs/dataset_versioning_design.md`
- `docs/sport_cv_yolo_compatibility.md`

Why random frame splitting is unsafe:

- adjacent video frames are highly correlated
- one player track may persist across many frames
- splitting individual frames can leak temporal context from train into val/test

Supported split strategies:

- `video` (default, recommended)
  - one source video stays in one split
- `track`
  - resolved track/frame groups stay together
- `temporal_block`
  - contiguous frame blocks stay together

Determinism:

- builds are deterministic for the same:
  - source annotations
  - source videos
  - split strategy
  - ratios
  - random seed
  - temporal block size

Source fingerprint:

- every dataset version stores `source_data_fingerprint`
- rebuild is allowed only if the current database fingerprint matches the original one
- if annotations changed, rebuild is refused and a new dataset version must be created

Output structure:

- `data/datasets/versions/{version_name}/`
- contains:
  - `images/train`
  - `images/val`
  - `images/test`
  - `labels/train`
  - `labels/val`
  - `labels/test`
  - `dataset.yaml`
  - `classes.txt`
  - `manifest.json`
  - `metadata.json`
  - `split_report.json`

Manifest / split report:

- `manifest.json` traces every exported image and label back to source video/frame/annotation/prediction ids
- `split_report.json` records actual ratios, group counts, source-video distribution, conflict resolutions, and leakage validation results

Validation:

- dataset versions run filesystem + label validation
- leakage validation checks:
  - no canonical frame in multiple splits
  - no source annotation in multiple splits
  - no source prediction in multiple splits
  - no orphan images or labels
  - split-strategy-specific leakage rules

API endpoints:

- `POST /dataset-versions`
- `GET /dataset-versions`
- `GET /dataset-versions/{dataset_version_id}`
- `GET /dataset-versions/{dataset_version_id}/manifest`
- `GET /dataset-versions/{dataset_version_id}/split-report`
- `GET /dataset-versions/{dataset_version_id}/download`
- `POST /dataset-versions/{dataset_version_id}/rebuild`

Frontend testing:

1. Select multiple annotated videos
2. Fill `Dataset Versions` form
3. Keep default `video` split unless you are specifically testing `track` or `temporal_block`
4. Click `Create Dataset Version`
5. Check returned:
   - status
   - source videos
   - fingerprint
   - train/val/test counts
   - validation result
6. Use:
   - `View Details`
   - `View Manifest`
   - `View Split Report`
   - `Download Dataset`
   - `Rebuild Dataset`

Backend testing:

1. Call `POST /dataset-versions`
2. Open `data/datasets/versions/{version_name}/`
3. Inspect `dataset.yaml`, `manifest.json`, `metadata.json`, `split_report.json`
4. Call `GET /dataset-versions`
5. Call `GET /dataset-versions/{id}`
6. Call `GET /dataset-versions/{id}/manifest`
7. Call `GET /dataset-versions/{id}/split-report`
8. Call `POST /dataset-versions/{id}/rebuild`

Difference from Step 12:

- Step 12 exports a single-video YOLO dataset snapshot
- Step 13 exports a versioned, multi-video, leakage-aware dataset with provenance, validation, and rebuild checks

## Troubleshooting

- If `GET /videos/{video_id}/annotation-candidates` returns `500`, check the backend traceback first.
- Also inspect older SQLite rows in `video_frames` and `jersey_predictions`; older rows with null frame metadata or stale frame relationships can break candidate serialization if they are not repaired or skipped.
