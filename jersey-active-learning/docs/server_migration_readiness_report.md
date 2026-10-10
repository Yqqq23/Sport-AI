# Server Migration Readiness Report

## Overall verdict

- Can this project be migrated directly? **Partially**
- Migration readiness: **partially ready**

The project is structurally portable to Linux, but it is **not ready for direct copy-and-run deployment without environment configuration changes**.

The main blockers are:

1. Windows-specific default paths in backend config
2. frontend default API URL hardcoded to `http://127.0.0.1:8000`
3. current PyTorch requirement is CPU-only, which is not ideal for a GPU server
4. SQLite is acceptable for one or two RAs, but risky for concurrent multi-user annotation/write workloads
5. long-running YOLO jobs are currently tied to the FastAPI app process rather than a robust worker queue system

## 1. Backend portability assessment

### Python and dependency requirements

Current backend dependencies from `backend/requirements.txt`:

- `fastapi==0.116.1`
- `uvicorn[standard]==0.35.0`
- `sqlalchemy==2.0.43`
- `pydantic==2.11.7`
- `pydantic-settings==2.10.1`
- `python-multipart==0.0.20`
- `python-dotenv==1.0.1`
- `numpy==2.1.1`
- `opencv-python==4.10.0.84`
- `torch==2.5.1+cpu`
- `torchvision==0.20.1+cpu`
- `ultralytics==8.3.39`
- `pytest==8.4.2`

Implication:

- this is **Python 3.11-oriented modern stack**
- backend should be deployed on a Linux machine with Python 3.10 or 3.11 preferred
- current requirements pin **CPU-only PyTorch**, so GPU acceleration is not actually enabled by the current requirements file

### Windows-specific path assumptions

Main blocker found in `backend/app/core/config.py`:

- `SPORT_CV_ROOT` default is:
  - `D:\25-26香港教育大学\继续卷output。。。\Sport-CV`

This will not work on Linux.

This must be overridden on the server, for example:

- `SPORT_CV_ROOT=/home/<user>/Sport-CV`

Other path behavior:

- `DATA_DIR` default is `../data`
- `DATABASE_URL` default is `sqlite:///./jersey_active_learning.db`

These are relative paths and are portable **if the backend is started from the backend directory consistently**.

### File storage portability

Storage layout from `backend/app/services/storage.py` is portable:

- `data/videos/`
- `data/frames/`
- `data/crops/`
- `data/annotations/`
- `data/exports/`

It uses `pathlib.Path`, which is Linux-safe.

### SQLite database path

Current default:

- `sqlite:///./jersey_active_learning.db`

This is portable, but it depends on the current working directory.

Safer server setting would be something explicit like:

- `DATABASE_URL=sqlite:////home/<user>/jersey-active-learning/backend/jersey_active_learning.db`

### Path handling quality

Overall path handling is mostly portable:

- `pathlib.Path` is used widely
- static files use resolved `data_dir`
- frame and export paths are composed by `Path`

### Static image serving on Linux

From `backend/app/main.py`:

- `/static/frames` is mounted from `settings.data_dir / "frames"`

This should work on Linux as long as:

- `DATA_DIR` is correct
- extracted frame files exist
- process user has read permission

## 2. Frontend portability assessment

### package.json / Vite

Frontend is simple and portable:

- React 18
- Vite 5

No Windows-only build dependency was found.

### API base URL

Major migration issue found in `frontend/src/api/client.js`:

- default API base URL is hardcoded to:
  - `http://127.0.0.1:8000`

On the server, this will only work if the browser is on the same machine, which it is not.

Server deployment will require:

- `VITE_API_BASE_URL=http://<server-ip>:8000`

or better:

- proxied same-origin deployment behind a reverse proxy

### localhost / 127.0.0.1 hardcoding

Also in backend config:

- `CORS_ORIGINS` defaults to local Vite dev origins only:
  - `http://127.0.0.1:5173`
  - `http://localhost:5173`

This must be changed for server access.

### Windows path display issues

The UI displays some raw backend paths in debug/export panels. These are not blockers, but Linux absolute paths will appear in the UI if those advanced panels are opened.

### Does frontend need `.env` on server?

Yes, practically.

At minimum for frontend build/runtime:

- `VITE_API_BASE_URL`

## 3. YOLO pre-annotation portability assessment

### Ultralytics dependency

Yes, backend depends on `ultralytics` for the MVP pre-annotation flow.

### Auto-download behavior

`backend/app/services/yolo_preannotation.py` tries models in this order:

1. `PLAYER_DETECTOR_WEIGHTS` if configured
2. `yolo11n.pt`
3. `yolov8n.pt`

Implication:

- if no custom path is set, it expects Ultralytics pretrained model loading behavior
- if those weights are not already cached, **internet access may be needed** for first-run auto-download

### Internet requirement

Potentially yes.

If the EdUHK server cannot access Ultralytics model download sources, YOLO pre-annotation may fail on first use unless weights are manually preloaded.

### CPU/GPU fallback

Current code does **not explicitly manage GPU selection** in the MVP pre-annotation service.

Also, current requirements install:

- `torch==2.5.1+cpu`
- `torchvision==0.20.1+cpu`

So the current project is actually configured for **CPU-only inference**, not Linux GPU usage.

### Missing ultralytics error clarity

Yes.

If Ultralytics cannot be imported, backend returns a clear error like:

- `YOLO unavailable: ...`

This part is acceptable.

## 4. Deployment risks

### Top 5 risks

1. **Wrong path configuration**
   - `SPORT_CV_ROOT` default is Windows-only
2. **Frontend cannot reach backend**
   - hardcoded `127.0.0.1:8000` default
3. **SQLite write concurrency risk**
   - multiple simultaneous RA writes may cause locking/contention
4. **Large disk usage**
   - uploaded videos, extracted frames, exports, and versioned datasets can grow quickly
5. **Long-running process stability**
   - async recognition/export jobs run inside the FastAPI process rather than a production worker queue

### Additional risks

- CORS misconfiguration
- port conflicts on `8000`, `8001`, `5173`
- browser upload size / reverse proxy body-size limits if later fronted by nginx
- file permission issues under shared Linux user directories

## 5. Server requirements checklist

Run these exact commands on the server:

```bash
python3 --version
node --version
npm --version
nvidia-smi
df -h
hostname -I
ss -ltnp | grep 8001
ss -ltnp | grep 5173
```

Recommended extra checks:

```bash
which python3
which node
which npm
pwd
free -h
```

## 6. Migration strategy recommendation

### Ranked by safety / effort

1. **A. simple manual deployment**
   - safest first test
   - lowest moving parts
   - best for proving the app can run on the server

2. **B. start scripts**
   - next best step after manual verification
   - good for repeatability

3. **D. reverse proxy**
   - recommended after backend/frontend are confirmed working
   - needed for cleaner public/internal access

4. **E. cloud tunnel**
   - useful if no reverse proxy / public routing is available
   - more operational complexity

5. **C. Docker**
   - highest setup effort for current project state
   - especially awkward with GPU, Torch, OpenCV, and data persistence if the team has not containerized this stack before

### Safest recommended approach

Start with:

- manual backend + frontend deployment under one Linux user
- test with one small video
- only after success decide whether to add reverse proxy / service management

## 7. Recommended server folder structure

Recommended:

```text
~/jersey-active-learning/
~/jersey-active-learning/backend/
~/jersey-active-learning/frontend/
~/jersey-active-learning/data/
```

And inside `data/`:

```text
data/videos/
data/frames/
data/crops/
data/annotations/
data/exports/
data/datasets/
```

## 8. Data migration guidance

### Should copy

- SQLite database file if you want existing metadata, predictions, annotations, and dataset version records
- `data/videos/` if you want uploaded source videos preserved
- `data/frames/` if you want existing extracted frames preserved
- `data/exports/` if you want old export artifacts preserved
- `data/datasets/` if you want versioned dataset outputs preserved

### Mandatory vs optional

Mandatory for preserving work:

- `backend/jersey_active_learning.db` (or wherever SQLite DB currently lives)
- `data/videos/`
- `data/frames/`

Strongly recommended if you want full history:

- `data/exports/`
- `data/datasets/`

If you do **not** copy the DB:

- existing annotations / metadata / progress / predictions are lost
- copied frames/videos alone are not enough to preserve state

## 9. What must be changed before migration

Must change:

1. backend environment variables
   - `SPORT_CV_ROOT`
   - likely `DATA_DIR`
   - likely `DATABASE_URL`
2. frontend API base URL
   - `VITE_API_BASE_URL`
3. backend CORS origins
   - `CORS_ORIGINS`
4. PyTorch/Ultralytics installation strategy
   - current requirements are CPU-only and not aligned with Linux GPU use

## 10. What should not be changed

Do not change before first migration test:

- database schema
- export logic
- annotation data format
- review-item API semantics
- file layout assumptions under `data/`

The first server deployment should focus on environment parity, not feature refactoring.

## 11. Dependencies that must be installed on server

Backend:

- Python 3.10/3.11
- pip / venv
- project requirements from `backend/requirements.txt`
- likely system packages for OpenCV runtime if needed by the server OS

Frontend:

- Node.js
- npm

Possible additional server-level runtime needs:

- ffmpeg may be useful later, though not currently required by the Python code shown
- GPU driver / CUDA only if GPU acceleration is desired later

## 12. Environment variables needed on server

Recommended minimum backend env:

```env
APP_ENV=production
DATABASE_URL=sqlite:////home/<user>/jersey-active-learning/backend/jersey_active_learning.db
DATA_DIR=/home/<user>/jersey-active-learning/data
CORS_ORIGINS=http://<server-ip>:5173,http://<server-ip>:8001
REAL_RECOGNITION_ENABLED=true
SPORT_CV_ROOT=/home/<user>/Sport-CV
PLAYER_DETECTOR_WEIGHTS=
JERSEY_OCR_WEIGHTS=
```

Recommended frontend env:

```env
VITE_API_BASE_URL=http://<server-ip>:8001
```

## 13. Ports recommendation

Recommended:

- backend: `8001`
- frontend preview/dev: `5173`

Why not default `8000`?

- `8000` is common and may already be occupied on shared servers
- `8001` is a safer first test port

## 14. Recommended safest first deployment test

1. copy project code to `~/jersey-active-learning`
2. copy a **single small test video** only
3. set Linux-safe `.env` values
4. start backend on `8001`
5. start frontend on `5173`
6. open site from browser via server IP
7. upload one small video
8. extract frames
9. test manual annotation path first
10. only then test YOLO pre-annotation

This is safer than testing GPU / OCR / advanced export first.

## 15. Rollback plan

If migration fails:

1. stop backend/frontend processes
2. do not overwrite local project state
3. keep original local DB and `data/` as the authoritative copy
4. remove only the server-side copied test deployment directory
5. reattempt with a smaller test scope (manual annotation only)

## Final recommendation

This project is **partially ready** for migration.

It is suitable for a careful staged Linux deployment, but not for blind direct copy-and-run.

Safest next action:

- prepare a Linux-specific `.env`
- plan a **manual single-video first deployment test** on port `8001`
- do not attempt GPU or public multi-user deployment on the first try
