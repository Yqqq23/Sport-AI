"""Run from repository root: python -m uvicorn webapp.app:app --workers 1."""
import json
import os
import re
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from webapp.video import render_selected, roster, transcode

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("BASKETBALL_WEB_DATA", str(ROOT / "outputs" / "webapp"))).resolve()
CONFIG = Path(os.environ.get("BASKETBALL_CONFIG", str(ROOT / "configs" / "default.yaml"))).resolve()
MAX_FRAMES = int(os.environ.get("BASKETBALL_MAX_FRAMES", "1800"))
MAX_UPLOAD = int(os.environ.get("BASKETBALL_MAX_UPLOAD_MB", "2048")) * 1024 * 1024
if MAX_FRAMES <= 0 or MAX_UPLOAD <= 0:
    raise ValueError("BASKETBALL_MAX_FRAMES and BASKETBALL_MAX_UPLOAD_MB must be positive")
LOCK = threading.RLock()


def job_dir(job_id: str) -> Path:
    if not re.fullmatch(r"[a-f0-9]{32}", job_id):
        raise HTTPException(404, "任务不存在")
    directory = DATA / job_id
    if not (directory / "job.json").is_file():
        raise HTTPException(404, "任务不存在")
    return directory


def load_job(directory: Path) -> dict:
    with LOCK:
        return json.loads((directory / "job.json").read_text(encoding="utf-8"))


def save_job(directory: Path, job: dict) -> None:
    with LOCK:
        temporary = directory / "job.tmp"
        temporary.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
        temporary.replace(directory / "job.json")


def update_job(directory: Path, **changes) -> None:
    with LOCK:
        job = load_job(directory)
        job.update(changes)
        save_job(directory, job)


def run_analysis(directory: Path) -> None:
    try:
        job = load_job(directory)
        update_job(directory, status="processing", progress=0)
        output = directory / "analysis"
        command = [sys.executable, "-u", str(ROOT / "src" / "pipeline.py"),
                   "--input", str(directory / job["source"]), "--config", str(CONFIG),
                   "--output-dir", str(output), "--max-frames", str(MAX_FRAMES),
                   "--jersey-backend", "perdigit", "--export-trajectories"]
        with (directory / "pipeline.log").open("w", encoding="utf-8") as log:
            with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                  errors="replace") as process:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    match = re.search(r"pipeline: frame (\d+)", line)
                    # Direct script logging uses __main__ rather than pipeline.
                    if match is None:
                        match = re.search(r"__main__: frame (\d+)", line)
                    if match:
                        update_job(directory, progress=min(89, int(90 * (int(match[1]) + 1) / job["total_frames"])))
                if process.wait() != 0:
                    raise RuntimeError("分析失败，请查看任务日志（模型、标定或依赖错误）。")
        update_job(directory, status="encoding", progress=90)
        meta = json.loads((output / "trajectory_meta.json").read_text(encoding="utf-8"))
        for kind in ("annotated", "topdown"):
            transcode(output / meta[kind], directory / f"{kind}.mp4")
        update_job(directory, status="completed", progress=100, players=roster(output),
                   calibrated=meta["calibrated"])
    except Exception as exc:
        update_job(directory, status="failed", error=str(exc))


def run_selection(directory: Path, number: str) -> None:
    try:
        job = load_job(directory)
        render_selected(directory / job["source"], directory / "analysis", number,
                        directory / f"player_{number}.mp4")
        result = {"status": "completed"}
    except Exception as exc:
        result = {"status": "failed", "error": str(exc)}
    with LOCK:
        job = load_job(directory)
        job["selections"][number] = result
        save_job(directory, job)


@asynccontextmanager
async def lifespan(app: FastAPI):
    DATA.mkdir(parents=True, exist_ok=True)
    # Interrupted jobs remain inspectable instead of polling forever after restart.
    for state in DATA.glob("*/job.json"):
        job = load_job(state.parent)
        if job["status"] in {"queued", "processing", "encoding"}:
            job.update(status="failed", error="服务重启中断了任务，请重新上传。")
        for selection in job.get("selections", {}).values():
            if selection["status"] == "processing":
                selection.update(status="failed", error="服务重启中断了生成，请重试。")
        save_job(state.parent, job)
    app.state.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="basketball")
    yield
    app.state.executor.shutdown(wait=True)


app = FastAPI(title="Basketball Video Analysis", lifespan=lifespan)


@app.get("/api/health")
def health():
    return {"ffmpeg": shutil.which("ffmpeg") is not None, "max_frames": MAX_FRAMES,
            "config_exists": CONFIG.is_file(), "max_upload_mb": MAX_UPLOAD // (1024 * 1024)}


@app.post("/api/jobs", status_code=202)
def upload_video(video: UploadFile = File(...)):
    if not shutil.which("ffmpeg"):
        raise HTTPException(503, "服务器尚未安装 FFmpeg，请先安装后上传。")
    suffix = Path(video.filename or "").suffix.lower()
    if suffix not in {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}:
        raise HTTPException(400, "请选择 MP4 / MOV / AVI / MKV / WebM 视频")
    identifier = uuid4().hex
    directory = DATA / identifier
    directory.mkdir(parents=True)
    source = directory / f"input{suffix}"
    try:
        size = 0
        with source.open("wb") as destination:
            while chunk := video.file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD:
                    raise HTTPException(413, f"视频超过 {MAX_UPLOAD // (1024 * 1024)} MB")
                destination.write(chunk)
        capture = cv2.VideoCapture(str(source))
        try:
            valid, _ = capture.read()
            frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            capture.release()
        if not valid:
            raise HTTPException(400, "视频无法解码，请检查文件")
        job = {"id": identifier, "filename": video.filename, "source": source.name,
               "created_at": datetime.now(timezone.utc).isoformat(), "status": "queued",
               "progress": 0, "total_frames": min(frames, MAX_FRAMES) if frames > 0 else MAX_FRAMES,
               "players": [], "selections": {}, "error": None}
        save_job(directory, job)
        app.state.executor.submit(run_analysis, directory)
        return public_job(job)
    except Exception:
        shutil.rmtree(directory)
        raise
    finally:
        video.file.close()


def public_job(job: dict) -> dict:
    result = {key: value for key, value in job.items() if key != "source"}
    base = f"/api/jobs/{job['id']}"
    result["log_url"] = base + "/log"
    result["videos"] = {kind: f"{base}/videos/{kind}" for kind in ("annotated", "topdown")} if job["status"] == "completed" else {}
    result["selections"] = {number: {**state, "url": f"{base}/videos/player_{number}" if state["status"] == "completed" else None}
                            for number, state in job["selections"].items()}
    return result


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    return public_job(load_job(job_dir(job_id)))


@app.post("/api/jobs/{job_id}/players/{number}", status_code=202)
def select_player(job_id: str, number: str):
    directory = job_dir(job_id)
    with LOCK:
        job = load_job(directory)
        if job["status"] != "completed":
            raise HTTPException(409, "请等待视频分析完成")
        if not re.fullmatch(r"[0-9]{1,3}", number) or number not in {p["number"] for p in job["players"]}:
            raise HTTPException(404, "未识别到该球衣号码")
        if job["selections"].get(number, {}).get("status") not in {"processing", "completed"}:
            job["selections"][number] = {"status": "processing"}
            save_job(directory, job)
            app.state.executor.submit(run_selection, directory, number)
        return public_job(job)


@app.get("/api/jobs/{job_id}/videos/{kind}")
def get_video(job_id: str, kind: str):
    directory = job_dir(job_id)
    if kind not in {"topdown", "annotated"} and not re.fullmatch(r"player_[0-9]{1,3}", kind):
        raise HTTPException(404, "视频不存在")
    path = directory / f"{kind}.mp4"
    if not path.is_file():
        raise HTTPException(404, "视频尚未生成")
    return FileResponse(path, media_type="video/mp4")


@app.get("/api/jobs/{job_id}/log")
def get_log(job_id: str):
    path = job_dir(job_id) / "pipeline.log"
    if not path.is_file():
        raise HTTPException(404, "任务尚未开始")
    return FileResponse(path, media_type="text/plain; charset=utf-8")


app.mount("/", StaticFiles(directory=Path(__file__).parent / "frontend", html=True), name="frontend")
