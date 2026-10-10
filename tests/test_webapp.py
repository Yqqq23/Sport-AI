import importlib
import io
import json
import shutil
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

backend = importlib.import_module("webapp.app")
video = importlib.import_module("webapp.video")


def make_video(path, frames=3):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
    assert writer.isOpened()
    for _ in range(frames):
        writer.write(np.full((120, 160, 3), 200, dtype=np.uint8))
    writer.release()


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(backend, "DATA", tmp_path / "jobs")
    monkeypatch.setattr(backend.shutil, "which", lambda _: "/usr/bin/ffmpeg")
    with TestClient(backend.app) as client:
        backend.app.state.executor.submit = Mock()
        yield client


def upload(client, tmp_path):
    source = tmp_path / "clip.mp4"
    make_video(source)
    response = client.post("/api/jobs", files={"video": ("clip.mp4", source.read_bytes(), "video/mp4")})
    assert response.status_code == 202, response.text
    return response.json()


def test_upload_isolation_validation_and_ui(client, tmp_path, monkeypatch):
    assert client.get("/").status_code == 200
    assert client.get("/app.js").status_code == 200
    first, second = upload(client, tmp_path), upload(client, tmp_path)
    assert first["id"] != second["id"]
    assert first["total_frames"] == 3
    assert client.get(f"/api/jobs/{first['id']}").json()["status"] == "queued"
    assert client.get("/api/jobs/invalid").status_code == 404
    assert client.post("/api/jobs", files={"video": ("bad.txt", b"abc")}).status_code == 400
    assert client.post("/api/jobs", files={"video": ("bad.mp4", b"abc")}).status_code == 400
    monkeypatch.setattr(backend, "MAX_UPLOAD", 2)
    assert client.post("/api/jobs", files={"video": ("large.mp4", b"abc")}).status_code == 413
    assert len(list(backend.DATA.iterdir())) == 2


def test_selection_cache_and_range(client, tmp_path):
    job = upload(client, tmp_path)
    base = f"/api/jobs/{job['id']}"
    directory = backend.job_dir(job["id"])
    assert client.post(base + "/players/23").status_code == 409
    backend.update_job(directory, status="completed", players=[{"number": "23", "frames": 3}])
    assert client.post(base + "/players/99").status_code == 404
    backend.app.state.executor.submit.reset_mock()
    for _ in range(2):
        assert client.post(base + "/players/23").json()["selections"]["23"]["status"] == "processing"
    assert backend.app.state.executor.submit.call_count == 1
    (directory / "topdown.mp4").write_bytes(b"0123456789")
    response = client.get(base + "/videos/topdown", headers={"Range": "bytes=2-5"})
    assert response.status_code == 206
    assert response.content == b"2345"
    assert client.get(base + "/videos/input").status_code == 404
    assert client.get(base + "/videos/player_23").status_code == 404


def test_pipeline_command_and_failure(client, tmp_path, monkeypatch):
    job = upload(client, tmp_path)
    directory = backend.job_dir(job["id"])
    commands = []

    class Process:
        stdout = io.StringIO("INFO __main__: frame 0\nmodel missing\n")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def wait(self):
            return 1

    def popen(command, **kwargs):
        commands.append((command, kwargs))
        return Process()

    monkeypatch.setattr(backend.subprocess, "Popen", popen)
    backend.run_analysis(directory)
    command, kwargs = commands[0]
    assert command[command.index("--jersey-backend") + 1] == "perdigit"
    assert command[command.index("--max-frames") + 1] == str(backend.MAX_FRAMES)
    assert "--export-trajectories" in command
    assert kwargs["cwd"] == backend.ROOT
    assert backend.load_job(directory)["status"] == "failed"
    assert "model missing" in (directory / "pipeline.log").read_text()


def test_selected_video_highlights_only_requested_number(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    make_video(source)
    (tmp_path / "trajectory_meta.json").write_text(json.dumps({"fps": 10, "width": 160, "height": 120}))
    cv2.imwrite(str(tmp_path / "court_template.png"), np.zeros((120, 120, 3), dtype=np.uint8))
    records = [{"frame": index, "players": [
        {"key": "motion:1", "number": "23", "bbox": [10, 20, 50, 80], "court": [20 + index * 10, 60]},
        {"key": "motion:2", "number": "7", "bbox": [100, 20, 140, 80], "court": [90, 60]},
    ]} for index in range(3)]
    (tmp_path / "trajectories.jsonl").write_text("\n".join(json.dumps(r) for r in records))
    assert video.roster(tmp_path) == [{"number": "7", "frames": 3}, {"number": "23", "frames": 3}]
    monkeypatch.setattr(video, "transcode", lambda source, target: shutil.copyfile(source, target))
    target = tmp_path / "player_23.mp4"
    video.render_selected(source, tmp_path, "23", target)
    capture = cv2.VideoCapture(str(target))
    try:
        assert capture.get(cv2.CAP_PROP_FRAME_COUNT) == 3
        ok, frame = capture.read()
        assert ok
        # Original is scaled by six. Selected torso stays bright; #7 is dimmed.
        assert frame[300, 180].mean() > 150
        assert frame[300, 720].mean() < 110
        assert frame.shape == (720, 1680, 3)
    finally:
        capture.release()
    assert not target.with_suffix(".raw.mp4").exists()


def test_restart_marks_interrupted_jobs(tmp_path, monkeypatch):
    monkeypatch.setattr(backend, "DATA", tmp_path)
    directory = tmp_path / ("a" * 32)
    directory.mkdir()
    backend.save_job(directory, {"status": "processing", "selections": {"7": {"status": "processing"}}})
    with TestClient(backend.app):
        job = backend.load_job(directory)
        assert job["status"] == "failed"
        assert job["selections"]["7"]["status"] == "failed"


def test_successful_analysis_publishes_roster_and_videos(client, tmp_path, monkeypatch):
    job = upload(client, tmp_path)
    directory = backend.job_dir(job["id"])
    output = directory / "analysis"
    output.mkdir()
    (output / "trajectory_meta.json").write_text(json.dumps({
        "annotated": "custom_annotated.mp4", "topdown": "custom_topdown.mp4", "calibrated": True,
    }))
    (output / "trajectories.jsonl").write_text(json.dumps({"frame": 0, "players": [{"number": "23"}]}))
    for name in ("custom_annotated.mp4", "custom_topdown.mp4"):
        (output / name).write_bytes(b"video")
    process = Mock()
    process.stdout = iter(["INFO __main__: frame 0\n"])
    process.wait.return_value = 0
    context = Mock()
    context.__enter__ = Mock(return_value=process)
    context.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(backend.subprocess, "Popen", Mock(return_value=context))
    monkeypatch.setattr(backend, "transcode", shutil.copyfile)
    backend.run_analysis(directory)
    result = client.get(f"/api/jobs/{job['id']}").json()
    assert result["status"] == "completed"
    assert result["progress"] == 100
    assert result["players"] == [{"number": "23", "frames": 1}]
    for url in result["videos"].values():
        assert client.get(url).content == b"video"
