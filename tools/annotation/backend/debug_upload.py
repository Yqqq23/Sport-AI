from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path

import requests
import uvicorn

from app.main import app


VIDEO_PATH = Path(r"D:\25-26香港教育大学\继续卷output。。。\2526shitu\jersey-active-learning\test_video\3 v 3 Basketball Game Side A 24022026_7s_clip.mp4")
ENDPOINT = "http://127.0.0.1:8000/upload-video"
TIMEOUT_SECONDS = 60
def is_port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        return sock.connect_ex((host, port)) == 0


def wait_for_server(host: str = "127.0.0.1", port: int = 8000, timeout: float = 15.0) -> bool:
    started = time.perf_counter()
    while time.perf_counter() - started < timeout:
        if is_port_open(host, port):
            return True
        time.sleep(0.5)
    return False


def main() -> None:
    started = time.perf_counter()
    print(f"Uploading: {VIDEO_PATH}")
    print(f"Endpoint: {ENDPOINT}")
    print(f"Timeout: {TIMEOUT_SECONDS}s")

    if not VIDEO_PATH.exists():
        print("Request succeeds: False")
        print("Status code: N/A")
        print(f"Error: file not found: {VIDEO_PATH}")
        print(f"Elapsed time: {time.perf_counter() - started:.2f}s")
        return

    server = None
    server_thread = None
    try:
        if not is_port_open("127.0.0.1", 8000):
            print("Server not detected on 127.0.0.1:8000, starting local uvicorn thread...")
            config = uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="info")
            server = uvicorn.Server(config)
            server_thread = threading.Thread(target=server.run, daemon=True)
            server_thread.start()
            if not wait_for_server():
                raise RuntimeError("Local FastAPI server failed to start within 15 seconds.")

        with VIDEO_PATH.open("rb") as stream:
            response = requests.post(
                ENDPOINT,
                files={"file": (VIDEO_PATH.name, stream, "video/mp4")},
                data={"frame_interval": "30"},
                timeout=TIMEOUT_SECONDS,
            )
        elapsed = time.perf_counter() - started
        print(f"Request succeeds: {response.ok}")
        print(f"Status code: {response.status_code}")
        try:
            print("Response JSON:")
            print(json.dumps(response.json(), ensure_ascii=False, indent=2))
        except Exception:
            print("Response Text:")
            print(response.text)
        print(f"Elapsed time: {elapsed:.2f}s")
    except requests.Timeout as exc:
        elapsed = time.perf_counter() - started
        print("Request succeeds: False")
        print("Status code: timeout")
        print(f"Error: {exc}")
        print(f"Elapsed time: {elapsed:.2f}s")
    except Exception as exc:
        elapsed = time.perf_counter() - started
        print("Request succeeds: False")
        print("Status code: error")
        print(f"Error: {exc}")
        print(f"Elapsed time: {elapsed:.2f}s")
    finally:
        if server is not None:
            server.should_exit = True
        if server_thread is not None:
            server_thread.join(timeout=5)


if __name__ == "__main__":
    main()
