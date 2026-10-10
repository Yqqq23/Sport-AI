"""Render selected jersey numbers without running inference again."""
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np


def transcode(source: Path, target: Path) -> None:
    temporary = target.with_suffix(".partial.mp4")
    result = subprocess.run([
        "ffmpeg", "-nostdin", "-y", "-i", str(source), "-an",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264",
        "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(temporary),
    ], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if result.returncode:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("FFmpeg conversion failed: " + result.stderr[-2000:])
    temporary.replace(target)


def roster(output: Path) -> list[dict]:
    counts = {}
    with (output / "trajectories.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            numbers = {p["number"] for p in json.loads(line)["players"] if p["number"] is not None}
            for number in numbers:
                counts[number] = counts.get(number, 0) + 1
    return [{"number": n, "frames": counts[n]} for n in sorted(counts, key=lambda n: (int(n), n))]


def render_selected(source: Path, output: Path, number: str, target: Path) -> None:
    meta = json.loads((output / "trajectory_meta.json").read_text(encoding="utf-8"))
    template = cv2.imread(str(output / "court_template.png"))
    if template is None:
        raise RuntimeError("Court template is missing")
    capture = cv2.VideoCapture(str(source))
    height = 720
    left_width = max(2, round(meta["width"] * height / meta["height"] / 2) * 2)
    right_width = max(2, round(template.shape[1] * height / template.shape[0] / 2) * 2)
    raw = target.with_suffix(".raw.mp4")
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), meta["fps"],
                             (left_width + right_width, height))
    if not capture.isOpened() or not writer.isOpened():
        capture.release()
        writer.release()
        raise RuntimeError("Cannot open selected-player video input/output")
    route = template.copy()
    previous = {}
    color = (60, 240, 255)
    try:
        with (output / "trajectories.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("Source video ended before trajectory data")
                selected = [p for p in record["players"] if p["number"] == number]
                view = (frame * 0.4).astype(np.uint8)
                points = []
                for player in selected:
                    x1, y1, x2, y2 = [int(v) for v in player["bbox"]]
                    x1, x2 = np.clip([x1, x2], 0, frame.shape[1])
                    y1, y2 = np.clip([y1, y2], 0, frame.shape[0])
                    view[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
                    cv2.rectangle(view, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(view, f"#{number}", (x1, max(25, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    if player["court"] is not None:
                        point = tuple(int(v) for v in player["court"])
                        if not (0 <= point[0] < route.shape[1] and 0 <= point[1] < route.shape[0]):
                            continue
                        last = previous.get(player["key"])
                        if last and last[0] == record["frame"] - 1:
                            cv2.line(route, last[1], point, color, 3, cv2.LINE_AA)
                        previous[player["key"]] = (record["frame"], point)
                        points.append(point)
                court = route.copy()
                for point in points:
                    cv2.circle(court, point, 10, (30, 80, 240), -1)
                label = f"#{number}  |  {record['frame'] / meta['fps']:.1f}s"
                if not selected:
                    label += "  |  Not observed / number unknown"
                cv2.putText(view, label, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                writer.write(np.hstack((cv2.resize(view, (left_width, height)),
                                        cv2.resize(court, (right_width, height)))))
    finally:
        capture.release()
        writer.release()
    try:
        transcode(raw, target)
    finally:
        raw.unlink(missing_ok=True)
