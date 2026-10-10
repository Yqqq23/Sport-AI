from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.config import settings


FIXED7_SUBDIR = "fixed7_jersey_tracking_ocr7"
FIXED7_SCRIPT_NAME = "run_basketball_pose_tracking_fixed7_ocr7_stable.py"


@dataclass
class Fixed7AdapterError(RuntimeError):
    error: str
    missing_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {"ok": False, "error": self.error}
        if self.missing_path:
            payload["missing_path"] = self.missing_path
        return payload


def fixed7_root() -> Path:
    return (settings.sport_cv_root / FIXED7_SUBDIR).resolve()


def fixed7_main_script() -> Path:
    return (fixed7_root() / FIXED7_SCRIPT_NAME).resolve()


def fixed7_default_detector_weights() -> Path:
    return (fixed7_root() / "models" / "rf_detr_player_number.pt").resolve()


def estimate_jersey_bbox(player_bbox: list[float]) -> list[int]:
    x1, y1, x2, y2 = [int(round(v)) for v in player_bbox]
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    jersey = [
        x1 + int(width * 0.2),
        y1 + int(height * 0.14),
        x2 - int(width * 0.2),
        y1 + int(height * 0.58),
    ]
    jersey[2] = max(jersey[0] + 1, jersey[2])
    jersey[3] = max(jersey[1] + 1, jersey[3])
    return jersey


def parse_fixed7_tracking_results(results_path: str | Path, source_frame_dir: str | Path | None = None) -> list[dict[str, Any]]:
    path = Path(results_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    output: list[dict[str, Any]] = []
    source_dir = Path(source_frame_dir) if source_frame_dir else None
    for frame_record in payload:
        frame_id = int(frame_record.get("frame_idx", 0))
        source_frame_path = None
        if source_dir is not None:
            source_frame_path = str((source_dir / f"frame_{frame_id:06d}.jpg").resolve())
        for player in frame_record.get("players", []):
            player_bbox = [int(round(v)) for v in player.get("bbox_xyxy") or player.get("raw_bbox_xyxy") or []]
            if len(player_bbox) != 4:
                continue
            number_bbox = player.get("number_bbox")
            jersey_bbox = [int(round(v)) for v in number_bbox] if isinstance(number_bbox, list) and len(number_bbox) == 4 else estimate_jersey_bbox(player_bbox)
            predicted_number = player.get("number_text") or ""
            confidence = float(player.get("number_conf") if predicted_number else player.get("player_conf", 0.0) or 0.0)
            output.append(
                {
                    "frame_id": frame_id,
                    "track_id": int(player.get("track_id", player.get("fixed_slot_id", 0)) or 0),
                    "player_bbox": player_bbox,
                    "jersey_bbox": jersey_bbox,
                    "predicted_number": predicted_number,
                    "confidence": confidence,
                    "source_frame_path": source_frame_path,
                }
            )
    return output


def run_fixed7_on_video(video_path: str, output_dir: str) -> list[dict[str, Any]]:
    script_path = fixed7_main_script()
    if not fixed7_root().exists():
        raise Fixed7AdapterError("fixed7 folder missing", str(fixed7_root()))
    if not script_path.exists():
        raise Fixed7AdapterError("fixed7 main script missing", str(script_path))

    detector_weights = fixed7_default_detector_weights()
    if not detector_weights.exists():
        raise Fixed7AdapterError("fixed7 detector weights missing", str(detector_weights))

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(script_path),
        "--input",
        str(Path(video_path).resolve()),
        "--output_dir",
        str(output_root.resolve()),
        "--tracker_backend",
        "bytetrack",
        "--detector_backend",
        "rf_detr",
        "--detector_weights",
        str(detector_weights),
        "--pose_config",
        "",
        "--pose_checkpoint",
        "",
        "--ball_model_path",
        "",
        "--max_frames",
        "10",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise Fixed7AdapterError(
            f"fixed7 run failed: {completed.stderr.strip() or completed.stdout.strip() or 'unknown error'}"
        )

    results_path = output_root / "tracking_pose_results.json"
    if not results_path.exists():
        raise Fixed7AdapterError("fixed7 output missing tracking_pose_results.json", str(results_path))
    return parse_fixed7_tracking_results(results_path)


def fixed7_diagnostic_report() -> dict[str, Any]:
    root = fixed7_root()
    script = fixed7_main_script()
    detector_weights = fixed7_default_detector_weights()
    sample_json = next(root.rglob("tracking_pose_results.json"), None) if root.exists() else None

    try:
        import pytesseract  # type: ignore

        _ = pytesseract.get_tesseract_version()
        tesseract_available = True
    except Exception:
        tesseract_available = False

    try:
        import torch  # type: ignore

        torch_status = {"ok": True, "cuda_available": bool(torch.cuda.is_available())}
    except Exception as exc:  # noqa: BLE001
        torch_status = {"ok": False, "error": str(exc)}

    try:
        import ultralytics  # type: ignore

        ultralytics_status = {"ok": True, "version": getattr(ultralytics, '__version__', 'unknown')}
    except Exception as exc:  # noqa: BLE001
        ultralytics_status = {"ok": False, "error": str(exc)}

    roboflow_detected = False
    if script.exists():
        text = script.read_text(encoding="utf-8", errors="ignore")
        roboflow_detected = "roboflow_workflow" in text or "InferenceHTTPClient" in text

    schema_detected = False
    schema_fields: list[str] = []
    if sample_json is not None:
        try:
            payload = json.loads(sample_json.read_text(encoding="utf-8"))
            if payload and isinstance(payload, list):
                first = payload[0]
                schema_detected = isinstance(first, dict) and "players" in first
                if schema_detected and first.get("players"):
                    schema_fields = sorted(first["players"][0].keys())
        except Exception:
            schema_detected = False

    return {
        "fixed7_folder_exists": root.exists(),
        "main_script_exists": script.exists(),
        "default_detector_weights_exists": detector_weights.exists(),
        "default_detector_weights_path": str(detector_weights),
        "tracking_pose_results_schema_detected": schema_detected,
        "sample_schema_fields": schema_fields,
        "tesseract_available": tesseract_available,
        "roboflow_config_detected": roboflow_detected,
        "torch_status": torch_status,
        "ultralytics_status": ultralytics_status,
    }
