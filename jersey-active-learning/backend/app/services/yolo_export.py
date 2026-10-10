import json
import math
from pathlib import Path
from typing import Any


SPORT_CV_DETECTED_FRAMEWORK = "Ultralytics YOLO"
SPORT_CV_DETECTED_MODEL_VERSION = "YOLO11-family weights via ultralytics 8.x"
SPORT_CV_TARGET_PROJECT = "Sport-CV"
YOLO_CLASS_MAPPING = {0: "jersey"}


def _parse_dataset_yaml(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    names: dict[int, str] = {}
    current_key: str | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if line.startswith("names:"):
            current_key = "names"
            continue
        if current_key == "names" and line.startswith("  ") and ":" in line:
            key_text, value_text = line.strip().split(":", 1)
            names[int(key_text.strip())] = value_text.strip().strip('"').strip("'")
            continue
        current_key = None
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip().strip('"').strip("'")
    if names:
        data["names"] = names
    if "nc" in data:
        try:
            data["nc"] = int(data["nc"])
        except ValueError:
            pass
    return data


def validate_yolo_dataset(export_root: str | Path) -> dict[str, object]:
    root = Path(export_root)
    errors: list[str] = []

    images_dir = root / "images"
    labels_dir = root / "labels"
    dataset_yaml = root / "dataset.yaml"
    classes_txt = root / "classes.txt"
    metadata_json = root / "metadata.json"

    for required_path, label in [
        (images_dir, "images directory"),
        (labels_dir, "labels directory"),
        (dataset_yaml, "dataset.yaml"),
        (classes_txt, "classes.txt"),
        (metadata_json, "metadata.json"),
    ]:
        if not required_path.exists():
            errors.append(f"missing {label}: {required_path}")

    if errors:
        return {"valid": False, "errors": errors}

    metadata = json.loads(metadata_json.read_text(encoding="utf-8"))
    dataset = _parse_dataset_yaml(dataset_yaml)
    classes = [line.strip() for line in classes_txt.read_text(encoding="utf-8").splitlines() if line.strip()]

    if classes != ["jersey"]:
        errors.append("classes.txt must contain exactly one class: jersey")
    if dataset.get("nc") != 1:
        errors.append("dataset.yaml nc must be 1")
    if dataset.get("names") != {0: "jersey"}:
        errors.append("dataset.yaml names must map 0 to jersey")

    image_files = sorted([path for path in images_dir.iterdir() if path.is_file()])
    label_files = sorted([path for path in labels_dir.iterdir() if path.is_file()])
    image_stems = {path.stem for path in image_files}
    label_stems = {path.stem for path in label_files}

    if image_stems != label_stems:
        missing_labels = sorted(image_stems - label_stems)
        missing_images = sorted(label_stems - image_stems)
        if missing_labels:
            errors.append(f"orphan images without labels: {missing_labels}")
        if missing_images:
            errors.append(f"orphan labels without images: {missing_images}")

    if len(image_files) != len(image_stems):
        errors.append("duplicate image filenames detected")
    if len(label_files) != len(label_stems):
        errors.append("duplicate label filenames detected")

    total_lines = 0
    for label_path in label_files:
        for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                errors.append(f"{label_path.name}:{line_number} must contain 5 values")
                continue
            try:
                class_id = int(parts[0])
                values = [float(item) for item in parts[1:]]
            except ValueError:
                errors.append(f"{label_path.name}:{line_number} contains non-numeric YOLO values")
                continue
            if class_id not in YOLO_CLASS_MAPPING:
                errors.append(f"{label_path.name}:{line_number} has invalid class_id {class_id}")
            if any(math.isnan(value) or math.isinf(value) for value in values):
                errors.append(f"{label_path.name}:{line_number} contains NaN or infinity")
            x_center, y_center, width, height = values
            if not (0.0 <= x_center <= 1.0):
                errors.append(f"{label_path.name}:{line_number} x_center out of range")
            if not (0.0 <= y_center <= 1.0):
                errors.append(f"{label_path.name}:{line_number} y_center out of range")
            if not (0.0 < width <= 1.0):
                errors.append(f"{label_path.name}:{line_number} width out of range")
            if not (0.0 < height <= 1.0):
                errors.append(f"{label_path.name}:{line_number} height out of range")
            total_lines += 1

    if int(metadata.get("image_count", 0)) != len(image_files):
        errors.append("metadata image_count does not match exported images")
    if int(metadata.get("label_file_count", 0)) != len(label_files):
        errors.append("metadata label_file_count does not match exported labels")
    if int(metadata.get("annotation_count", 0)) != total_lines:
        errors.append("metadata annotation_count does not match total YOLO label lines")
    if metadata.get("class_mapping") != {"0": "jersey"}:
        errors.append("metadata class_mapping must be {'0': 'jersey'}")
    if metadata.get("detected_framework") != SPORT_CV_DETECTED_FRAMEWORK:
        errors.append("metadata detected_framework does not match Sport-CV analysis")

    sample_ok = bool(image_files) and bool(label_files)
    if not sample_ok:
        errors.append("at least one image-label pair is required")

    return {"valid": not errors, "errors": errors}
