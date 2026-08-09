"""Validation of a written YOLO export.

This module does not write anything; it re-reads an export from disk and proves
it is what it claims to be. The writers live in
``app.api.routes.export_yolo_dataset`` (flat, per video) and
``app.services.dataset_versioning`` (split, versioned).

It used to hardcode a single ``jersey`` class. The class scheme now comes from
``app.services.basketball_classes``, which reads ``train/basketball.yaml``, so
this file has no opinion of its own about what the classes are. The point of
validating here at all is that a dataset with wrong class indices is completely
silent: Ultralytics imports it happily and trains every class under the wrong
label.
"""

import json
import math
from pathlib import Path
from typing import Any

from app.services.basketball_classes import (
    BASKETBALL_CLASS_NAMES,
    CLASS_COUNT,
    CLASS_NAMES_IN_INDEX_ORDER,
    NUMBER_BEARING_CLASS_ID,
    class_mapping_for_metadata,
    clean_jersey_number,
    compare_with_basketball_yaml,
)


SPORT_CV_DETECTED_FRAMEWORK = "Ultralytics YOLO"
SPORT_CV_DETECTED_MODEL_VERSION = "YOLO11-family weights via ultralytics 8.x"
SPORT_CV_TARGET_PROJECT = "Sport-CV"

#: Name of the OCR ground-truth side channel written next to the labels.
JERSEY_NUMBERS_FILENAME = "jersey_numbers.json"
JERSEY_NUMBERS_SCHEMA = "basketball-v2/jersey_numbers/v1"

#: Kept for callers that still import it. Derived, never hand-written.
YOLO_CLASS_MAPPING: dict[int, str] = dict(BASKETBALL_CLASS_NAMES)


def validate_label_file(
    label_path: Path, display_name: str, errors: list[str]
) -> tuple[int, dict[int, int]]:
    """Check one YOLO label file. Returns (line count, per-class counts)."""
    total_lines = 0
    per_class: dict[int, int] = {}
    for line_number, raw_line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{display_name}:{line_number} must contain 5 values")
            continue
        try:
            class_id = int(parts[0])
            values = [float(item) for item in parts[1:]]
        except ValueError:
            errors.append(f"{display_name}:{line_number} contains non-numeric YOLO values")
            continue
        if class_id not in BASKETBALL_CLASS_NAMES:
            errors.append(
                f"{display_name}:{line_number} has class_id {class_id}, which is not one of "
                f"the {CLASS_COUNT} basketball classes {CLASS_NAMES_IN_INDEX_ORDER}"
            )
        if any(math.isnan(value) or math.isinf(value) for value in values):
            errors.append(f"{display_name}:{line_number} contains NaN or infinity")
            continue
        x_center, y_center, width, height = values
        if not (0.0 <= x_center <= 1.0):
            errors.append(f"{display_name}:{line_number} x_center out of range")
        if not (0.0 <= y_center <= 1.0):
            errors.append(f"{display_name}:{line_number} y_center out of range")
        if not (0.0 < width <= 1.0):
            errors.append(f"{display_name}:{line_number} width out of range")
        if not (0.0 < height <= 1.0):
            errors.append(f"{display_name}:{line_number} height out of range")
        per_class[class_id] = per_class.get(class_id, 0) + 1
        total_lines += 1
    return total_lines, per_class


def validate_jersey_numbers_file(
    numbers_path: Path,
    label_lines_by_file: dict[str, list[str]],
    errors: list[str],
) -> int:
    """Check jersey_numbers.json against the labels it annotates.

    The file is only useful as OCR ground truth if every row still points at a
    real jersey line. A stale or misaligned row would quietly score OCR against
    the wrong box, so each one is re-resolved here.
    """
    if not numbers_path.exists():
        errors.append(f"missing {JERSEY_NUMBERS_FILENAME}: {numbers_path}")
        return 0

    try:
        payload = json.loads(numbers_path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"{JERSEY_NUMBERS_FILENAME} is not valid JSON: {exc}")
        return 0

    if payload.get("schema") != JERSEY_NUMBERS_SCHEMA:
        errors.append(
            f"{JERSEY_NUMBERS_FILENAME} schema must be '{JERSEY_NUMBERS_SCHEMA}', "
            f"got {payload.get('schema')!r}"
        )
    if int(payload.get("class_id", -1)) != NUMBER_BEARING_CLASS_ID:
        errors.append(
            f"{JERSEY_NUMBERS_FILENAME} class_id must be {NUMBER_BEARING_CLASS_ID} "
            f"(the jersey class), got {payload.get('class_id')!r}"
        )

    entries = payload.get("entries")
    if not isinstance(entries, list):
        errors.append(f"{JERSEY_NUMBERS_FILENAME} 'entries' must be a list")
        return 0

    for index, entry in enumerate(entries):
        where = f"{JERSEY_NUMBERS_FILENAME}[{index}]"
        label_file = str(entry.get("label_file", ""))
        line_index = entry.get("label_line_index")
        number = entry.get("jersey_number")

        lines = label_lines_by_file.get(label_file)
        if lines is None:
            errors.append(f"{where} references unknown label file {label_file!r}")
            continue
        if not isinstance(line_index, int) or not (0 <= line_index < len(lines)):
            errors.append(
                f"{where} label_line_index {line_index!r} is out of range for "
                f"{label_file} ({len(lines)} lines)"
            )
            continue

        line_class = lines[line_index].split()[0] if lines[line_index].split() else ""
        if line_class != str(NUMBER_BEARING_CLASS_ID):
            errors.append(
                f"{where} points at {label_file} line {line_index}, whose class is "
                f"{line_class!r}, not the jersey class {NUMBER_BEARING_CLASS_ID}"
            )
        if not isinstance(number, str) or not number:
            errors.append(f"{where} jersey_number must be a non-empty string, got {number!r}")
            continue
        if clean_jersey_number(number) != number:
            errors.append(
                f"{where} jersey_number {number!r} is not normalized. Ground truth must "
                "go through src/jersey.py:clean_jersey_number, or OCR accuracy measured "
                "against it will be wrong."
            )
    return len(entries)


def validate_yolo_dataset(export_root: str | Path) -> dict[str, object]:
    """Validate a flat (non-split) YOLO export directory."""
    root = Path(export_root)
    errors: list[str] = []

    images_dir = root / "images"
    labels_dir = root / "labels"
    dataset_yaml = root / "dataset.yaml"
    classes_txt = root / "classes.txt"
    metadata_json = root / "metadata.json"
    jersey_numbers_json = root / JERSEY_NUMBERS_FILENAME

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
    classes = [
        line.strip()
        for line in classes_txt.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    # The class scheme is compared field by field against train/basketball.yaml.
    # Nothing here re-states what the classes should be.
    errors.extend(compare_with_basketball_yaml(dataset_yaml))
    if classes != CLASS_NAMES_IN_INDEX_ORDER:
        errors.append(
            f"classes.txt must list the {CLASS_COUNT} basketball classes in INDEX order "
            f"{CLASS_NAMES_IN_INDEX_ORDER}, got {classes}. Order is not cosmetic: "
            "position is the class id."
        )

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
    counts_by_class: dict[int, int] = {}
    label_lines_by_file: dict[str, list[str]] = {}
    for label_path in label_files:
        label_lines_by_file[label_path.name] = [
            line.strip()
            for line in label_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        lines, per_class = validate_label_file(label_path, label_path.name, errors)
        total_lines += lines
        for class_id, count in per_class.items():
            counts_by_class[class_id] = counts_by_class.get(class_id, 0) + count

    jersey_number_count = validate_jersey_numbers_file(
        jersey_numbers_json, label_lines_by_file, errors
    )

    if int(metadata.get("image_count", 0)) != len(image_files):
        errors.append("metadata image_count does not match exported images")
    if int(metadata.get("label_file_count", 0)) != len(label_files):
        errors.append("metadata label_file_count does not match exported labels")
    if int(metadata.get("annotation_count", 0)) != total_lines:
        errors.append("metadata annotation_count does not match total YOLO label lines")
    if metadata.get("class_mapping") != class_mapping_for_metadata():
        errors.append(
            f"metadata class_mapping must be {class_mapping_for_metadata()}, "
            f"got {metadata.get('class_mapping')}"
        )
    if int(metadata.get("class_count", 0)) != CLASS_COUNT:
        errors.append(f"metadata class_count must be {CLASS_COUNT}")
    if metadata.get("detected_framework") != SPORT_CV_DETECTED_FRAMEWORK:
        errors.append("metadata detected_framework does not match Sport-CV analysis")

    if not image_files or not label_files:
        errors.append("at least one image-label pair is required")

    checks: dict[str, Any] = {
        "class_names": classes,
        "annotation_counts_by_class": {
            BASKETBALL_CLASS_NAMES[class_id]: count
            for class_id, count in sorted(counts_by_class.items())
        },
        "jersey_number_count": jersey_number_count,
        # Not an error: a first annotation pass legitimately covers only some
        # classes. Reported so an empty class is noticed now, not after training.
        "classes_with_no_annotations": [
            name
            for index, name in sorted(BASKETBALL_CLASS_NAMES.items())
            if counts_by_class.get(index, 0) == 0
        ],
    }
    return {"valid": not errors, "errors": errors, "checks": checks}
