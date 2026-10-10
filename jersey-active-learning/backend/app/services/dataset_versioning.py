import hashlib
import json
import math
import random
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.dataset import DatasetVersion, ManualAnnotation, VideoAsset, VideoFrame
from app.services.yolo_export import SPORT_CV_TARGET_PROJECT


EXPORTABLE_LABEL_STATUSES = {"correct", "wrong_bbox", "wrong_number"}
DATASET_FORMAT = "yolo_detection_versioned_v1"
TASK_TYPE = "jersey_bbox_detection"
CLASS_MAPPING = {"0": "jersey"}
SPLIT_NAMES = ["train", "val", "test"]


def _datasets_versions_root() -> Path:
    root = (settings.data_dir / "datasets" / "versions").resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _dataset_root(version_name: str) -> Path:
    root = _datasets_versions_root() / version_name
    root.mkdir(parents=True, exist_ok=True)
    for split_name in SPLIT_NAMES:
        (root / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split_name).mkdir(parents=True, exist_ok=True)
    return root


def _canonical_frame_path(video_id: int, frame_id: int) -> Path:
    return (settings.data_dir / "frames" / str(video_id) / f"frame_{frame_id:06d}.jpg").resolve()


def _parse_bbox(raw_value: object) -> list[int]:
    value = raw_value
    if isinstance(raw_value, str):
        value = json.loads(raw_value)
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("bbox must contain 4 values")
    return [int(item) for item in value]


def _frame_path_for_annotation(annotation: ManualAnnotation, frame_map: dict[tuple[int, int], VideoFrame]) -> Path | None:
    key = (annotation.video_id, annotation.frame_id)
    frame_row = frame_map.get(key)
    if frame_row is not None and frame_row.frame_path:
        path = Path(frame_row.frame_path)
        if path.exists():
            return path.resolve()
    if annotation.image_path:
        path = Path(annotation.image_path)
        if path.exists():
            return path.resolve()
    canonical = _canonical_frame_path(annotation.video_id, annotation.frame_id)
    if canonical.exists():
        return canonical
    return None


def _resolve_dimensions(frame_row: VideoFrame | None, frame_path: Path | None) -> tuple[int | None, int | None]:
    if frame_row is not None and frame_row.image_width and frame_row.image_height:
        return int(frame_row.image_width), int(frame_row.image_height)
    if frame_path is None or not frame_path.exists():
        return None, None
    image = cv2.imdecode(np.fromfile(str(frame_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return None, None
    height, width = image.shape[:2]
    return int(width), int(height)


def _normalized_yolo_bbox(bbox: list[int], image_width: int, image_height: int) -> tuple[list[int], list[float]]:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(x1), image_width))
    y1 = max(0, min(int(y1), image_height))
    x2 = max(0, min(int(x2), image_width))
    y2 = max(0, min(int(y2), image_height))
    if x1 >= x2 or y1 >= y2:
        raise ValueError("invalid bbox after clamping")
    width = x2 - x1
    height = y2 - y1
    x_center = ((x1 + x2) / 2.0) / image_width
    y_center = ((y1 + y2) / 2.0) / image_height
    normalized_width = width / image_width
    normalized_height = height / image_height
    values = [x_center, y_center, normalized_width, normalized_height]
    if any(math.isnan(value) or math.isinf(value) for value in values):
        raise ValueError("bbox normalization produced invalid value")
    if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0 and 0.0 < normalized_width <= 1.0 and 0.0 < normalized_height <= 1.0):
        raise ValueError("normalized bbox out of range")
    return [x1, y1, x2, y2], values


def _serialize_source_record(annotation: ManualAnnotation) -> dict[str, Any]:
    return {
        "video_id": int(annotation.video_id),
        "frame_id": int(annotation.frame_id),
        "annotation_id": int(annotation.annotation_id),
        "prediction_id": int(annotation.prediction_id),
        "corrected_bbox": _parse_bbox(annotation.corrected_bbox),
        "corrected_number": annotation.corrected_number,
        "label_status": annotation.label_status,
        "deleted_at": annotation.deleted_at.isoformat() if annotation.deleted_at else None,
    }


def compute_source_data_fingerprint(annotations: list[ManualAnnotation]) -> str:
    payload = sorted((_serialize_source_record(annotation) for annotation in annotations), key=lambda item: (item["video_id"], item["frame_id"], item["annotation_id"]))
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def collect_exportable_annotations(db: Session, video_ids: list[int]) -> dict[str, Any]:
    videos = db.query(VideoAsset).filter(VideoAsset.id.in_(video_ids)).all()
    found_ids = {video.id for video in videos}
    missing = sorted(set(video_ids) - found_ids)
    if missing:
        raise HTTPException(status_code=404, detail=f"Videos not found: {missing}")

    annotations = (
        db.query(ManualAnnotation)
        .filter(
            ManualAnnotation.video_id.in_(video_ids),
            ManualAnnotation.deleted_at.is_(None),
            ManualAnnotation.label_status.in_(tuple(EXPORTABLE_LABEL_STATUSES)),
        )
        .order_by(ManualAnnotation.video_id.asc(), ManualAnnotation.frame_id.asc(), ManualAnnotation.track_id.asc(), ManualAnnotation.annotation_id.asc())
        .all()
    )
    grouped_by_video: dict[int, list[ManualAnnotation]] = defaultdict(list)
    for annotation in annotations:
        grouped_by_video[int(annotation.video_id)].append(annotation)

    missing_annotations = [video_id for video_id in video_ids if not grouped_by_video.get(int(video_id))]
    if missing_annotations:
        raise HTTPException(status_code=400, detail=f"Videos have no exportable annotations: {missing_annotations}")

    frame_rows = (
        db.query(VideoFrame)
        .filter(VideoFrame.video_id.in_(video_ids))
        .order_by(VideoFrame.video_id.asc(), VideoFrame.frame_id.asc())
        .all()
    )
    frame_map = {(int(frame.video_id), int(frame.frame_id)): frame for frame in frame_rows}
    return {
        "videos": videos,
        "annotations": annotations,
        "frame_map": frame_map,
        "fingerprint": compute_source_data_fingerprint(annotations),
    }


def build_canonical_images(annotations: list[ManualAnnotation], frame_map: dict[tuple[int, int], VideoFrame]) -> dict[tuple[int, int], dict[str, Any]]:
    images: dict[tuple[int, int], dict[str, Any]] = {}
    for annotation in annotations:
        key = (int(annotation.video_id), int(annotation.frame_id))
        frame_row = frame_map.get(key)
        frame_path = _frame_path_for_annotation(annotation, frame_map)
        image_width, image_height = _resolve_dimensions(frame_row, frame_path)
        image = images.get(key)
        if image is None:
            image = {
                "canonical_key": key,
                "source_video_id": int(annotation.video_id),
                "source_frame_id": int(annotation.frame_id),
                "source_frame_row_id": int(frame_row.id) if frame_row is not None else None,
                "source_frame_path": str(frame_path) if frame_path is not None else None,
                "image_width": image_width,
                "image_height": image_height,
                "annotations": [],
                "track_ids": set(),
                "source_annotation_ids": set(),
                "source_prediction_ids": set(),
            }
            images[key] = image
        image["annotations"].append(annotation)
        image["track_ids"].add(int(annotation.track_id))
        image["source_annotation_ids"].add(int(annotation.annotation_id))
        image["source_prediction_ids"].add(int(annotation.prediction_id))
        if image["source_frame_path"] is None and frame_path is not None:
            image["source_frame_path"] = str(frame_path)
        if image["image_width"] is None:
            image["image_width"] = image_width
        if image["image_height"] is None:
            image["image_height"] = image_height

    for image in images.values():
        image["track_ids"] = sorted(image["track_ids"])
        image["source_annotation_ids"] = sorted(image["source_annotation_ids"])
        image["source_prediction_ids"] = sorted(image["source_prediction_ids"])
    return images


def build_image_groups(canonical_images: dict[tuple[int, int], dict[str, Any]], split_strategy: str, temporal_block_size: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    warnings: list[dict[str, Any]] = []
    if split_strategy == "video":
        grouped: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for key, image in canonical_images.items():
            grouped[int(image["source_video_id"])].append(key)
        return [
            {
                "group_key": f"video:{video_id}",
                "group_type": "video",
                "image_keys": sorted(image_keys),
                "source_video_ids": [video_id],
            }
            for video_id, image_keys in sorted(grouped.items())
        ], warnings

    if split_strategy == "temporal_block":
        groups: list[dict[str, Any]] = []
        per_video: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for key, image in canonical_images.items():
            per_video[int(image["source_video_id"])].append(key)
        for video_id, keys in sorted(per_video.items()):
            sorted_keys = sorted(keys, key=lambda item: item[1])
            for index in range(0, len(sorted_keys), temporal_block_size):
                block_keys = sorted_keys[index:index + temporal_block_size]
                block_index = index // temporal_block_size
                groups.append(
                    {
                        "group_key": f"temporal:{video_id}:{block_index}",
                        "group_type": "temporal_block",
                        "image_keys": block_keys,
                        "source_video_ids": [video_id],
                        "block_index": block_index,
                    }
                )
        return groups, warnings

    if split_strategy == "track":
        adjacency: dict[str, set[str]] = defaultdict(set)
        node_images: dict[str, set[tuple[int, int]]] = defaultdict(set)
        for key, image in canonical_images.items():
            frame_node = f"frame:{key[0]}:{key[1]}"
            node_images[frame_node].add(key)
            track_nodes = {f"track:{key[0]}:{track_id}" for track_id in image["track_ids"]}
            for track_node in track_nodes:
                adjacency[frame_node].add(track_node)
                adjacency[track_node].add(frame_node)
                node_images[track_node].add(key)
            if len(track_nodes) > 1:
                warnings.append(
                    {
                        "frame_key": [key[0], key[1]],
                        "involved_track_groups": sorted(track_nodes),
                        "resolution_reason": "shared_frame_forces_same_split",
                    }
                )

        visited: set[str] = set()
        groups: list[dict[str, Any]] = []
        for node in sorted(node_images.keys()):
            if node in visited:
                continue
            stack = [node]
            component_nodes: set[str] = set()
            component_images: set[tuple[int, int]] = set()
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                component_nodes.add(current)
                component_images.update(node_images.get(current, set()))
                for neighbor in sorted(adjacency.get(current, set())):
                    if neighbor not in visited:
                        stack.append(neighbor)
            video_ids = sorted({image_key[0] for image_key in component_images})
            groups.append(
                {
                    "group_key": f"track_component:{len(groups)}",
                    "group_type": "track_component",
                    "image_keys": sorted(component_images),
                    "source_video_ids": video_ids,
                    "component_nodes": sorted(component_nodes),
                }
            )
        return groups, warnings

    raise ValueError(f"Unsupported split_strategy: {split_strategy}")


def assign_groups_to_splits(groups: list[dict[str, Any]], split_ratios: dict[str, float], random_seed: int) -> dict[str, str]:
    rng = random.Random(random_seed)
    shuffled_groups = list(groups)
    rng.shuffle(shuffled_groups)
    shuffled_groups.sort(key=lambda item: -len(item["image_keys"]))

    total_images = sum(len(group["image_keys"]) for group in groups)
    targets = {split_name: total_images * split_ratios[split_name] for split_name in SPLIT_NAMES}
    counts = {split_name: 0 for split_name in SPLIT_NAMES}
    assignments: dict[str, str] = {}

    for group in shuffled_groups:
        group_size = len(group["image_keys"])
        best_split = None
        best_score = None
        for split_name in SPLIT_NAMES:
            deficit = targets[split_name] - counts[split_name]
            score = (deficit, -counts[split_name], -SPLIT_NAMES.index(split_name))
            if best_score is None or score > best_score:
                best_score = score
                best_split = split_name
        assert best_split is not None
        assignments[group["group_key"]] = best_split
        counts[best_split] += group_size
    return assignments


def _build_manifest_record(image: dict[str, Any], split_name: str, exported_image_path: Path, exported_label_path: Path, annotation_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "dataset_image_id": f"{image['source_video_id']}_{image['source_frame_id']}",
        "split": split_name,
        "exported_image_path": str(exported_image_path.resolve()),
        "exported_label_path": str(exported_label_path.resolve()),
        "source_video_id": image["source_video_id"],
        "source_frame_id": image["source_frame_id"],
        "source_frame_row_id": image["source_frame_row_id"],
        "source_frame_path": image["source_frame_path"],
        "image_width": image["image_width"],
        "image_height": image["image_height"],
        "source_annotation_ids": image["source_annotation_ids"],
        "source_prediction_ids": image["source_prediction_ids"],
        "track_ids": image["track_ids"],
        "annotations": annotation_payloads,
    }


def _write_version_files(dataset_root: Path, manifest: list[dict[str, Any]], metadata: dict[str, Any], split_report: dict[str, Any]) -> tuple[Path, Path, Path, Path]:
    dataset_yaml = dataset_root / "dataset.yaml"
    dataset_yaml.write_text(
        "\n".join([
            f'path: "{str(dataset_root.resolve()).replace(chr(92), "/")}"',
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "nc: 1",
            "names:",
            "  0: jersey",
            "",
        ]),
        encoding="utf-8",
    )
    classes_txt = dataset_root / "classes.txt"
    classes_txt.write_text("jersey\n", encoding="utf-8")
    manifest_path = dataset_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata_path = dataset_root / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    split_report_path = dataset_root / "split_report.json"
    split_report_path.write_text(json.dumps(split_report, ensure_ascii=False, indent=2), encoding="utf-8")
    return dataset_yaml, classes_txt, manifest_path, metadata_path


def validate_dataset_version(export_root: str | Path, split_strategy: str) -> dict[str, Any]:
    root = Path(export_root)
    errors: list[str] = []
    warnings: list[str] = []
    manifest_path = root / "manifest.json"
    metadata_path = root / "metadata.json"
    split_report_path = root / "split_report.json"
    if not manifest_path.exists():
        errors.append(f"manifest missing: {manifest_path}")
    if not metadata_path.exists():
        errors.append(f"metadata missing: {metadata_path}")
    if not split_report_path.exists():
        errors.append(f"split_report missing: {split_report_path}")
    if errors:
        return {"valid": False, "errors": errors, "warnings": warnings, "checks": {}}

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    split_report = json.loads(split_report_path.read_text(encoding="utf-8"))

    dataset_yaml = root / "dataset.yaml"
    classes_txt = root / "classes.txt"
    if not dataset_yaml.exists():
        errors.append(f"dataset.yaml missing: {dataset_yaml}")
    if not classes_txt.exists():
        errors.append(f"classes.txt missing: {classes_txt}")
    if errors:
        return {"valid": False, "errors": errors, "warnings": warnings, "checks": {}}

    classes = [line.strip() for line in classes_txt.read_text(encoding="utf-8").splitlines() if line.strip()]
    if classes != ["jersey"]:
        errors.append("classes.txt must contain exactly one class: jersey")

    image_to_split: dict[tuple[int, int], str] = {}
    annotation_ids: set[int] = set()
    prediction_ids: set[int] = set()
    video_split_map: dict[int, set[str]] = defaultdict(set)
    track_split_map: dict[tuple[int, int], set[str]] = defaultdict(set)
    temporal_block_map: dict[tuple[int, int], set[str]] = defaultdict(set)

    split_counts = {split_name: {"images": 0, "annotations": 0} for split_name in SPLIT_NAMES}
    image_files_seen: set[str] = set()
    label_files_seen: set[str] = set()
    for item in manifest:
        split_name = item["split"]
        split_counts[split_name]["images"] += 1
        key = (int(item["source_video_id"]), int(item["source_frame_id"]))
        if key in image_to_split and image_to_split[key] != split_name:
            errors.append(f"canonical image {key} appears in multiple splits")
        image_to_split[key] = split_name
        exported_image_path = Path(item["exported_image_path"])
        exported_label_path = Path(item["exported_label_path"])
        if not exported_image_path.exists():
            errors.append(f"exported image missing: {exported_image_path}")
        if not exported_label_path.exists():
            errors.append(f"exported label missing: {exported_label_path}")
        if exported_image_path.name in image_files_seen:
            errors.append(f"duplicate exported image filename: {exported_image_path.name}")
        image_files_seen.add(exported_image_path.name)
        if exported_label_path.name in label_files_seen:
            errors.append(f"duplicate exported label filename: {exported_label_path.name}")
        label_files_seen.add(exported_label_path.name)
        video_split_map[int(item["source_video_id"])].add(split_name)
        block_size = int(metadata.get("temporal_block_size", 30) or 30)
        temporal_block_map[(int(item["source_video_id"]), int(item["source_frame_id"]) // block_size)].add(split_name)
        for annotation in item["annotations"]:
            annotation_id = int(annotation["source_annotation_id"])
            prediction_id = int(annotation["prediction_id"])
            track_key = (int(item["source_video_id"]), int(annotation["track_id"]))
            if annotation_id in annotation_ids:
                errors.append(f"source_annotation_id duplicated across splits: {annotation_id}")
            annotation_ids.add(annotation_id)
            if prediction_id in prediction_ids:
                errors.append(f"source_prediction_id duplicated across splits: {prediction_id}")
            prediction_ids.add(prediction_id)
            track_split_map[track_key].add(split_name)
            split_counts[split_name]["annotations"] += 1

        label_lines = [line.strip() for line in exported_label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(label_lines) != len(item["annotations"]):
            errors.append(f"label line count mismatch for {exported_label_path.name}")
        for line_number, line in enumerate(label_lines, start=1):
            parts = line.split()
            if len(parts) != 5:
                errors.append(f"{exported_label_path.name}:{line_number} must contain 5 values")
                continue
            try:
                class_id = int(parts[0])
                values = [float(value) for value in parts[1:]]
            except ValueError:
                errors.append(f"{exported_label_path.name}:{line_number} contains non-numeric values")
                continue
            if class_id != 0:
                errors.append(f"{exported_label_path.name}:{line_number} invalid class_id {class_id}")
            x_center, y_center, width, height = values
            if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0 and 0.0 < width <= 1.0 and 0.0 < height <= 1.0):
                errors.append(f"{exported_label_path.name}:{line_number} has invalid normalized bbox")

    if split_strategy == "video":
        for video_id, splits in video_split_map.items():
            if len(splits) > 1:
                errors.append(f"video {video_id} appears in multiple splits: {sorted(splits)}")
    if split_strategy == "track":
        for track_key, splits in track_split_map.items():
            if len(splits) > 1:
                errors.append(f"track group {track_key} appears in multiple splits: {sorted(splits)}")
    if split_strategy == "temporal_block":
        for block_key, splits in temporal_block_map.items():
            if len(splits) > 1:
                errors.append(f"temporal block {block_key} appears in multiple splits: {sorted(splits)}")

    for split_name in SPLIT_NAMES:
        metadata_split = metadata.get("split_counts", {}).get(split_name, {})
        if int(metadata_split.get("images", 0)) != split_counts[split_name]["images"]:
            errors.append(f"metadata image count mismatch for split {split_name}")
        if int(metadata_split.get("annotations", 0)) != split_counts[split_name]["annotations"]:
            errors.append(f"metadata annotation count mismatch for split {split_name}")

    checks = {
        "image_count": len(manifest),
        "annotation_count": sum(len(item["annotations"]) for item in manifest),
        "class_names": classes,
        "split_report_present": bool(split_report),
    }
    return {"valid": not errors, "errors": errors, "warnings": warnings, "checks": checks}


def load_dataset_manifest(dataset_version: DatasetVersion) -> Any:
    path = Path(dataset_version.export_path) / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _load_split_report(dataset_version: DatasetVersion) -> Any:
    path = Path(dataset_version.export_path) / "split_report.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _dataset_version_to_record(dataset_version: DatasetVersion) -> dict[str, Any]:
    validation_result = None
    if dataset_version.validation_result_json:
        try:
            validation_result = json.loads(dataset_version.validation_result_json)
        except Exception:
            validation_result = dataset_version.validation_result_json
    return {
        "dataset_version_id": dataset_version.dataset_version_id,
        "version_name": dataset_version.version_name,
        "description": dataset_version.description,
        "dataset_format": dataset_version.dataset_format,
        "task_type": dataset_version.task_type,
        "split_strategy": dataset_version.split_strategy,
        "random_seed": dataset_version.random_seed,
        "train_ratio": dataset_version.train_ratio,
        "val_ratio": dataset_version.val_ratio,
        "test_ratio": dataset_version.test_ratio,
        "temporal_block_size": dataset_version.temporal_block_size,
        "strict_split": bool(dataset_version.strict_split),
        "source_video_ids": json.loads(dataset_version.source_video_ids),
        "source_annotation_count": dataset_version.source_annotation_count,
        "exported_image_count": dataset_version.exported_image_count,
        "exported_annotation_count": dataset_version.exported_annotation_count,
        "train_image_count": dataset_version.train_image_count,
        "val_image_count": dataset_version.val_image_count,
        "test_image_count": dataset_version.test_image_count,
        "train_annotation_count": dataset_version.train_annotation_count,
        "val_annotation_count": dataset_version.val_annotation_count,
        "test_annotation_count": dataset_version.test_annotation_count,
        "export_path": dataset_version.export_path,
        "metadata_path": dataset_version.metadata_path,
        "status": dataset_version.status,
        "validation_result_json": validation_result,
        "source_data_fingerprint": dataset_version.source_data_fingerprint,
        "created_at": dataset_version.created_at,
        "completed_at": dataset_version.completed_at,
    }


def build_dataset_version(db: Session, *, version_name: str, description: str | None, video_ids: list[int], train_ratio: float, val_ratio: float, test_ratio: float, random_seed: int, split_strategy: str, temporal_block_size: int, strict_split: bool) -> dict[str, Any]:
    existing = db.query(DatasetVersion).filter(DatasetVersion.version_name == version_name).first()
    if existing is not None:
        raise HTTPException(status_code=409, detail=f"Dataset version already exists: {version_name}")

    collected = collect_exportable_annotations(db, video_ids)
    annotations: list[ManualAnnotation] = collected["annotations"]
    frame_map: dict[tuple[int, int], VideoFrame] = collected["frame_map"]
    fingerprint = collected["fingerprint"]

    dataset_root = _dataset_root(version_name)
    dataset_version = DatasetVersion(
        version_name=version_name,
        description=description,
        dataset_format=DATASET_FORMAT,
        task_type=TASK_TYPE,
        split_strategy=split_strategy,
        random_seed=random_seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        temporal_block_size=temporal_block_size,
        strict_split=1 if strict_split else 0,
        source_video_ids=json.dumps(video_ids),
        source_annotation_count=len(annotations),
        export_path=str(dataset_root.resolve()),
        metadata_path=str((dataset_root / "metadata.json").resolve()),
        status="building",
        source_data_fingerprint=fingerprint,
    )
    db.add(dataset_version)
    db.commit()
    db.refresh(dataset_version)

    try:
        canonical_images = build_canonical_images(annotations, frame_map)
        groups, conflict_warnings = build_image_groups(canonical_images, split_strategy, temporal_block_size)
        split_ratios = {"train": train_ratio, "val": val_ratio, "test": test_ratio}
        assignments = assign_groups_to_splits(groups, split_ratios, random_seed)

        split_to_image_keys: dict[str, list[tuple[int, int]]] = {split_name: [] for split_name in SPLIT_NAMES}
        group_counts = {split_name: 0 for split_name in SPLIT_NAMES}
        for group in groups:
            split_name = assignments[group["group_key"]]
            split_to_image_keys[split_name].extend(group["image_keys"])
            group_counts[split_name] += 1

        warnings: list[str] = []
        if strict_split:
            for split_name, ratio in split_ratios.items():
                if ratio > 0 and not split_to_image_keys[split_name]:
                    raise HTTPException(status_code=400, detail=f"strict_split requested but split '{split_name}' cannot be populated without violating grouping constraints")
        else:
            for split_name, ratio in split_ratios.items():
                if ratio > 0 and not split_to_image_keys[split_name]:
                    warnings.append(f"Split '{split_name}' could not be populated without violating grouping constraints")

        manifest: list[dict[str, Any]] = []
        skipped_unclear_count = 0
        skipped_invalid_bbox_count = 0
        skipped_missing_image_count = 0
        skipped_annotations: list[dict[str, Any]] = []
        split_counts = {split_name: {"images": 0, "annotations": 0} for split_name in SPLIT_NAMES}

        for split_name in SPLIT_NAMES:
            for image_key in sorted(split_to_image_keys[split_name], key=lambda item: (item[0], item[1])):
                image = canonical_images[image_key]
                source_frame_path = Path(image["source_frame_path"]) if image["source_frame_path"] else None
                if source_frame_path is None or not source_frame_path.exists():
                    for annotation in image["annotations"]:
                        skipped_missing_image_count += 1
                        skipped_annotations.append({"annotation_id": int(annotation.annotation_id), "frame_id": int(annotation.frame_id), "reason": "missing_image"})
                    continue
                image_width = image["image_width"]
                image_height = image["image_height"]
                if not image_width or not image_height:
                    for annotation in image["annotations"]:
                        skipped_missing_image_count += 1
                        skipped_annotations.append({"annotation_id": int(annotation.annotation_id), "frame_id": int(annotation.frame_id), "reason": "missing_image"})
                    continue

                label_lines: list[str] = []
                annotation_payloads: list[dict[str, Any]] = []
                for annotation in image["annotations"]:
                    if annotation.label_status == "unclear":
                        skipped_unclear_count += 1
                        skipped_annotations.append({"annotation_id": int(annotation.annotation_id), "frame_id": int(annotation.frame_id), "reason": "unclear_label_status"})
                        continue
                    try:
                        corrected_bbox = _parse_bbox(annotation.corrected_bbox)
                        clamped_bbox, yolo_bbox = _normalized_yolo_bbox(corrected_bbox, int(image_width), int(image_height))
                    except Exception:
                        skipped_invalid_bbox_count += 1
                        skipped_annotations.append({"annotation_id": int(annotation.annotation_id), "frame_id": int(annotation.frame_id), "reason": "invalid_bbox"})
                        continue
                    label_lines.append(f"0 {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}")
                    annotation_payloads.append(
                        {
                            "source_annotation_id": int(annotation.annotation_id),
                            "prediction_id": int(annotation.prediction_id),
                            "track_id": int(annotation.track_id),
                            "corrected_bbox": clamped_bbox,
                            "corrected_number": annotation.corrected_number,
                            "label_status": annotation.label_status,
                            "yolo_bbox": yolo_bbox,
                        }
                    )

                if not annotation_payloads:
                    continue

                image_filename = f"video_{image['source_video_id']}_frame_{image['source_frame_id']:06d}{source_frame_path.suffix or '.jpg'}"
                label_filename = f"video_{image['source_video_id']}_frame_{image['source_frame_id']:06d}.txt"
                exported_image_path = dataset_root / "images" / split_name / image_filename
                exported_label_path = dataset_root / "labels" / split_name / label_filename
                shutil.copy2(source_frame_path, exported_image_path)
                exported_label_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")

                manifest.append(_build_manifest_record(image, split_name, exported_image_path, exported_label_path, annotation_payloads))
                split_counts[split_name]["images"] += 1
                split_counts[split_name]["annotations"] += len(annotation_payloads)

        total_images = len(manifest)
        total_annotations = sum(item["annotations"] and len(item["annotations"]) or 0 for item in manifest)
        for split_name, ratio in split_ratios.items():
            if ratio > 0 and split_counts[split_name]["images"] == 0:
                warnings.append(f"Split '{split_name}' ended up empty after filtering invalid or missing samples")
        actual_ratios = {split_name: (split_counts[split_name]["images"] / total_images if total_images else 0.0) for split_name in SPLIT_NAMES}
        split_quality = "good"
        if warnings:
            split_quality = "limited"
        if total_images == 0 or total_annotations == 0:
            split_quality = "invalid"

        metadata = {
            "dataset_version_id": dataset_version.dataset_version_id,
            "version_name": version_name,
            "description": description,
            "dataset_format": DATASET_FORMAT,
            "task_type": TASK_TYPE,
            "source": "jersey-active-learning",
            "target_project": SPORT_CV_TARGET_PROJECT,
            "source_video_ids": video_ids,
            "split_strategy": split_strategy,
            "random_seed": random_seed,
            "temporal_block_size": temporal_block_size,
            "strict_split": strict_split,
            "requested_ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
            "actual_ratios": actual_ratios,
            "image_count": total_images,
            "annotation_count": total_annotations,
            "split_counts": split_counts,
            "skipped_unclear_count": skipped_unclear_count,
            "skipped_invalid_bbox_count": skipped_invalid_bbox_count,
            "skipped_missing_image_count": skipped_missing_image_count,
            "split_quality": split_quality,
            "warnings": warnings,
            "source_data_fingerprint": fingerprint,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        split_report = {
            "requested_ratios": metadata["requested_ratios"],
            "actual_ratios": actual_ratios,
            "group_counts_per_split": group_counts,
            "image_counts_per_split": {split_name: split_counts[split_name]["images"] for split_name in SPLIT_NAMES},
            "annotation_counts_per_split": {split_name: split_counts[split_name]["annotations"] for split_name in SPLIT_NAMES},
            "source_video_distribution": {
                split_name: sorted({item["source_video_id"] for item in manifest if item["split"] == split_name}) for split_name in SPLIT_NAMES
            },
            "track_distribution": {
                split_name: sorted({track_id for item in manifest if item["split"] == split_name for track_id in item["track_ids"]}) for split_name in SPLIT_NAMES
            },
            "temporal_block_distribution": {
                split_name: sorted({f"{item['source_video_id']}:{item['source_frame_id'] // temporal_block_size}" for item in manifest if item["split"] == split_name}) if split_strategy == "temporal_block" else []
                for split_name in SPLIT_NAMES
            },
            "frame_conflict_resolutions": conflict_warnings,
            "warnings": warnings,
        }

        _, _, manifest_path, metadata_path = _write_version_files(dataset_root, manifest, metadata, split_report)
        validation_result = validate_dataset_version(dataset_root, split_strategy)
        split_report["leakage_validation_results"] = validation_result
        (dataset_root / "split_report.json").write_text(json.dumps(split_report, ensure_ascii=False, indent=2), encoding="utf-8")

        if not validation_result["valid"]:
            dataset_version.status = "failed"
            dataset_version.validation_result_json = json.dumps(validation_result, ensure_ascii=False)
            dataset_version.completed_at = datetime.now(timezone.utc)
            db.add(dataset_version)
            db.commit()
            raise HTTPException(status_code=500, detail={"message": "Dataset version validation failed", "errors": validation_result["errors"]})

        dataset_version.exported_image_count = total_images
        dataset_version.exported_annotation_count = total_annotations
        dataset_version.train_image_count = split_counts["train"]["images"]
        dataset_version.val_image_count = split_counts["val"]["images"]
        dataset_version.test_image_count = split_counts["test"]["images"]
        dataset_version.train_annotation_count = split_counts["train"]["annotations"]
        dataset_version.val_annotation_count = split_counts["val"]["annotations"]
        dataset_version.test_annotation_count = split_counts["test"]["annotations"]
        dataset_version.metadata_path = str(metadata_path.resolve())
        dataset_version.status = "completed"
        dataset_version.validation_result_json = json.dumps(validation_result, ensure_ascii=False)
        dataset_version.completed_at = datetime.now(timezone.utc)
        db.add(dataset_version)
        db.commit()
        db.refresh(dataset_version)
        return _dataset_version_to_record(dataset_version)
    except HTTPException as exc:
        dataset_version.status = "failed"
        dataset_version.validation_result_json = json.dumps({"valid": False, "errors": [str(exc.detail)]}, ensure_ascii=False)
        dataset_version.completed_at = datetime.now(timezone.utc)
        db.add(dataset_version)
        db.commit()
        raise
    except Exception as exc:
        dataset_version.status = "failed"
        dataset_version.validation_result_json = json.dumps({"valid": False, "errors": [str(exc)]}, ensure_ascii=False)
        dataset_version.completed_at = datetime.now(timezone.utc)
        db.add(dataset_version)
        db.commit()
        raise


def rebuild_dataset_version(db: Session, dataset_version: DatasetVersion) -> dict[str, Any]:
    source_video_ids = json.loads(dataset_version.source_video_ids)
    collected = collect_exportable_annotations(db, source_video_ids)
    current_fingerprint = collected["fingerprint"]
    if current_fingerprint != dataset_version.source_data_fingerprint:
        raise HTTPException(status_code=409, detail="Source annotations changed. Create a new dataset version.")
    return _dataset_version_to_record(dataset_version)


def list_dataset_versions(db: Session) -> list[dict[str, Any]]:
    rows = db.query(DatasetVersion).order_by(DatasetVersion.created_at.desc(), DatasetVersion.dataset_version_id.desc()).all()
    return [
        {
            "dataset_version_id": row.dataset_version_id,
            "version_name": row.version_name,
            "status": row.status,
            "split_strategy": row.split_strategy,
            "source_video_ids": json.loads(row.source_video_ids),
            "exported_image_count": row.exported_image_count,
            "exported_annotation_count": row.exported_annotation_count,
            "created_at": row.created_at,
        }
        for row in rows
    ]
