"""COCO export validation.

Kept in step with the YOLO export: both describe the same annotations, so a
5-class YOLO export next to a 1-class COCO export would be a trap for whoever
picks the COCO one.

COCO category ids are 1-based by convention while YOLO class ids are 0-based, so
the mapping is ``category_id = yolo_class_id + 1``. That offset is defined once,
here, and must not be re-derived anywhere else.
"""

import json
from pathlib import Path

from app.services.basketball_classes import BASKETBALL_CLASS_NAMES, CLASS_COUNT

#: COCO ids start at 1; YOLO class ids start at 0.
COCO_CATEGORY_ID_OFFSET = 1

#: The categories block every COCO export must contain, in class-index order.
COCO_CATEGORIES: list[dict[str, object]] = [
    {
        "id": class_id + COCO_CATEGORY_ID_OFFSET,
        "name": BASKETBALL_CLASS_NAMES[class_id],
        "supercategory": "basketball",
    }
    for class_id in range(CLASS_COUNT)
]

_VALID_CATEGORY_IDS = {int(category["id"]) for category in COCO_CATEGORIES}


def coco_category_id(class_id: int) -> int:
    """Convert a YOLO class id into its COCO category id."""
    return int(class_id) + COCO_CATEGORY_ID_OFFSET


def validate_coco_dataset(instances_path: str | Path) -> dict[str, object]:
    path = Path(instances_path)
    errors: list[str] = []

    if not path.exists():
        return {"valid": False, "errors": [f"instances.json not found: {path}"]}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"valid": False, "errors": [f"invalid JSON: {exc}"]}

    images = payload.get("images")
    annotations = payload.get("annotations")
    categories = payload.get("categories")

    if not isinstance(images, list):
        errors.append("images must be a list")
        images = []
    if not isinstance(annotations, list):
        errors.append("annotations must be a list")
        annotations = []
    if not isinstance(categories, list):
        errors.append("categories must be a list")
        categories = []

    declared = [
        {"id": item.get("id"), "name": item.get("name")}
        for item in sorted(categories, key=lambda entry: entry.get("id") or 0)
    ]
    expected = [{"id": item["id"], "name": item["name"]} for item in COCO_CATEGORIES]
    if declared != expected:
        errors.append(f"categories must be exactly {expected}, got {declared}")

    image_ids = [item.get("id") for item in images]
    if len(image_ids) != len(set(image_ids)):
        errors.append("image IDs must be unique")
    annotation_ids = [item.get("id") for item in annotations]
    if len(annotation_ids) != len(set(annotation_ids)):
        errors.append("annotation IDs must be unique")

    image_id_set = {item.get("id") for item in images}
    images_dir = path.parent.parent / "images"
    for image in images:
        file_name = image.get("file_name")
        if not (images_dir / str(file_name)).exists():
            errors.append(f"referenced image file missing: {file_name}")

    for annotation in annotations:
        if annotation.get("image_id") not in image_id_set:
            errors.append(f"annotation {annotation.get('id')} references missing image_id {annotation.get('image_id')}")
        if annotation.get("category_id") not in _VALID_CATEGORY_IDS:
            errors.append(
                f"annotation {annotation.get('id')} has category_id "
                f"{annotation.get('category_id')}, not one of {sorted(_VALID_CATEGORY_IDS)}"
            )
        bbox = annotation.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4 or not all(isinstance(value, (int, float)) for value in bbox):
            errors.append(f"annotation {annotation.get('id')} has invalid bbox")
            continue
        if bbox[2] <= 0 or bbox[3] <= 0:
            errors.append(f"annotation {annotation.get('id')} has non-positive bbox width/height")
        if annotation.get("area", 0) <= 0:
            errors.append(f"annotation {annotation.get('id')} has non-positive area")

    return {"valid": not errors, "errors": errors}
