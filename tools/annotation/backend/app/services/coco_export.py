import json
from pathlib import Path


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

    jersey_categories = [item for item in categories if item.get("id") == 1 and item.get("name") == "jersey"]
    if len(jersey_categories) != 1:
        errors.append("exactly one jersey category with id=1 must exist")

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
        if annotation.get("category_id") != 1:
            errors.append(f"annotation {annotation.get('id')} has invalid category_id {annotation.get('category_id')}")
        bbox = annotation.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4 or not all(isinstance(value, (int, float)) for value in bbox):
            errors.append(f"annotation {annotation.get('id')} has invalid bbox")
            continue
        if bbox[2] <= 0 or bbox[3] <= 0:
            errors.append(f"annotation {annotation.get('id')} has non-positive bbox width/height")
        if annotation.get("area", 0) <= 0:
            errors.append(f"annotation {annotation.get('id')} has non-positive area")

    return {"valid": not errors, "errors": errors}
