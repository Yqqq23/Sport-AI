"""Train / validate the 4-class basketball detector (player, ball, rim, jersey).

Salvaged from:
  SRC_B/basketball_tracking_pipeline.py:37-73   -> train_basketball_detector
  SRC_B/basketball_tracking_pipeline.py:113-131 -> validate_model

Minimal adaptations (per the Phase A brief):
  - base weights default to YOLO26 instead of yolo11s.pt
  - device defaults to 0 (GPU) instead of "cpu"
  - every path comes from the command line, none are hardcoded
  - validation now reports per-class mAP so ball and rim can be read separately
    (the source printed only the dataset-wide map50 / map)
  - the source's `if __name__ == "__main__"` block had every call commented out;
    replaced with a real argparse CLI

Training hyperparameters carried over unchanged from the source:
  epochs=100, imgsz=960, batch=-1 (auto), workers=0, patience=50, save_period=10
The imgsz=960 default matters: ball and rim are small objects and the source
comment called this out explicitly.

Run on the GPU server, e.g.:
  python train/train.py train --data train/basketball.yaml --model yolo26s.pt
  python train/train.py val   --data train/basketball.yaml --weights runs/detect/train/basketball_baseline/weights/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path


def train_basketball_detector(
    data: Path,
    model: str = "yolo26s.pt",
    epochs: int = 100,
    imgsz: int = 960,
    device: str = "0",
    project: Path = Path("runs/detect/train"),
    name: str = "basketball_baseline",
    batch: int = -1,
    workers: int = 0,
    patience: int = 50,
    save_period: int = 10,
) -> Path:
    """Train the 4-class detector.

    Args:
        data: path to basketball.yaml
        model: base weights. YOLO26 variants: yolo26n/s/m/l/x.pt
        imgsz: 960+ recommended for small objects (ball, rim)
        device: "0" for first GPU, "0,1" for multi-GPU, "cpu" to force CPU
    Returns:
        Path to best.pt
    """
    from ultralytics import YOLO

    data = Path(data)
    if not data.exists():
        raise FileNotFoundError(f"Dataset config not found: {data}")

    detector = YOLO(model)
    detector.train(
        data=str(data),
        epochs=epochs,
        imgsz=imgsz,
        device=device,
        batch=batch,
        workers=workers,
        patience=patience,
        save_period=save_period,
        project=str(project),
        name=name,
    )

    best_model_path = Path(project) / name / "weights" / "best.pt"
    print(f"Training completed. Best model: {best_model_path}")
    return best_model_path


def validate_model(
    data: Path,
    weights: Path,
    imgsz: int = 960,
    device: str = "0",
) -> dict[str, float]:
    """Validate and report mAP per class.

    The source printed only dataset-wide map50/map. Ball and rim are the classes
    most likely to be weak, so they must be visible individually.
    """
    from ultralytics import YOLO

    data = Path(data)
    weights = Path(weights)
    if not data.exists():
        raise FileNotFoundError(f"Dataset config not found: {data}")
    if not weights.exists():
        raise FileNotFoundError(
            f"Weights not found: {weights}\n"
            "Train first, or point --weights at an existing checkpoint."
        )

    detector = YOLO(str(weights))
    metrics = detector.val(data=str(data), imgsz=imgsz, device=device)

    print()
    print("=" * 62)
    print(f"{'class':<12}{'mAP50':>12}{'mAP50-95':>12}")
    print("-" * 62)

    summary: dict[str, float] = {
        "all/mAP50": float(metrics.box.map50),
        "all/mAP50-95": float(metrics.box.map),
    }
    print(f"{'all':<12}{metrics.box.map50:>12.4f}{metrics.box.map:>12.4f}")

    # names maps class index -> class name; ap_class_index tells which classes
    # actually appeared in the validation set.
    names = getattr(detector, "names", {}) or {}
    class_indices = list(getattr(metrics.box, "ap_class_index", []) or [])
    for row, class_idx in enumerate(class_indices):
        class_name = names.get(int(class_idx), str(class_idx))
        ap50 = float(metrics.box.ap50[row])
        ap = float(metrics.box.ap[row])
        summary[f"{class_name}/mAP50"] = ap50
        summary[f"{class_name}/mAP50-95"] = ap
        print(f"{class_name:<12}{ap50:>12.4f}{ap:>12.4f}")

    missing = sorted(set(names) - {int(i) for i in class_indices})
    if missing:
        print("-" * 62)
        for class_idx in missing:
            print(f"{names.get(class_idx, class_idx):<12}{'no val labels':>24}")
    print("=" * 62)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train / validate the basketball 4-class detector")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train the detector")
    p_train.add_argument("--data", type=Path, required=True, help="Path to basketball.yaml")
    p_train.add_argument("--model", default="yolo26s.pt", help="Base weights (yolo26n/s/m/l/x.pt)")
    p_train.add_argument("--epochs", type=int, default=100)
    p_train.add_argument("--imgsz", type=int, default=960, help="960+ recommended for ball/rim")
    p_train.add_argument("--device", default="0", help='"0", "0,1", or "cpu"')
    p_train.add_argument("--project", type=Path, default=Path("runs/detect/train"))
    p_train.add_argument("--name", default="basketball_baseline")
    p_train.add_argument("--batch", type=int, default=-1, help="-1 = auto batch size")
    p_train.add_argument("--workers", type=int, default=0)
    p_train.add_argument("--patience", type=int, default=50)
    p_train.add_argument("--save-period", dest="save_period", type=int, default=10)

    p_val = sub.add_parser("val", help="Validate a checkpoint, per-class mAP")
    p_val.add_argument("--data", type=Path, required=True, help="Path to basketball.yaml")
    p_val.add_argument("--weights", type=Path, required=True, help="Path to best.pt")
    p_val.add_argument("--imgsz", type=int, default=960)
    p_val.add_argument("--device", default="0")

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "train":
        train_basketball_detector(
            data=args.data,
            model=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            device=args.device,
            project=args.project,
            name=args.name,
            batch=args.batch,
            workers=args.workers,
            patience=args.patience,
            save_period=args.save_period,
        )
    elif args.command == "val":
        validate_model(
            data=args.data,
            weights=args.weights,
            imgsz=args.imgsz,
            device=args.device,
        )


if __name__ == "__main__":
    main()
