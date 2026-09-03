# =============================================================================
# DEPRECATED - THIS MODULE RAISES ON IMPORT. DO NOT USE IT.
#
# This is the old 41-class CLOSED-SET classifier. Its central premise - that
# the set of jersey numbers is known ahead of time and can be enumerated as a
# class list - is false.
#
# The number set is OPEN. FIBA 3x3 allows 0, 00 and 1-99: 101 legal numbers,
# four players per team, no duplicates within a team. Any one match shows eight
# of them and a different eight the next match. A model with a fixed class list
# cannot name a number outside that list, and one given all 101 classes leaves
# 93 of them with zero training examples - where an untrained logit still wins
# an argmax sometimes, manufacturing confident predictions for numbers the
# model has never seen.
#
# REPLACEMENT: train/jersey_net.py - a per-digit model. Three heads (length,
# digit1, digit2) on a shared trunk, so an image labelled "23" trains both
# "this is a 2" and "this is a 3", and any of the 101 legal numbers can be
# emitted without ever having appeared in training. Whole numbers are
# reassembled downstream by accumulating per-digit probabilities across frames.
#
# The file is kept rather than deleted because its augmentation docstrings
# record why RandomDownscale and the flip ban exist, and that reasoning carried
# forward into train/augment.py. It is a document now, not code.
#
# The raise below is deliberate and is not a placeholder. A comment stops
# nobody - not a colleague skimming the tree, not an agent grepping for
# "train jersey". Only an exception does.
# =============================================================================

"""Train the local jersey-number classifier.  [DEPRECATED - see header above]

    python train/train_jersey.py --data data/jersey_ocr --epochs 40

Produces ``jersey_classifier.pt``: a self-describing checkpoint carrying the
weights, the class list, the input size and the normalization constants, so
``src/jersey.py`` never has to be told how the model was trained.

Run scripts/download_jersey_ocr_dataset.py first; this script reads the cleaned
``manifest.json`` it writes and never re-derives labels.

WHY THIS IS A CLASSIFIER AND NOT AN OCR MODEL
---------------------------------------------
Jersey numbers are a CLOSED SET. There is no value in a model that can emit
arbitrary strings when the only legal answers are a few dozen numbers - a
sequence model just gives itself more ways to be wrong. Closed-set
classification also yields a calibrated per-number confidence, which is exactly
what the cross-frame voting in src/jersey.py consumes.

THE CLASS LIST COMES FROM THE DATA
----------------------------------
41 numbers survive cleaning, not 100. A 0-99 head would leave 59 classes with
zero training examples, and an untrained logit will still win an argmax
sometimes - producing confident predictions for numbers the model has literally
never seen. If your footage contains a number outside the trained list, the fix
is more DATA, not more classes; the checkpoint records the list so the pipeline
can report the gap instead of hiding it.

THE AUGMENTATION IS THE WHOLE GAME
----------------------------------
The source data is NBA broadcast footage cropped to 224x224 - close, sharp,
well-lit. The deployment target is a fixed wide 3x3 camera where the same number
occupies 20-40 px and is frequently motion blurred. Training on the source
domain as-is produces a model with excellent validation accuracy that collapses
on real input.

``RandomDownscale`` is therefore not a nicety, it is the point: every sample is
crushed to roughly the resolution the pipeline actually sees, then restored, so
the network is forced to learn features that survive that trip. Same reasoning
for the blur, the JPEG artifacts, and the bbox jitter (which reproduces the
detector's mAP50-95 ~0.64 localization error).

NOTE THE ABSENCE OF A HORIZONTAL FLIP. Digits are not mirror-invariant; flipping
turns a 2 into a shape that is not a 2 while keeping the label "2". It is in
almost every default augmentation recipe and it is wrong here.
"""

from __future__ import annotations

DEPRECATED = """\
train/train_jersey.py is DEPRECATED and must not be used.

It is the old 41-class closed-set jersey classifier. The jersey number set is
OPEN - FIBA 3x3 allows 0, 00 and 1-99 (101 legal numbers) - so a fixed class
list cannot name a number it was not trained on, and padding the list out to
101 classes leaves 93 of them untrained but still able to win an argmax.

Use the per-digit model instead:

    train/jersey_net.py        JerseyNet - length / digit1 / digit2 heads
    train/pretrain_svhn.py     stage 1
    train/finetune_soccernet.py stage 3

An image labelled "23" trains both "this is a 2" and "this is a 3", so every
sample contributes to all ten digit shapes and any legal number can be emitted
without ever having appeared in training.

--
train/train_jersey.py 已废弃,请勿使用。
此模块为已废弃的 41 类闭集分类器,号码集合是开放的(FIBA 3x3 允许 0/00/1-99),
请改用 train/jersey_net.py 的逐位数字模型。
"""

raise RuntimeError(DEPRECATED)

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# Matches jersey.OCRGateConfig.upscale_to_height, so the model trains at the
# exact size the pipeline hands it at inference time.
INPUT_SIZE = 96
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------
# Domain-gap augmentation
# --------------------------------------------------------------------------


class RandomDownscale:
    """Crush the image to a realistic capture resolution, then restore it.

    THE most important transform here. A 3x3 wide-camera jersey number is
    20-40 px tall; the training images are 224 px. Without this the model learns
    fine strokes that simply do not exist in deployment.
    """

    def __init__(self, min_px: int = 14, max_px: int = 48, probability: float = 0.9) -> None:
        self.min_px = min_px
        self.max_px = max_px
        self.probability = probability

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.probability:
            return image
        target = random.randint(self.min_px, self.max_px)
        small = image.resize((target, target), Image.BILINEAR)
        return small.resize((INPUT_SIZE, INPUT_SIZE), Image.BICUBIC)


class RandomMotionBlur:
    """Directional blur, standing in for a player moving during exposure."""

    def __init__(self, max_kernel: int = 7, probability: float = 0.35) -> None:
        self.max_kernel = max_kernel
        self.probability = probability

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.probability:
            return image
        import cv2

        array = np.array(image)
        size = random.choice([3, 5, self.max_kernel])
        kernel = np.zeros((size, size), dtype=np.float32)
        if random.random() < 0.5:
            kernel[size // 2, :] = 1.0
        else:
            kernel[:, size // 2] = 1.0
        kernel /= kernel.sum()
        return Image.fromarray(cv2.filter2D(array, -1, kernel))


class RandomJPEG:
    """Recompression artifacts, as produced by the video decode path."""

    def __init__(self, min_quality: int = 25, probability: float = 0.35) -> None:
        self.min_quality = min_quality
        self.probability = probability

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.probability:
            return image
        import io

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=random.randint(self.min_quality, 85))
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


def build_transforms(train: bool):
    from torchvision import transforms

    if not train:
        return transforms.Compose([
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

    return transforms.Compose([
        # Reproduces the detector's bbox jitter plus this project's pad_ratio.
        transforms.RandomResizedCrop(
            INPUT_SIZE, scale=(0.65, 1.0), ratio=(0.75, 1.33)
        ),
        transforms.RandomRotation(10, fill=0),
        RandomDownscale(),
        RandomMotionBlur(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05),
        RandomJPEG(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.12)),
    ])


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


class JerseyDataset(Dataset):
    def __init__(self, root: Path, rows: list[dict], classes: list[str], train: bool) -> None:
        self.root = root
        self.rows = rows
        self.class_to_index = {c: i for i, c in enumerate(classes)}
        self.transform = build_transforms(train)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = Image.open(self.root / row["image"]).convert("RGB")
        return self.transform(image), self.class_to_index[row["label"]]


def make_sampler(rows: list[dict], classes: list[str]) -> WeightedRandomSampler:
    """Balance the 315:2 class imbalance by sampling, not by loss weighting.

    Weighting the loss leaves rare classes seen just as rarely; oversampling
    actually shows them to the network, which matters far more when the rarest
    classes have single-digit example counts.
    """
    counts = Counter(row["label"] for row in rows)
    weights = [1.0 / counts[row["label"]] for row in rows]
    return WeightedRandomSampler(weights, num_samples=len(rows), replacement=True)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------


def build_model(num_classes: int):
    """ImageNet-pretrained ResNet18 with a fresh head.

    2.9k training images is far too little to train a backbone from scratch;
    pretrained low-level filters (edges, strokes) transfer to digits perfectly
    well. ResNet18 at 96x96 runs in ~1-2 ms per crop on any modern GPU, which is
    what makes local inference viable at video rates - the entire reason for
    replacing a hosted endpoint that cost 1.3-6.8 SECONDS per call.
    """
    from torchvision import models

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# --------------------------------------------------------------------------
# Train / evaluate
# --------------------------------------------------------------------------


@torch.no_grad()
def evaluate(model, loader, device, classes) -> tuple[float, dict[str, float]]:
    model.eval()
    correct = 0
    total = 0
    per_class_hit: Counter = Counter()
    per_class_total: Counter = Counter()
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        predictions = model(images).argmax(1)
        correct += (predictions == targets).sum().item()
        total += targets.numel()
        for target, prediction in zip(targets.tolist(), predictions.tolist()):
            per_class_total[target] += 1
            per_class_hit[target] += int(target == prediction)
    accuracy = correct / max(1, total)
    per_class = {
        classes[i]: per_class_hit[i] / per_class_total[i]
        for i in sorted(per_class_total)
    }
    return accuracy, per_class


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the jersey number classifier")
    # Second guard, and not redundant. The module-level raise is what actually
    # fires today; this one survives someone deleting that line to "just run it
    # once" - which is precisely how a deprecated closed-set model gets trained
    # and quietly shipped.
    raise RuntimeError(DEPRECATED)

    parser.add_argument("--data", type=Path, default=Path("data/jersey_ocr"))
    parser.add_argument("--out", type=Path, default=Path("train/jersey_classifier.pt"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    manifest_path = args.data / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(
            f"No manifest at {manifest_path}. Run:\n"
            f"  python scripts/download_jersey_ocr_dataset.py --out {args.data}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    classes = manifest["classes"]
    splits = manifest["splits"]

    print(f"classes ({len(classes)}): {' '.join(classes)}")
    print(f"train {len(splits['train'])}  valid {len(splits['valid'])}  test {len(splits['test'])}")

    train_set = JerseyDataset(args.data, splits["train"], classes, train=True)
    valid_set = JerseyDataset(args.data, splits["valid"], classes, train=False)
    test_set = JerseyDataset(args.data, splits["test"], classes, train=False)

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size,
        sampler=make_sampler(splits["train"], classes),
        num_workers=args.workers, pin_memory=True, drop_last=True,
    )
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, num_workers=args.workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.workers)

    device = torch.device(args.device)
    model = build_model(len(classes)).to(device)
    # Label smoothing: several classes are visually confusable at 20 px (6/8,
    # 3/8, 1/7) and an overconfident wrong vote is worse for the downstream
    # stabilizer than a hesitant right one.
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr * 10, epochs=args.epochs, steps_per_epoch=len(train_loader)
    )

    best_accuracy = 0.0
    args.out.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device, non_blocking=True), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), targets)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running += loss.item()

        accuracy, _ = evaluate(model, valid_loader, device, classes)
        marker = ""
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            marker = "  <- best, saved"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "classes": classes,
                    "input_size": INPUT_SIZE,
                    "mean": MEAN,
                    "std": STD,
                    "arch": "resnet18",
                    "val_accuracy": accuracy,
                    "source": manifest["source"],
                },
                args.out,
            )
        print(
            f"epoch {epoch:3d}/{args.epochs}  loss {running/max(1,len(train_loader)):.4f}"
            f"  val_acc {accuracy:.4f}{marker}"
        )

    print(f"\nbest validation accuracy: {best_accuracy:.4f}")

    checkpoint = torch.load(args.out, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    test_accuracy, per_class = evaluate(model, test_loader, device, classes)
    print(f"test accuracy: {test_accuracy:.4f}\n")
    print("weakest classes on test:")
    for label, acc in sorted(per_class.items(), key=lambda kv: kv[1])[:10]:
        print(f"  #{label:<3s} {acc:.3f}   (train examples: {manifest['class_counts'][label]})")

    print(f"\nsaved {args.out}")
    print("\nUse it:")
    print("  jersey:")
    print("    backend: local")
    print(f"    local: {{ weights: {args.out} }}")


if __name__ == "__main__":
    main()
