"""Datasets for the three training stages, all behind one interface.

Every dataset here yields the same 4-tuple::

    (image_tensor, length, digit1, digit2)

with the three targets already expanded by :func:`jersey_net.expand_label` and
``-1`` meaning "this head learns nothing from this sample". That uniformity is
what lets the stage-3 loader concatenate synthetic and SoccerNet rows without
either side knowing the other exists, and what lets one training loop and one
evaluation pass serve all three stages.

The datasets differ enormously in what they can teach:

    SVHN        ~73k real photographed digits, single digit each. Teaches
                digit SHAPE under real optics. Has no concept of length, so it
                supervises digit1 only and leaves the other two heads alone.

    synthetic   101 numbers x N, the ONLY source that covers the full legal
                range. Teaches the length distinction and the rare numbers no
                real footage will ever contain enough of.

    SoccerNet   2853 real tracklets, low-res and motion blurred - the closest
                public data to a wide 3x3 camera. Teaches what a jersey number
                actually looks like on a moving athlete, including the large
                fraction of frames where it is not readable at all.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset

from augment import build_transforms
from jersey_net import expand_label


class JerseyDatasetMixin:
    """Shared ``__getitem__`` tail: expand a raw number into the 4-tuple."""

    def _emit(self, image: Image.Image, number: str | int | None):
        length, digit1, digit2 = expand_label(number)
        return self.transform(image), length, digit1, digit2


# --------------------------------------------------------------------------
# Stage 1 - SVHN
# --------------------------------------------------------------------------


class SVHNDigits(Dataset, JerseyDatasetMixin):
    """torchvision's SVHN, reshaped to this project's target tuple.

    SVHN crops are single digits, so ``length`` and ``digit2`` are emitted as
    -1 and the corresponding heads see nothing. This is deliberate rather than
    a limitation to work around: the length head has no meaningful target here
    (a house-number crop is not a one- or two-digit jersey), and inventing one
    would teach it something false. It stays randomly initialised until stage
    3, which is where length actually gets supervised.

    torchvision already remaps SVHN's original label 10 to 0, so the labels
    arriving here are honest 0-9.
    """

    def __init__(self, root: str | Path, split: str = "train", train: bool = True, download: bool = False) -> None:
        import torchvision

        self.inner = torchvision.datasets.SVHN(
            root=str(root), split=split, download=download
        )
        self.transform = build_transforms(train)

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, index: int):
        image, label = self.inner[index]
        image = image.convert("RGB")
        # Bypass _emit: the label is already a bare digit, and routing it
        # through expand_label would make it a one-digit *number* and start
        # training the length head on house numbers.
        return self.transform(image), -1, int(label), -1


# --------------------------------------------------------------------------
# Stage 2 - synthetic
# --------------------------------------------------------------------------


class SyntheticJersey(Dataset, JerseyDatasetMixin):
    """Rendered numbers from scripts/gen_synthetic.py, read via its manifest.

    The manifest is the contract: this class never re-derives a label from a
    filename, so renaming files or changing the directory layout cannot
    silently corrupt the labels.
    """

    def __init__(self, root: str | Path, train: bool = True, numbers: set[str] | None = None) -> None:
        self.root = Path(root)
        manifest = self.root / "manifest.json"
        if not manifest.exists():
            raise FileNotFoundError(
                f"{manifest} not found - run scripts/gen_synthetic.py first"
            )
        rows = json.loads(manifest.read_text(encoding="utf-8"))["rows"]
        self.rows = [r for r in rows if numbers is None or r["number"] in numbers]
        self.transform = build_transforms(train)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = Image.open(self.root / row["image"]).convert("RGB")
        return self._emit(image, row["number"])


# --------------------------------------------------------------------------
# Stage 3 - SoccerNet
# --------------------------------------------------------------------------


class SoccerNetJersey(Dataset, JerseyDatasetMixin):
    """SoccerNet jersey-2023 tracklets, flattened to individual frames.

    The ground truth labels a TRACKLET, not a frame: one number (or -1 for
    "never legible in this tracklet") for a folder of crops. Flattening to
    frames is the right call anyway because the model is a single-frame model,
    but it means a tracklet labelled "23" contributes frames where the player
    has turned and the number is genuinely invisible. Those mislabelled frames
    are noise the augmentation and the cross-frame accumulation are expected to
    absorb; there is no per-frame ground truth available to do better.

    ``frames_per_tracklet`` caps how many frames each tracklet contributes, so
    a 400-frame tracklet cannot outvote a 30-frame one. Without it the loss is
    dominated by whichever players happened to be tracked longest.

    Tracklets labelled -1 become unclear samples and train the length head on
    class 0 - that is the only source of real unreadable examples in the whole
    pipeline, and it is a large one.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        train: bool = True,
        frames_per_tracklet: int = 12,
        seed: int = 0,
    ) -> None:
        self.root = Path(root)
        self.transform = build_transforms(train)
        self.rows: list[dict] = []

        images_dir, gt = self._locate(split)
        rng = random.Random(seed)

        for tracklet_id, number in gt.items():
            folder = images_dir / str(tracklet_id)
            if not folder.is_dir():
                continue
            frames = sorted(
                p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )
            if not frames:
                continue
            if len(frames) > frames_per_tracklet:
                frames = rng.sample(frames, frames_per_tracklet)
            # -1 in the ground truth is exactly what expand_label treats as
            # unclear, so no translation is needed here.
            label = None if int(number) < 0 else str(int(number))
            for frame in frames:
                self.rows.append({"image": frame, "number": label, "tracklet": str(tracklet_id)})

    def _locate(self, split: str) -> tuple[Path, dict]:
        """Find the images dir and ground-truth json for ``split``.

        The downloader's exact layout has moved between releases, so probe the
        handful of shapes it has used rather than hardcoding one and failing
        with a confusing FileNotFoundError three weeks from now.
        """
        candidates = [
            self.root / "jersey-2023" / split,
            self.root / split,
            self.root / "jersey-2023" / "jersey-2023" / split,
        ]
        for base in candidates:
            if not base.is_dir():
                continue
            images = base / "images"
            if not images.is_dir():
                images = base
            for name in (f"{split}_gt.json", "gt.json", f"{split}_gt.txt"):
                gt_path = base / name
                if gt_path.exists():
                    return images, json.loads(gt_path.read_text(encoding="utf-8"))
        raise FileNotFoundError(
            f"no SoccerNet jersey-2023 '{split}' split under {self.root}. "
            f"Tried: {[str(c) for c in candidates]}"
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = Image.open(row["image"]).convert("RGB")
        return self._emit(image, row["number"])

    def tracklet_ids(self) -> list[str]:
        """Used to split train/val BY TRACKLET.

        Splitting by frame would put near-duplicate crops of the same player on
        both sides and report a validation score that is mostly memorisation.
        """
        return sorted({row["tracklet"] for row in self.rows})


# --------------------------------------------------------------------------
# Mixing
# --------------------------------------------------------------------------


def mix_one_to_one(dataset_a: Dataset, dataset_b: Dataset, seed: int = 0) -> Dataset:
    """Concatenate two datasets after truncating the larger to match.

    A 1:1 mix by CONSTRUCTION rather than by sampling weights, so the epoch
    length is honest and every epoch sees the same balance. Truncation is
    random per call rather than a head slice, which for the synthetic set
    matters - its rows are generated number by number, so a head slice would
    keep only the low numbers.
    """
    n = min(len(dataset_a), len(dataset_b))
    rng = random.Random(seed)

    def clip(dataset):
        if len(dataset) == n:
            return dataset
        return torch.utils.data.Subset(dataset, rng.sample(range(len(dataset)), n))

    return ConcatDataset([clip(dataset_a), clip(dataset_b)])


# --------------------------------------------------------------------------
# Review artifacts
# --------------------------------------------------------------------------


def save_contact_sheet(
    samples: list[tuple[Image.Image, str]],
    path: str | Path,
    cols: int = 5,
    cell: int = 128,
    pad: int = 22,
) -> Path:
    """Tile labelled samples into one PNG for a human to eyeball.

    Every stage writes one of these before training on the data. Automated
    checks confirm a dataset downloaded and parsed; they cannot tell you the
    degradation is so severe a person could not read the number either, which
    is the specific failure mode that silently wastes a training run.
    """
    from PIL import ImageDraw

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = (len(samples) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * (cell + pad)), (24, 24, 28))
    draw = ImageDraw.Draw(sheet)

    for i, (image, label) in enumerate(samples):
        r, c = divmod(i, cols)
        x, y = c * cell, r * (cell + pad)
        # NEAREST so the reviewer sees the actual pixels the model gets, not a
        # smoothed version that looks sharper than the data really is.
        sheet.paste(image.convert("RGB").resize((cell, cell), Image.NEAREST), (x, y))
        draw.text((x + 4, y + cell + 5), label, fill=(235, 235, 235))

    sheet.save(path)
    return path
