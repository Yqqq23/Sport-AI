"""Stage 1: pretrain the digit1 head and the backbone on SVHN.

    python train/pretrain_svhn.py --download          # first run, needs network
    python train/pretrain_svhn.py --epochs 20         # offline afterwards

    # sanity-check the download before spending a GPU on it:
    python train/pretrain_svhn.py --download --sample-only

WHY SVHN
--------
The model needs to know what digits look like through a real lens before it is
asked anything harder, and there is no jersey dataset large enough to teach
that. SVHN is ~600k photographed house-number digits (73k in the standard
train split, 531k more in `extra`) - genuinely blurred, skewed, badly lit,
partially occluded, shot at every angle. That is the right prior: digit
identity under real optical degradation, learned from real photons rather than
from a font renderer.

What SVHN cannot teach is anything jersey-specific - it has no notion of a
two-digit number as a unit, no 00, no unreadable case, and its digits sit on
house walls rather than on moving fabric. So this stage trains ONLY the digit1
head. The length and digit2 heads stay randomly initialised and are supervised
for the first time in stage 3, which is why their loss terms are ignored here
rather than being fed some invented target.

Concretely: SVHNDigits emits length=-1 and digit2=-1, JerseyLoss's
ignore_index drops both terms, and only digit1's gradient reaches the shared
backbone. The backbone is what actually carries forward - the digit1 head's
weights are a bonus.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from torch.utils.data import DataLoader

from jersey_data import SVHNDigits, save_contact_sheet
from jersey_net import JerseyLoss, JerseyNet
from trainer import train_model

REPO = Path(__file__).resolve().parent.parent


def sample_sheet(root: Path, out: Path, n: int = 20) -> Path:
    """Tile n raw SVHN crops for a human to confirm the download is intact.

    A corrupted or half-downloaded .mat still loads and still yields tensors of
    the right shape; the training loop would happily consume garbage for 20
    epochs. Looking at twenty images costs seconds and rules that out.
    """
    import torchvision

    dataset = torchvision.datasets.SVHN(root=str(root), split="train", download=False)
    picks = random.Random(0).sample(range(len(dataset)), n)
    samples = [(dataset[i][0].convert("RGB"), f"label {int(dataset[i][1])}") for i in picks]
    return save_contact_sheet(samples, out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=REPO / "data" / "svhn")
    parser.add_argument("--out", type=Path, default=REPO / "checkpoints" / "stage1_svhn.pt")
    parser.add_argument("--log", type=Path, default=REPO / "logs" / "stage1_svhn.log")
    parser.add_argument("--sheet", type=Path, default=REPO / "data" / "debug" / "svhn_sample.png")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--download", action="store_true", help="fetch SVHN (needs network)")
    parser.add_argument(
        "--sample-only",
        action="store_true",
        help="write the contact sheet and stop, before committing a GPU to this",
    )
    args = parser.parse_args()

    args.data.mkdir(parents=True, exist_ok=True)

    # SVHN's own test split is the validation set. It is a real held-out split
    # from different houses, which is a far better check than carving a random
    # slice out of train - SVHN crops from one house number are near-duplicates
    # and a random split would leak them across the boundary.
    train_set = SVHNDigits(args.data, split="train", train=True, download=args.download)
    val_set = SVHNDigits(args.data, split="test", train=False, download=args.download)

    sheet = sample_sheet(args.data, args.sheet)
    print(f"wrote {sheet} - LOOK AT THIS before trusting the run", flush=True)
    if args.sample_only:
        print(f"train={len(train_set)} val={len(val_set)}; stopping (--sample-only)")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin = device.type == "cuda"

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=pin,
        drop_last=True,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin,
        persistent_workers=args.workers > 0,
    )

    model = JerseyNet(pretrained=True)

    # Weight length to 0 as documentation, not as mechanism: the targets are
    # all -1 so the term is already dropped. If someone later mounts a dataset
    # here that does carry length, this line is what stops it from quietly
    # training a head this stage has no business touching.
    criterion = JerseyLoss(w_len=0.0, w_digit1=1.0, w_digit2=0.0)

    train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        out_path=args.out,
        stage="stage1_svhn",
        data_version=f"SVHN train={len(train_set)} test={len(val_set)}",
        log_file=args.log,
        extra_sidecar={
            "trained_heads": ["digit1"],
            "untrained_heads": ["length", "digit2"],
            "note": (
                "digit1 and the backbone only. length/digit2 are randomly "
                "initialised here and first supervised in stage 3."
            ),
        },
    )


if __name__ == "__main__":
    main()
