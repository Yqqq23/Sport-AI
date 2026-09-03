"""Stage 3: fine-tune on SoccerNet mixed 1:1 with synthetic, all three heads.

    python train/finetune_soccernet.py --inspect-only       # check the data first
    python train/finetune_soccernet.py                      # the real run

WHAT CHANGES FROM STAGE 1
-------------------------
Stage 1 taught the backbone what digits look like through a real lens, using
the digit1 head alone. Everything else was left untouched: the length and
digit2 heads are still randomly initialised on entry here, and this is where
they are supervised for the first time.

So this stage is doing three things at once - teaching two brand-new heads,
adapting the backbone from house numbers to jerseys, and doing it on data
where half the labels are tracklet-level approximations. That is why the
backbone is frozen for the first two epochs.

THE 1:1 MIX IS THE POINT
------------------------
Neither source is sufficient alone and they fail in opposite directions.

SoccerNet is real: real motion blur, real fabric, real players turning away
mid-tracklet. But its number distribution is whatever those 2853 tracklets
happened to contain, so the rare numbers are rare there for exactly the same
reason they are rare in our own footage - and a model trained on it alone
inherits that bias.

Synthetic has perfect coverage - 101 numbers, equal counts, by construction -
but it is a font render. Train on it alone and the model learns glyph shapes
that no camera produces.

Mixed 1:1 by CONSTRUCTION rather than by sampling weights, so epoch length is
honest and every epoch sees the same balance. mix_one_to_one truncates the
larger set randomly, which matters for synthetic: its rows are generated
number by number, so a head slice would keep only the low numbers.

WHY THE BACKBONE FREEZES FOR TWO EPOCHS
---------------------------------------
Two of the three heads start from noise. Their first gradients are large and
meaningless, and letting those reach a backbone that already carries SVHN's
digit features destroys the thing stage 1 was for. Two epochs is enough for
the heads to stop flailing; after that everything trains together.

VALIDATION IS SOCCERNET, NOT THE MIX
------------------------------------
Checkpoint selection runs on SoccerNet's held-out test split alone, because
the real domain is the one the model is judged on. Validating on the mix would
let a rising synthetic score - easy, and measured on the same renderer the
training data came from - mask a falling real score.

Synthetic is still scored every epoch as a DIAGNOSTIC. It is what shows
whether the rare numbers are surviving, since SoccerNet's test split cannot
tell you about a digit combination it does not contain. It just does not get a
vote on which weights ship.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from torch.utils.data import DataLoader, Subset

from jersey_data import SoccerNetJersey, SyntheticJersey, mix_one_to_one
from jersey_net import JerseyLoss, JerseyNet
from trainer import train_model

REPO = Path(__file__).resolve().parent.parent


def split_indices(n: int, val_fraction: float, seed: int = 0) -> tuple[list[int], list[int]]:
    """Deterministic index split, so a resumed run validates on the same rows."""
    order = list(range(n))
    random.Random(seed).shuffle(order)
    cut = int(n * (1.0 - val_fraction))
    return order[:cut], order[cut:]


def describe(name: str, dataset) -> None:
    """Print the label composition, because a silent mislabel is expensive.

    Specifically worth seeing: how many unclear samples SoccerNet contributes
    (it should be a large fraction - that is the point), and that the synthetic
    side really does carry all 101 numbers rather than however many survived
    a partial generation run.
    """
    lengths = Counter()
    numbers = set()
    for row in getattr(dataset, "rows", []):
        number = row.get("number")
        lengths[0 if number is None else len(str(number))] += 1
        if number is not None:
            numbers.add(str(number))

    print(
        f"  {name}: {len(dataset)} samples  "
        f"unclear={lengths[0]} one-digit={lengths[1]} two-digit={lengths[2]}  "
        f"distinct numbers={len(numbers)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", type=Path, default=REPO / "data" / "synthetic")
    parser.add_argument("--soccernet", type=Path, default=REPO / "data" / "soccernet")
    parser.add_argument("--init", type=Path, default=REPO / "checkpoints" / "stage1_svhn.pt")
    parser.add_argument("--out", type=Path, default=REPO / "checkpoints" / "stage3_soccernet.pt")
    parser.add_argument("--log", type=Path, default=REPO / "logs" / "stage3_soccernet.log")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--freeze-epochs", type=int, default=2)
    parser.add_argument("--frames-per-tracklet", type=int, default=12)
    parser.add_argument("--synthetic-val-fraction", type=float, default=0.05)
    parser.add_argument(
        "--inspect-only", action="store_true",
        help="load and describe both datasets, then stop without training",
    )
    args = parser.parse_args()

    # SoccerNet's own test split is the validation set: a real held-out set of
    # different players. Carving a slice out of train would be worse even done
    # by tracklet, since both halves would come from the same matches.
    sn_train = SoccerNetJersey(
        args.soccernet, split="train", train=True, frames_per_tracklet=args.frames_per_tracklet
    )
    sn_val = SoccerNetJersey(
        args.soccernet, split="test", train=False, frames_per_tracklet=args.frames_per_tracklet
    )

    syn_all = SyntheticJersey(args.synthetic, train=True)
    syn_val_view = SyntheticJersey(args.synthetic, train=False)
    train_idx, val_idx = split_indices(len(syn_all), args.synthetic_val_fraction)
    syn_train = Subset(syn_all, train_idx)
    syn_val = Subset(syn_val_view, val_idx)

    print("datasets:")
    describe("soccernet/train", sn_train)
    describe("soccernet/test ", sn_val)
    describe("synthetic (all)", syn_all)
    print(f"  synthetic split: train={len(syn_train)} val={len(syn_val)}")
    print(f"  soccernet train tracklets: {len(sn_train.tracklet_ids())}")

    mixed = mix_one_to_one(syn_train, sn_train)
    print(f"\nmixed 1:1 -> {len(mixed)} samples "
          f"({min(len(syn_train), len(sn_train))} from each side)")

    if args.inspect_only:
        print("\nstopping (--inspect-only)")
        return

    if not args.init.exists():
        raise SystemExit(
            f"stage 1 checkpoint not found: {args.init}\n"
            f"Run train/pretrain_svhn.py first - starting from ImageNet alone "
            f"throws away the digit prior this stage is built on."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin = device.type == "cuda"

    def loader(dataset, shuffle: bool, drop_last: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.workers,
            pin_memory=pin,
            drop_last=drop_last,
            persistent_workers=args.workers > 0,
        )

    model = JerseyNet(pretrained=False)
    state = torch.load(args.init, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    print(f"\nloaded stage 1 weights from {args.init}")
    print("  length and digit2 heads were never supervised in stage 1 - "
          "they start from their random init here")

    # All three heads weighted equally. The imbalance that matters is handled
    # by ignore_index, not by these weights: digit2 simply sees fewer
    # supervised targets than digit1 because one-digit numbers have no second
    # digit, and that is correct rather than something to compensate for.
    criterion = JerseyLoss(w_len=1.0, w_digit1=1.0, w_digit2=1.0)

    train_model(
        model,
        loader(mixed, shuffle=True, drop_last=True),
        loader(sn_val, shuffle=False),
        criterion,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        out_path=args.out,
        stage="stage3_soccernet",
        data_version=(
            f"synthetic({len(syn_train)}) + soccernet-jersey-2023({len(sn_train)}) "
            f"mixed 1:1 -> {len(mixed)}; val=soccernet/test({len(sn_val)})"
        ),
        freeze_backbone_epochs=args.freeze_epochs,
        log_file=args.log,
        diagnostic_loaders={"synthetic-val": loader(syn_val, shuffle=False)},
        extra_sidecar={
            "trained_heads": ["length", "digit1", "digit2"],
            "initialised_from": str(args.init),
            "selection_val": "soccernet/test only - synthetic is diagnostic",
            "frames_per_tracklet": args.frames_per_tracklet,
            "note": (
                "SoccerNet labels are per-tracklet, so frames where the player "
                "has turned carry the tracklet's number as label. That noise is "
                "expected and is absorbed by cross-frame accumulation at inference."
            ),
        },
    )


if __name__ == "__main__":
    main()
