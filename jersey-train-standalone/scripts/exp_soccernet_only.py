"""Control experiment: train on SoccerNet ALONE, no synthetic. 10 epochs.

    python scripts/exp_soccernet_only.py

Answers one question and only one: how much of the 0.27 is the synthetic mix
poisoning the model, and how much is the ceiling of the SoccerNet data as this
pipeline currently loads it?

  result >> 0.27  the mix is the problem. Synthetic is dominating the shared
                  trunk, and the fix is on the mixing/ratio/data side.

  result ~= 0.27  the mix is NOT the problem. The ceiling is in how SoccerNet
                  itself is loaded and labelled, and re-tuning the ratio would
                  waste a week. Look at Q1 (geometry) and Q2 (label noise).

Deliberately a SEPARATE script rather than a flag on finetune_soccernet.py:
this is an experiment, and the stage-3 recipe should not grow options that
exist to answer a one-off question. Nothing under train/ is modified.

Everything else is held identical to stage 3 - same init, same optimizer, same
schedule shape, same validation set, same selection metric - so the only
variable is the presence of synthetic data.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))

import torch
from torch.utils.data import DataLoader

from jersey_data import SoccerNetJersey
from jersey_net import JerseyLoss, JerseyNet
from trainer import train_model

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--soccernet", type=Path, default=REPO / "data" / "soccernet")
    parser.add_argument("--init", type=Path, default=REPO / "checkpoints" / "stage1_svhn.pt")
    parser.add_argument("--out", type=Path,
                        default=REPO / "checkpoints" / "exp_soccernet_only.pt")
    parser.add_argument("--log", type=Path, default=REPO / "logs" / "exp_soccernet_only.log")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--freeze-epochs", type=int, default=2)
    parser.add_argument("--frames-per-tracklet", type=int, default=12)
    args = parser.parse_args()

    train_set = SoccerNetJersey(args.soccernet, split="train", train=True,
                                frames_per_tracklet=args.frames_per_tracklet)
    val_set = SoccerNetJersey(args.soccernet, split="test", train=False,
                              frames_per_tracklet=args.frames_per_tracklet)

    print(f"SoccerNet only: train={len(train_set)} val={len(val_set)}")
    print(f"  (stage 3 for comparison trained on a 1:1 mix, 533 batches/epoch)")

    if not args.init.exists():
        raise SystemExit(f"missing {args.init} - this must start from the same "
                         f"stage 1 weights as the run it is being compared to")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin = device.type == "cuda"

    def loader(dataset, shuffle: bool, drop_last: bool = False) -> DataLoader:
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=args.workers, pin_memory=pin,
                          drop_last=drop_last, persistent_workers=args.workers > 0)

    model = JerseyNet(pretrained=False)
    model.load_state_dict(torch.load(args.init, map_location="cpu", weights_only=True))

    train_model(
        model,
        loader(train_set, shuffle=True, drop_last=True),
        loader(val_set, shuffle=False),
        JerseyLoss(),
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        out_path=args.out,
        stage="exp_soccernet_only",
        data_version=f"CONTROL: soccernet-only({len(train_set)}), no synthetic",
        freeze_backbone_epochs=args.freeze_epochs,
        log_file=args.log,
        extra_sidecar={
            "experiment": "control - isolates the effect of the synthetic mix",
            "compare_against": "stage3_soccernet.pt (1:1 mix), selection metric 0.2746",
        },
    )


if __name__ == "__main__":
    main()
