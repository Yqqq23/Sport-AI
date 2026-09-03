"""Fetch the SoccerNet jersey-2023 tracklets for stage 3.

    python scripts/download_soccernet.py
    python scripts/download_soccernet.py --root /home/s1143757/datasets/SoccerNet

WHAT THIS DATASET IS FOR
------------------------
2853 player tracklets, numbers 1-99, with tracklets whose number is never
legible labelled -1. Low resolution, motion blurred, players turning away
mid-track - it is the closest public data to a fixed wide 3x3 camera, which
is the only reason it is worth the download. Stage 3 mixes it 1:1 with
synthetic so the model gets both real degradation and full number coverage.

The -1 tracklets are not waste. They are the only source of REAL unreadable
examples anywhere in this pipeline, and they are what teaches the length head
to say "unclear" instead of guessing. A confident wrong guess actively poisons
the cross-frame accumulation downstream; a confident "unclear" costs nothing.

WHY train AND test BUT NOT challenge
------------------------------------
The challenge split's 1211 tracklets have hidden labels - they exist for
leaderboard submission, so downloading them gets bytes and no supervision.
train trains, test validates.

IF THIS ASKS FOR A PASSWORD
---------------------------
Some SoccerNet tasks are behind an NDA form. This script does NOT prompt: it
catches that case and exits with instructions, because a script blocked on
stdin under nohup is a hang with no error message. Stages 1 and 2 do not
depend on this data and can proceed without it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

NDA_HELP = """
SoccerNet asked for a password, which means jersey-2023 is behind their NDA form.

  1. Fill in the form linked from https://www.soccer-net.org/data
  2. They email a password
  3. Re-run with:  python scripts/download_soccernet.py --password '<password>'

Nothing else is blocked by this. Stages 1 (SVHN) and 2 (synthetic) are
independent, and stage 3 can be run later against the same checkpoint.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO / "data" / "soccernet")
    parser.add_argument("--password", default=None, help="NDA password, if they issued one")
    parser.add_argument(
        "--splits", nargs="+", default=["train", "test"],
        help="challenge is deliberately excluded - its labels are hidden",
    )
    args = parser.parse_args()

    if "challenge" in args.splits:
        print(
            "refusing to download the challenge split: its 1211 tracklets have hidden "
            "labels, so it costs bandwidth and teaches nothing.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    try:
        from SoccerNet.Downloader import SoccerNetDownloader
    except ImportError:
        print(
            "The SoccerNet package is not installed. It is needed ONLY for this "
            "download - training itself needs nothing beyond torch/torchvision.\n\n"
            "    pip install SoccerNet\n",
            file=sys.stderr,
        )
        raise SystemExit(1)

    args.root.mkdir(parents=True, exist_ok=True)
    downloader = SoccerNetDownloader(LocalDirectory=str(args.root))
    if args.password:
        downloader.password = args.password

    print(f"downloading jersey-2023 splits={args.splits} -> {args.root}", flush=True)
    try:
        downloader.downloadDataTask(task="jersey-2023", split=args.splits)
    except Exception as error:
        message = str(error).lower()
        if "password" in message or "unauthor" in message or "403" in message:
            print(NDA_HELP, file=sys.stderr)
            raise SystemExit(3)
        raise

    print(f"\ndone. Verify with:\n  python train/finetune_soccernet.py --inspect-only")


if __name__ == "__main__":
    main()
