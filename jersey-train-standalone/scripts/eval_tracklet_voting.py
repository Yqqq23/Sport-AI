"""What matters is TRACKLET accuracy, not single-frame accuracy - measure it.

    python scripts/eval_tracklet_voting.py --checkpoint checkpoints/stage3_soccernet.pt

WHY THIS SCRIPT EXISTS
-----------------------
Four separate configurations (torso crop alone, torso crop + old synthetic mix
at 10 epochs, torso crop + length-balanced synthetic mix at 30 epochs) all
landed the SINGLE-FRAME selection metric in the same 0.25-0.28 band. Geometry,
epoch budget, and synthetic length balance have all been tried and none of
them move that number. Before spending more effort tuning stage 3 further,
it is worth checking whether 0.25-0.28 per frame is actually a problem.

It might not be. The whole point of accumulating votes across frames -
SlotTracker's job at inference time - is that the DEPLOYED system was never
designed to need a highly accurate single-frame classifier. A digit head with
28% per-digit macro-F1 is well above the ~10% chance floor; averaged over the
dozens of frames a real tracklet contains, plus the no-duplicate-within-a-team
constraint SlotTracker applies on top, the number that actually gets reported
for a PLAYER could be far more accurate than any single crop of them.

This script does not implement SlotTracker (that is explicitly out of scope
for this training package). It does the simplest possible version of the same
idea - average per-frame softmax probabilities across every frame of a
tracklet, then argmax - and reports the result at the TRACKLET level, next to
the single-frame number, so the two can be compared directly. If tracklet
accuracy is dramatically higher, that is the answer: stop tuning stage 3, this
checkpoint is probably already usable, and effort belongs in stage 4 (real
footage) instead. If tracklet accuracy is ALSO stuck low, that says the
per-frame noise floor is high enough that no amount of averaging fixes it -
which points squarely at SoccerNet's label noise (Q2: uniform random frame
sampling, no orientation filtering) as the real remaining bottleneck.

TWO AGGREGATION STRATEGIES, REPORTED SIDE BY SIDE
--------------------------------------------------
  probability averaging   sum softmax probabilities across all frames, argmax
                           the sum. This is the simple version of what a real
                           accumulator does - frames that are more confident
                           pull the average further, so a handful of clear
                           frames can outweigh many ambiguous ones.

  majority vote            argmax each frame independently, then take the
                           most common per-frame answer. Cruder - a confident
                           frame and a barely-guessing frame count equally -
                           included as a floor: if probability averaging beats
                           it by a lot, confidence calibration is doing real
                           work, not just volume of frames.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from jersey_data import SoccerNetJersey
from jersey_net import expand_label, load_checkpoint

REPO = Path(__file__).resolve().parent.parent


def collect_frame_probs(model, dataset: SoccerNetJersey, device: torch.device, batch_size: int):
    """One batched forward pass over every frame. Order matches dataset.rows."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    len_probs, d1_probs, d2_probs = [], [], []

    model.eval()
    with torch.no_grad():
        for images, _t_len, _t_d1, _t_d2 in loader:
            logit_len, logit_d1, logit_d2 = model(images.to(device))
            len_probs.append(F.softmax(logit_len, dim=1).cpu())
            d1_probs.append(F.softmax(logit_d1, dim=1).cpu())
            d2_probs.append(F.softmax(logit_d2, dim=1).cpu())

    return torch.cat(len_probs), torch.cat(d1_probs), torch.cat(d2_probs)


def exact_match(pred_len: int, pred_d1: int, pred_d2: int, true_len: int, true_d1: int, true_d2: int) -> bool:
    if pred_len != true_len:
        return False
    if true_len == 0:
        return True
    if pred_d1 != true_d1:
        return False
    return true_len == 1 or pred_d2 == true_d2


def evaluate(dataset: SoccerNetJersey, len_probs, d1_probs, d2_probs, min_frames: int) -> dict:
    """Group frame-level probabilities by tracklet, aggregate, score both ways."""
    by_tracklet: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(dataset.rows):
        by_tracklet[row["tracklet"]].append(i)

    n_frame_hits = n_frame_total = 0
    n_avg_hits = n_vote_hits = n_tracklets = 0
    skipped_short = 0
    by_frame_count: dict[str, list[bool]] = defaultdict(list)  # bucket -> [avg hit?]

    for tracklet_id, indices in by_tracklet.items():
        if len(indices) < min_frames:
            skipped_short += 1
            continue

        true_number = dataset.rows[indices[0]]["number"]
        true_len, true_d1, true_d2 = expand_label(true_number)
        n_tracklets += 1

        # --- single-frame baseline: does each frame, scored alone, get it right ---
        for i in indices:
            p_len = int(len_probs[i].argmax())
            p_d1 = int(d1_probs[i].argmax())
            p_d2 = int(d2_probs[i].argmax())
            n_frame_total += 1
            n_frame_hits += exact_match(p_len, p_d1, p_d2, true_len, true_d1, true_d2)

        # --- probability averaging ---
        avg_len = len_probs[indices].mean(dim=0)
        avg_d1 = d1_probs[indices].mean(dim=0)
        avg_d2 = d2_probs[indices].mean(dim=0)
        avg_hit = exact_match(int(avg_len.argmax()), int(avg_d1.argmax()), int(avg_d2.argmax()),
                              true_len, true_d1, true_d2)
        n_avg_hits += avg_hit

        # --- majority vote ---
        vote_len = Counter(int(len_probs[i].argmax()) for i in indices).most_common(1)[0][0]
        vote_d1 = Counter(int(d1_probs[i].argmax()) for i in indices).most_common(1)[0][0]
        vote_d2 = Counter(int(d2_probs[i].argmax()) for i in indices).most_common(1)[0][0]
        n_vote_hits += exact_match(vote_len, vote_d1, vote_d2, true_len, true_d1, true_d2)

        bucket = "2-5" if len(indices) <= 5 else ("6-9" if len(indices) <= 9 else "10+")
        by_frame_count[bucket].append(avg_hit)

    return {
        "n_tracklets": n_tracklets,
        "skipped_short": skipped_short,
        "single_frame_exact": n_frame_hits / max(n_frame_total, 1),
        "prob_avg_exact": n_avg_hits / max(n_tracklets, 1),
        "majority_vote_exact": n_vote_hits / max(n_tracklets, 1),
        "by_frame_count": {
            bucket: (sum(hits) / len(hits), len(hits)) for bucket, hits in by_frame_count.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=REPO / "checkpoints" / "stage3_soccernet.pt")
    parser.add_argument("--soccernet", type=Path, default=REPO / "data" / "soccernet")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--frames-per-tracklet", type=int, default=30,
        help="higher than training's 12 on purpose - this measures what a real "
             "deployment sees (every frame in a possession), not the training cap",
    )
    parser.add_argument("--min-frames", type=int, default=2, help="tracklets shorter than this are skipped")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(args.checkpoint, device=device)

    dataset = SoccerNetJersey(
        args.soccernet, split=args.split, train=False,
        frames_per_tracklet=args.frames_per_tracklet,
    )
    print(f"{args.split}: {len(dataset)} frames across "
          f"{len(dataset.tracklet_ids())} tracklets (cap {args.frames_per_tracklet}/tracklet)")

    len_probs, d1_probs, d2_probs = collect_frame_probs(model, dataset, device, args.batch_size)
    result = evaluate(dataset, len_probs, d1_probs, d2_probs, args.min_frames)

    print(f"\nskipped {result['skipped_short']} tracklets with < {args.min_frames} frames")
    print(f"scored {result['n_tracklets']} tracklets\n")

    print(f"{'single-frame (each frame scored alone)':<45}{result['single_frame_exact']:.4f}")
    print(f"{'majority vote (per-frame argmax, then mode)':<45}{result['majority_vote_exact']:.4f}")
    print(f"{'probability averaging (mean softmax, argmax)':<45}{result['prob_avg_exact']:.4f}")

    lift = result["prob_avg_exact"] / max(result["single_frame_exact"], 1e-9)
    print(f"\nprobability-averaging lift over single-frame: {lift:.2f}x")

    print("\naccuracy by tracklet length (more frames -> more accumulation to work with):")
    for bucket in ("2-5", "6-9", "10+"):
        if bucket in result["by_frame_count"]:
            acc, n = result["by_frame_count"][bucket]
            print(f"  {bucket:<6} frames: {acc:.4f}  (n={n} tracklets)")

    print(
        "\nreading this:\n"
        "  if prob_avg_exact is dramatically higher than single-frame, cross-frame\n"
        "    accumulation is doing its job - stop tuning stage 3, this checkpoint is\n"
        "    likely already usable, and effort is better spent on stage 4 (real footage).\n"
        "  if prob_avg_exact is ALSO stuck low, the per-frame noise floor is too high\n"
        "    for averaging to fix - that points at SoccerNet's label noise (Q2) as the\n"
        "    real bottleneck, not anything tuneable in stage 3's recipe."
    )


if __name__ == "__main__":
    main()
