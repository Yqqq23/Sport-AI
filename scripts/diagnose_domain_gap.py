"""Diagnose why stage 3 scores 0.99 on synthetic and 0.27 on SoccerNet.

    python scripts/diagnose_domain_gap.py

Read-only. Touches no training code, writes only PNGs under data/debug/.

Answers three questions:

  Q1  What does the loader actually hand the model for a SoccerNet sample?
      Whole-body crop squashed to 96x96, or something number-shaped?
  Q2  Where does the validation set's readable=10271 come from, and is
      anything filtering out frames where the player has turned away?
  Q3  How far apart are the two datasets in raw image statistics?

The contact sheets are the real deliverable. Numbers can be argued with; two
grids side by side, both showing exactly what the network receives, cannot.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))

import numpy as np
import torch
from PIL import Image

from augment import build_transforms
from jersey_data import SoccerNetJersey, SyntheticJersey, save_contact_sheet, torso_crop
from jersey_net import INPUT_SIZE, MEAN, STD

REPO = Path(__file__).resolve().parent.parent
SAMPLE_N = 200


def rule(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# --------------------------------------------------------------------------
# Q1 - what geometry does the model actually see
# --------------------------------------------------------------------------


def q1_geometry(sn: SoccerNetJersey, syn: SyntheticJersey) -> None:
    rule("Q1  What the loader hands the model")

    rng = random.Random(0)
    sn_rows = rng.sample(sn.rows, min(SAMPLE_N, len(sn.rows)))
    syn_rows = rng.sample(syn.rows, min(SAMPLE_N, len(syn.rows)))

    def dims(paths):
        out = []
        for p in paths:
            with Image.open(p) as im:
                out.append(im.size)  # (W, H)
        return np.array(out, dtype=np.float64)

    sn_wh = dims([r["image"] for r in sn_rows])
    syn_wh = dims([syn.root / r["image"] for r in syn_rows])

    for name, wh in (("SoccerNet", sn_wh), ("synthetic", syn_wh)):
        w, h = wh[:, 0], wh[:, 1]
        aspect = h / np.maximum(w, 1)
        print(f"\n{name} raw crops (n={len(wh)}):")
        print(f"  width   median {np.median(w):6.1f}   range {w.min():.0f}-{w.max():.0f}")
        print(f"  height  median {np.median(h):6.1f}   range {h.min():.0f}-{h.max():.0f}")
        print(f"  H/W     median {np.median(aspect):6.2f}   range {aspect.min():.2f}-{aspect.max():.2f}")

    # The loader's first transform is Resize((96, 96)) - a NON-uniform resize.
    # Anything whose aspect ratio is not 1.0 gets distorted, and the distortion
    # is exactly the median H/W above.
    sn_aspect = np.median(sn_wh[:, 1] / np.maximum(sn_wh[:, 0], 1))
    print(f"\nResize((96,96)) is NOT aspect-preserving.")
    print(f"  SoccerNet median H/W = {sn_aspect:.2f}, so a crop is squeezed")
    print(f"  horizontally by ~{sn_aspect:.2f}x. Digits arrive systematically fatter")
    print(f"  than they were shot, and fatter than any synthetic digit.")

    # A SoccerNet crop is a whole player. If the number band is ~18% of body
    # height (a torso number on a standing player), this is what survives.
    sn_h = np.median(sn_wh[:, 1])
    print(f"\nHow tall is the NUMBER inside the 96px frame?")
    print(f"  SoccerNet crop is a whole player, median height {sn_h:.0f}px.")
    for band in (0.12, 0.18, 0.25):
        print(f"    if the number spans {band:.0%} of body height "
              f"-> {96 * band:5.1f}px of the 96px input")
    print(f"  synthetic digits are rendered at 40-60% of frame height")
    print(f"    -> {96 * 0.40:.0f}-{96 * 0.60:.0f}px of the 96px input")
    print(f"\n  That is a ~{(96 * 0.50) / (96 * 0.18):.1f}x scale mismatch on the ONLY")
    print(f"  thing the model is supposed to read, before any augmentation.")


def as_model_sees(dataset, n=20, train=False):
    """Post-transform, de-normalized samples - what the tensor actually holds.

    Shared by every sheet in this file so a reviewer compares tensor contents,
    not files on disk: if a sheet looks wrong, the model saw exactly that.
    """
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    dataset.transform = build_transforms(train)
    picks = random.Random(1).sample(range(len(dataset)), n)
    out = []
    for i in picks:
        tensor, length, d1, d2 = dataset[i]
        array = ((tensor * std + mean).clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
        label = "unclear" if length == 0 else (f"{d1}" if length == 1 else f"{d1}{d2}")
        out.append((Image.fromarray(array), label))
    return out


def q1_sheets(
    sn_after: SoccerNetJersey, sn_before: SoccerNetJersey, syn: SyntheticJersey, out_dir: Path
) -> None:
    """Render every dataset variant exactly as the network receives it.

    ``sn_before``/``sn_after`` share the same tracklet sampling (same seed,
    same frames_per_tracklet), so index i is the SAME source frame in both -
    an apples-to-apples before/after for the torso-crop fix, not two random
    draws that happen to differ.
    """
    for name, dataset, train in (
        ("soccernet_BEFORE_wholebody_val", sn_before, False),
        ("soccernet_AFTER_torsocrop_val", sn_after, False),
        ("soccernet_AFTER_torsocrop_train_aug", sn_after, True),
        ("synthetic_val", syn, False),
        ("synthetic_train_aug", syn, True),
    ):
        path = out_dir / f"domain_gap_{name}.png"
        save_contact_sheet(as_model_sees(dataset, train=train), path)
        print(f"  wrote {path}")


# --------------------------------------------------------------------------
# Q2 - where readable=10271 comes from, and what is filtered
# --------------------------------------------------------------------------


def q2_sampling(root: Path, frames_per_tracklet: int) -> None:
    rule("Q2  Frame sampling and the readable=10271 figure")

    print("\nSTRUCTURAL ANSWER, straight from jersey_data.SoccerNetJersey.__init__:")
    print("  frames = sorted(all image files in the tracklet folder)")
    print("  if len(frames) > frames_per_tracklet: frames = rng.sample(frames, N)")
    print("  -> a UNIFORM RANDOM sample. There is NO filtering of any kind:")
    print("     not for player orientation, not for blur, not for occlusion,")
    print("     not for whether the number is visible in that specific frame.")
    print("  Every sampled frame inherits the TRACKLET's label.")

    for split in ("train", "test"):
        try:
            ds = SoccerNetJersey(root, split=split, train=False,
                                 frames_per_tracklet=frames_per_tracklet)
        except FileNotFoundError as error:
            print(f"\n{split}: {error}")
            continue

        per_tracklet = Counter(row["tracklet"] for row in ds.rows)
        labels = {}
        for row in ds.rows:
            labels[row["tracklet"]] = row["number"]

        readable_tracklets = [t for t, n in labels.items() if n is not None]
        unclear_tracklets = [t for t, n in labels.items() if n is None]
        readable_frames = sum(per_tracklet[t] for t in readable_tracklets)
        unclear_frames = sum(per_tracklet[t] for t in unclear_tracklets)

        print(f"\n{split} split:")
        print(f"  tracklets           {len(labels)}")
        print(f"    labelled 1-99     {len(readable_tracklets)}")
        print(f"    labelled -1       {len(unclear_tracklets)}")
        print(f"  frames emitted      {len(ds.rows)}")
        print(f"    from readable     {readable_frames}")
        print(f"    from unclear      {unclear_frames}")

        counts = Counter(per_tracklet.values())
        print(f"  frames per tracklet: " +
              ", ".join(f"{k}x{v}" for k, v in sorted(counts.items())))
        capped = sum(v for k, v in counts.items() if k == frames_per_tracklet)
        print(f"    {capped} tracklets hit the cap of {frames_per_tracklet}")

        if split == "test":
            print(f"\n  => readable={readable_frames} is {len(readable_tracklets)} "
                  f"tracklets x up to {frames_per_tracklet} frames.")
            print(f"     Every one of those frames is labelled with its tracklet's")
            print(f"     number REGARDLESS of whether the number is visible in it.")
            print(f"     A player who turns away for half a tracklet contributes")
            print(f"     ~half its frames as confidently mislabelled training data.")

        length_dist = Counter()
        for t, n in labels.items():
            length_dist[0 if n is None else len(str(n))] += per_tracklet[t]
        print(f"  length target distribution: "
              f"unclear={length_dist[0]} one-digit={length_dist[1]} "
              f"two-digit={length_dist[2]}")


# --------------------------------------------------------------------------
# Q3 - image statistics
# --------------------------------------------------------------------------


def _stats(image: Image.Image) -> dict:
    """Brightness, gradient energy and high-frequency share for one image.

    Computed on the 96x96 the model receives, in grayscale, so the two
    datasets are compared on identical footing.
    """
    gray = np.asarray(
        image.convert("L").resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR),
        dtype=np.float32,
    ) / 255.0

    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    gradient = float(np.sqrt((gx ** 2).mean() + (gy ** 2).mean()))

    # Share of spectral energy outside the central quarter of the FFT - a
    # blur-insensitive proxy for how much fine detail survives.
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(gray - gray.mean())))
    total = spectrum.sum() + 1e-9
    c = INPUT_SIZE // 2
    r = INPUT_SIZE // 8
    low = spectrum[c - r:c + r, c - r:c + r].sum()

    return {
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "gradient": gradient,
        "high_freq": float(1.0 - low / total),
    }


def q3_statistics(sn: SoccerNetJersey, syn: SyntheticJersey) -> None:
    """Brightness/contrast/gradient/high-freq, whole-body vs torso-cropped vs synthetic.

    Computed straight from disk, torso_crop applied in-process for the "after"
    column - this measures the geometry fix in isolation from any change to
    the synthetic side, so a shrinking gap here is attributable to the crop.
    """
    rule("Q3  Image statistics, 200 samples each")

    rng = random.Random(2)
    sn_rows = rng.sample(sn.rows, min(SAMPLE_N, len(sn.rows)))
    syn_rows = rng.sample(syn.rows, min(SAMPLE_N, len(syn.rows)))

    collected = {"SoccerNet (whole body)": [], "SoccerNet (torso crop)": [], "synthetic": []}
    for row in sn_rows:
        with Image.open(row["image"]) as im:
            whole = im.convert("RGB")
            collected["SoccerNet (whole body)"].append(_stats(whole))
            collected["SoccerNet (torso crop)"].append(_stats(torso_crop(whole)))
    for row in syn_rows:
        with Image.open(syn.root / row["image"]) as im:
            collected["synthetic"].append(_stats(im.convert("RGB")))

    keys = ("brightness", "contrast", "gradient", "high_freq")
    cols = list(collected.keys())
    print(f"\n{'metric':<12}" + "".join(f"{c:>26}" for c in cols))
    print(f"{'':12}" + "".join(f"{'mean +/- sd':>26}" for c in cols))
    print("-" * (12 + 26 * len(cols)))

    values = {c: {k: np.array([s[k] for s in collected[c]]) for k in keys} for c in cols}
    for key in keys:
        row_text = f"{key:<12}"
        for c in cols:
            a = values[c][key]
            row_text += f"{a.mean():14.4f} +/-{a.std():7.4f}"
        print(row_text)

    print("\nsynthetic/SoccerNet ratio, whole-body crop vs torso crop:")
    for key in keys:
        before = values["SoccerNet (whole body)"][key].mean()
        after = values["SoccerNet (torso crop)"][key].mean()
        syn_mean = values["synthetic"][key].mean()
        ratio_before = syn_mean / before if before else float("nan")
        ratio_after = syn_mean / after if after else float("nan")
        moved = "narrowed" if abs(ratio_after - 1) < abs(ratio_before - 1) else "widened"
        print(f"  {key:<12} before={ratio_before:6.2f}  after={ratio_after:6.2f}  ({moved} toward 1.0)")

    print("\nreading this table:")
    print("  a large gap in ANY row is a cue the network can use to tell the two")
    print("    sources apart, and a network that can tell them apart is free to")
    print("    learn two unrelated functions - which is what 0.99 vs 0.27 was.")
    print("  the torso crop removes grass/limbs and enlarges the number, so")
    print("    contrast and brightness are expected to move toward synthetic;")
    print("    it does NOT change either dataset's degradation pipeline, so it")
    print("    is not expected to fully close the gap by itself.")


# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", type=Path, default=REPO / "data" / "synthetic")
    parser.add_argument("--soccernet", type=Path, default=REPO / "data" / "soccernet")
    parser.add_argument("--out", type=Path, default=REPO / "data" / "debug")
    parser.add_argument("--frames-per-tracklet", type=int, default=12)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # Same seed and frames_per_tracklet on both, so sn_before.rows[i] and
    # sn_after.rows[i] are the SAME source frame - crop_torso is the only
    # variable, which is what makes the before/after sheets a fair comparison.
    sn_after = SoccerNetJersey(args.soccernet, split="test", train=False,
                               frames_per_tracklet=args.frames_per_tracklet,
                               crop_torso=True)
    sn_before = SoccerNetJersey(args.soccernet, split="test", train=False,
                                frames_per_tracklet=args.frames_per_tracklet,
                                crop_torso=False)
    syn = SyntheticJersey(args.synthetic, train=False)

    q1_geometry(sn_before, syn)
    rule("Q1  Contact sheets - what the network actually receives")
    q1_sheets(sn_after, sn_before, syn, args.out)
    q2_sampling(args.soccernet, args.frames_per_tracklet)
    q3_statistics(sn_after, syn)

    rule("LOOK AT THESE THREE FILES")
    print(f"  {args.out / 'domain_gap_soccernet_BEFORE_wholebody_val.png'}  (old behaviour)")
    print(f"  {args.out / 'domain_gap_soccernet_AFTER_torsocrop_val.png'}   (torso-crop fix)")
    print(f"  {args.out / 'domain_gap_synthetic_val.png'}")
    print("\nCompare AFTER to synthetic, not BEFORE - BEFORE is there to show what")
    print("changed. If AFTER and synthetic still do not look like the same task,")
    print("and no amount of mixing ratio or learning rate will bridge them.")


if __name__ == "__main__":
    main()
