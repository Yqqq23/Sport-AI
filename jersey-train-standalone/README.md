# jersey-train-standalone

Trains a per-digit jersey-number recogniser. Self-contained: it needs a GPU, a
Python environment, and an internet connection for three downloads. It does not
need video, a detector checkpoint, annotation data, or any part of
`basketball-v2`.

It produces one file pair — `checkpoints/stage3_soccernet.pt` and
`stage3_soccernet.json` — which is the entire output. Copy those two files to
wherever stage 4 runs.

## What this model does, and why it is not a classifier

Jersey numbers are an **open set**. FIBA 3x3 allows `0`, `00` and `1`–`99` —
101 legal numbers — with four players per team and no duplicates within a team,
so any one match shows eight of them, and a different eight next match.

A closed-set classifier cannot work here. Trained on one match's eight numbers
it cannot name a ninth; given a 101-way head it leaves 93 classes with zero
examples, and an untrained logit still wins an argmax sometimes — producing
confident predictions for numbers the model has never seen.

So the model recognises **digits**, three heads on a shared ResNet18 trunk:

| head | outputs | meaning |
|---|---|---|
| `length` | 3 | `0` = unreadable, `1` = one digit, `2` = two digits |
| `digit1` | 10 | first digit, 0–9 |
| `digit2` | 10 | second digit, 0–9; undefined when `length == 1` |

An image labelled `23` trains both "this is a 2" and "this is a 3", so eight
numbers still cover most of the ten digit shapes. `0` and `00` stay distinct —
both are legal, a team may field both, and the length head is what separates
them.

The length head's class index is literally the digit count, with `0` — an
impossible count — carrying the unreadable case. Whole numbers are reassembled
downstream, where per-digit probabilities accumulate across frames. **Single-
frame accuracy is not the product**; it only has to be good enough to feed that
accumulator.

## Requirements

- Python 3.9+
- A CUDA GPU (it will run on CPU, unusably slowly)
- ~8 GiB of disk — see [Disk budget](#disk-budget)
- At least one bold sans-serif TrueType font. `fonts-dejavu` is enough;
  `scripts/gen_synthetic.py --list-fonts` shows what was found.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run it, from zero

Every stage detaches through `scripts/run_jersey_training.sh`, so an SSH drop
cannot kill a 30-epoch run. The commands below are the underlying calls; use
either.

### 0. Confirm the code works — 10 seconds, no data needed

```bash
python train/smoke_jersey_net.py
```

Checks label expansion, 96×96 output, the absence of flip augmentation, and
that a batch mixing two-digit / one-digit / unreadable samples produces a
finite loss. If this fails, nothing below is worth starting.

### 1. SVHN pretraining

```bash
bash scripts/run_jersey_training.sh stage1
tail -f logs/stage1_svhn.log
```

Downloads SVHN (~235 MiB) and trains **the `digit1` head and the backbone
only**, 20 epochs, AdamW 3e-4 cosine, batch 128.

SVHN is ~73k photographed house-number digits — genuinely blurred, skewed,
badly lit. That is the right prior: digit identity under real optical
degradation, learned from real photons rather than from a font renderer. It has
no notion of a two-digit number, no `00`, and no unreadable case, so the
`length` and `digit2` heads are left at their random init and first supervised
in stage 3.

**Produces:** `checkpoints/stage1_svhn.pt` + `.json`,
`data/debug/svhn_sample.png`.

> Look at `svhn_sample.png`. A half-downloaded `.mat` still loads and still
> yields correctly-shaped tensors; the loop would consume garbage for 20 epochs
> without complaining. Use `--sample-only` to write the sheet and stop.

### 2. Synthetic data — the review gate

```bash
bash scripts/run_jersey_training.sh synthetic     # foreground, on purpose
```

Renders 101 numbers × 500 = 50,500 crops, then destroys them along the real
capture path: occlusion → perspective → motion blur → **downscale to 20–40 px
and back to 96** → gaussian → JPEG.

This is the only source that can cover the full legal range: to see number 87
in real footage, a player wearing 87 has to have been filmed. Coverage is the
whole reason it exists — it cannot supply realism, which is what stages 1 and 3
are for.

The downscale is the load-bearing step. Without it the model learns fine stroke
detail that does not survive to a 20–40 px jersey number, giving excellent
validation numbers and a collapse on real input.

**Produces:** `data/synthetic/` (~102 MiB) with `manifest.json`, and
`data/debug/synthetic_sample.png`.

> **Stop here and look at the contact sheet.** Too sharp and the model learns
> detail that is not there in deployment; unreadable and those samples teach
> noise. You should be able to read most numbers with effort and fail on a few.

### 3. SoccerNet fine-tuning

```bash
python scripts/download_soccernet.py              # train + test only
python train/finetune_soccernet.py --inspect-only # verify before committing 30 epochs
bash scripts/run_jersey_training.sh stage3
tail -f logs/stage3_soccernet.log
```

Loads `stage1_svhn.pt`, mixes synthetic with SoccerNet **1:1**, and trains all
three heads — 30 epochs, AdamW 3e-4 cosine, batch 64, backbone frozen for the
first 2.

The freeze matters because two of the three heads enter this stage from random
init. Their first gradients are large and meaningless, and letting those reach a
backbone that already carries SVHN's digit features undoes the stage it was
pretrained for.

Tracklets labelled `-1` become unreadable samples and train length class 0.
They are the only real unreadable examples anywhere in this pipeline, and they
are what lets the model say "unclear" instead of guessing — a confident wrong
guess poisons the cross-frame accumulation downstream; a confident "unclear"
costs nothing.

**Selection runs on SoccerNet's held-out test split alone.** Validating on the
mix would let a rising synthetic score — easy, and measured against the same
renderer that produced the training data — mask a falling real one. Synthetic is
still scored every epoch and logged as a diagnostic; it just gets no vote.

**Produces:** `checkpoints/stage3_soccernet.pt` + `.json`. **This is the
deliverable.**

#### If SoccerNet asks for a password

It is behind an NDA form. `download_soccernet.py` exits with instructions
rather than prompting — a script blocked on stdin under `nohup` is a hang with
no error message. Fill in the form linked from <https://www.soccer-net.org/data>,
then re-run with `--password '<password>'`.

Stages 1 and 2 do not depend on this data. If the form is pending, run them
now and come back; stage 3 picks up from the same `stage1_svhn.pt`.

The `challenge` split is deliberately refused: its 1211 tracklets have hidden
labels, so it costs bandwidth and teaches nothing.

## Reading the output

Every stage ends with a report: a 3×3 length confusion matrix, two 10×10 digit
matrices, and explicit two-way leakage on the pairs that actually cost accuracy
here — `6↔9`, `0↔6↔8`, `1↔7`, `3↔8`, `5↔6`.

**Checkpoint selection is mean per-digit macro-F1.** Whole-number exact match is
printed but cannot select: the number distribution is long-tailed, so a model
that learns the common numbers and abandons the rest still scores well while
being useless on exactly the videos where a rare number appears. Length F1 is
excluded from selection too — it is the easiest head, and including it would let
a model coast on it while both digit heads degrade.

`6↔9` is worth watching specifically. It is the pair a vertical flip creates,
and it is why **no flip augmentation exists anywhere in this codebase** — a
vertical flip maps 6 onto 9 with the label untouched, training the model to
confuse the one pair it can least afford. `RandomAffine`'s `scale` stays
strictly positive for the same reason, so it cannot introduce a mirror.

## Handing off to `basketball-v2`

```
this package  ──  stage3_soccernet.pt + .json  ──▶  basketball-v2
                  (~43 MiB, copy both)              stage 4 domain adaptation
                                                    → LocalJerseyClassifier
```

Copy **both** files. The `.json` sidecar carries input size, normalization
constants, head semantics and the decode rule, so the consuming side reads the
contract instead of keeping a second hardcoded copy of it — two copies of `96`
and the ImageNet constants is exactly how a preprocessing mismatch ships
unnoticed, since the model still returns confident logits, they are just wrong.

**The receiving side does not need any code from this package.** The checkpoint
is a plain `state_dict`; `basketball-v2` has its own copy of the model
definition at `train/jersey_net.py`. To load it there:

```python
from jersey_net import load_checkpoint          # basketball-v2's copy
model = load_checkpoint("stage3_soccernet.pt", device="cuda")
```

`load_checkpoint` rebuilds the architecture with `pretrained=False`, so it
needs no network and no ImageNet cache.

If you ever need to load it without *any* shared code, the sidecar has enough
to rebuild it by hand: torchvision `resnet18` with the final `fc` removed, plus
`Linear(512, 3)`, `Linear(512, 10)`, `Linear(512, 10)` named `head_len`,
`head_digit1`, `head_digit2`.

Stage 4 domain adaptation and the inference side stay in `basketball-v2`. They
need the annotation data and the video, which is why they live on the account
that has them.

## Disk budget

Measured where measurable; estimates are marked as such.

| Item | Size | Basis |
|---|---:|---|
| This code | 0.1 MiB | measured, 11 files |
| Python env (`torch` CUDA + `torchvision`) | **~3–4 GiB** | CPU-only torch measures 1.13 GiB here; CUDA wheels bundle ~2–3 GiB of `nvidia_*` libs. **Estimate** |
| ImageNet ResNet18 weights | 44.7 MiB | measured (`46,830,571` B), downloaded once |
| SVHN `train_32x32.mat` | 174 MiB | published size (182 MB) |
| SVHN `test_32x32.mat` | 61 MiB | published size (64.3 MB) |
| Synthetic 50,500 × JPEG q92 | 102 MiB logical / **197 MiB on disk** | measured: mean 2,110 B/img over 120 samples × 50,500. On-disk figure assumes 4 KiB blocks — see below |
| SoccerNet jersey-2023 archive | ~2 GiB | **Estimate**, see below |
| SoccerNet extracted | ~2 GiB | **Estimate** |
| Checkpoints (stage1 + stage3) | 86 MiB | measured, 42.76 MiB each; only the best epoch is kept per stage, not one per epoch |
| Logs + contact sheets | <10 MiB | ~50 epochs of per-step lines, plus two PNGs |

**Peak** — the moment the SoccerNet archive and its extraction coexist:

```
  env 3.5 GiB + SoccerNet 4.0 GiB + SVHN 235 MiB + synthetic 197 MiB
              + weights 45 MiB + checkpoints 86 MiB + logs 10 MiB
  ≈ 8.1 GiB
```

**Ask for 10 GiB.** The uncertainty is almost entirely SoccerNet and the CUDA
wheels; everything else is measured and small.

If the quota is hard-capped lower, `--per-number 250` halves the synthetic set
(a real cost to rare-number coverage), and `--frames-per-tracklet 6` halves what
SoccerNet contributes per epoch without changing what is on disk.

### What to delete, and when

| Delete | When | Frees |
|---|---|---|
| SoccerNet archive | after extraction succeeds | ~2 GiB |
| `data/svhn/*.mat` | after stage 1 finishes | 235 MiB |
| `data/synthetic/` | after stage 3 finishes | 197 MiB |
| `checkpoints/stage1_svhn.pt` | after stage 3 finishes | 43 MiB |

**Steady state while training** (archive deleted, everything else live):
**~6.1 GiB**.

**After stage 3, keeping only the deliverable:** the env (~3.5 GiB) plus 43 MiB
of checkpoint. Everything else is regenerable — synthetic from the seeded
generator, SVHN and SoccerNet by re-download.

Nothing in the pipeline reads a dataset after its stage completes, so those
deletions are safe mid-run. The one thing to keep until you are sure you are
done is `stage1_svhn.pt`: stage 3 restarts from it, and regenerating it costs
20 epochs.

### On the synthetic storage format

Currently 50,500 individual JPEGs at quality 92. Measured alternatives, same
images:

| Format | 50,500 images | Note |
|---|---:|---|
| **JPEG q92 (current)** | **102 MiB** | |
| JPEG q85 | 89 MiB | 13% smaller, more artifacts |
| PNG | 483 MiB | 4.7× larger for no benefit |
| raw `uint8` `.npy` | 1.3 GiB | worst option |

JPEG is clearly right. The degradation pipeline already applies a JPEG pass at
quality 22–80, so the storage encode at 92 is strictly gentler than what the
image has already been through — it introduces no artifact class the data does
not already contain.

The real cost is not bytes, it is **files**. At a mean of 2,110 B against a
4 KiB filesystem block, every image wastes ~1.9 KiB: 102 MiB of data occupies
~197 MiB of disk, and burns 50,500 inodes. If the quota is tight or inodes are
capped, packing the same JPEG bytes into one blob plus an offset index would
recover ~95 MiB and 50,499 inodes at no quality cost — the images stay
byte-identical, only the container changes.

Not implemented; the current loader reads loose files through `manifest.json`.
