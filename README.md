# basketball-v2

Near-real-time 2D basketball analysis. One video in, three artifacts out:

| Output | File | Contents |
|---|---|---|
| 1 | `annotated.mp4` | Original camera view with player / ball / rim boxes and a stable jersey number above each player |
| 2 | `topdown.mp4` | Calibrated top-down court, each player a dot with a motion trail |
| 3 | `shots.csv` + `shots.json` | Per-player made-shot records: jersey number, timestamp, court position |

Detection is Ultralytics YOLO26 with self-trained 5-class weights. Tracking uses
Ultralytics native tracking with a **configurable** backend. No Roboflow cloud
API, no SAM2, no pose estimation.

---

## Table of contents

1. [Server setup, in order](#1-server-setup-in-order)
2. [Dataset layout](#2-dataset-layout)
3. [Training and validation](#3-training-and-validation)
4. [Court calibration](#4-court-calibration)
5. [Running the pipeline](#5-running-the-pipeline)
6. [Choosing a tracker backend](#6-choosing-a-tracker-backend)
7. [Parameter calibration guide](#7-parameter-calibration-guide)
8. [Known limitations](#8-known-limitations)
9. [Project layout](#9-project-layout)

---

## 1. Server setup, in order

### Install torch first, separately

`torch` is **not** pinned in `requirements.txt` on purpose. The correct build
depends on the CUDA version of your GPU server, and installing it from plain
PyPI can silently pull a CPU-only or mismatched-CUDA wheel that appears to work
until inference is 40x slower than expected.

```bash
nvidia-smi                      # read the CUDA version first
python -m venv .venv && source .venv/bin/activate

# pick the index URL matching your server's CUDA, e.g. cu121:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Then everything else

```bash
pip install -r requirements.txt

# Jersey number recognition. Install ONE of these; paddleocr is the default pick.
pip install paddleocr          # recommended
# pip install easyocr          # alternative, digit allowlist constrains decoding

# tesseract is the LAST-RESORT fallback only - see the warning below.
# It is a SYSTEM package; the pip module is only a binding.
sudo apt-get update && sudo apt-get install -y tesseract-ocr
```

> **Do not rely on tesseract for jersey numbers.** It was built for scanned
> documents, not for 25-40 px digits on a moving shirt, and it cannot report a
> real confidence — it returns a constant 0.7, which makes the
> confidence-weighted voting in `src/jersey.py` meaningless. `jersey.backend`
> defaults to `auto`, which prefers `paddleocr`, then `easyocr`, and only falls
> back to tesseract with a loud warning. If the run summary prints
> `jersey backend  PytesseractJerseyOCR`, install one of the other two.

> **`ultralytics>=8.4.63` is required, not preferred.** The `ocsort`,
> `deepocsort`, `fasttrack` and `tracktrack` backends were added in that
> release. `tracktrack.yaml` is this project's **default** backend, so an older
> ultralytics cannot run the default config at all — it ships only `botsort`
> and `bytetrack` and raises on any other `tracker_type`.

### Verify before doing anything else

```bash
python scripts/check_env.py
```

It checks CUDA availability and GPU model, asserts the ultralytics version,
confirms all six tracker YAMLs exist inside the package, loads `yolo26n.pt`,
looks for the tesseract binary, and validates your config paths. Every failure
prints a specific fix. Missing weights and missing calibration are reported as
warnings, not failures — those are expected on a fresh checkout.

### Confirm the pipeline runs before you train anything

```bash
python scripts/smoke_test.py --input /path/to/clip.mp4
```

This uses official `yolo26n.pt` and only proves the plumbing works. Because
`yolo26n.pt` is a COCO model, `player` is really `person` and ball / rim /
jersey come back empty. That is expected.

---

## 2. Dataset layout

### Build it first

`train/basketball.yaml` describes a dataset that does not exist on a fresh
checkout. The only labelled data in the repo is `dataset/`, a Roboflow export in
a different 10-class alphabetical scheme. Convert it:

```bash
python scripts/prepare_dataset.py            # refuses if the output exists
python scripts/prepare_dataset.py --overwrite
```

It maps the 10 source classes onto these 5 **by name, never by index** (Roboflow
renumbers alphabetically on re-export, so a positional mapping would silently
relabel everything), refuses to run on an unknown source class, and prints a
verification report: box counts before and after the mapping, image/label
pairing, class-id and coordinate ranges, per-class presence in train and val,
and a field-by-field comparison against `train/basketball.yaml`.

Output:

```
basketball-v2/
  train/
    basketball.yaml                <- class scheme, source of truth
    datasets/basketball/
      data.yaml                    <- GENERATED, absolute path. Train with this
      images/{train,val,test}/
      labels/{train,val,test}/
```

`test/` is converted but deliberately **not referenced** by the generated
`data.yaml`, so it cannot be spent as a tuning set by accident.

### Which YAML to pass to training

```bash
python train/train.py train --data train/datasets/basketball/data.yaml
```

> **Not `train/basketball.yaml`.** Ultralytics resolves a relative `path:`
> against its own configured **`DATASETS_DIR`** — not the working directory, and
> not the YAML's own location (`ultralytics/data/utils.py`:
> `path = Path(... data.get("path") ...)`, then
> `if not path.is_absolute(): path = (DATASETS_DIR / path).resolve()`).
>
> `DATASETS_DIR` defaults to `~/datasets` and is frequently pointed elsewhere by
> an unrelated project — it is a global user setting, not a per-repo one. So
> `path: datasets/basketball` lands wherever that setting says, and Ultralytics
> reports the dataset as missing without explaining why.
>
> `prepare_dataset.py` therefore writes an **absolute** `path:` into the
> generated `data.yaml` (computed at runtime, never hardcoded in source) and
> prints, in section 7 of its report, exactly where each YAML resolves to and
> whether the data is there. Regenerate it on each machine.
>
> `train/basketball.yaml` keeps its relative path so it stays machine
> independent in version control. It remains the **source of truth for the class
> scheme**; the generated file is verified against it field by field.

Each `labels/*.txt` has one line per object, coordinates **normalised 0-1**,
`class_id cx cy w h`:

```
0 0.4812 0.6390 0.0640 0.2210     <- player
1 0.5121 0.3007 0.0180 0.0320     <- ball

Each `labels/*.txt` has one line per object, coordinates **normalised 0-1**,
`class_id cx cy w h`:

```
0 0.4812 0.6390 0.0640 0.2210     <- player
1 0.5121 0.3007 0.0180 0.0320     <- ball
2 0.4990 0.2455 0.0710 0.0290     <- rim
3 0.4830 0.5820 0.0300 0.0400     <- jersey (number region on the player)
4 0.1120 0.6640 0.0480 0.1900     <- referee
```

`train/basketball.yaml` is the **source of truth** for the class scheme.
`src/detect.py` mirrors it as constants, and the annotation tool checks its own
generated `data.yaml` against it field by field before exporting.

| id | name | meaning |
|---|---|---|
| 0 | `player` | whole player |
| 1 | `ball` | the basketball |
| 2 | `rim` | the hoop |
| 3 | `jersey` | the region of the shirt carrying the number |
| 4 | `referee` | detected **only so it can be excluded** — see below |

New classes are **appended, never inserted**. Indices 0-3 are baked into every
label file already annotated and into the role split in `src/track.py`, so
re-ordering would silently relabel existing data with no error anywhere.

**`referee` feeds nothing downstream, and that is the point.** It is not
tracked, never owns a jersey number, and never reaches output 3 —
`src/track.py` drops it during the role split. It exists because without a label
for "person who is not a player" the detector has nowhere to put a referee and
learns to file them under `player`, which puts a non-player on the top-down
court, into the possession contest, and potentially credited with a basket.
Detecting them is *how* they get excluded.

**The number text (0-99) is not a detection class.** Turning digits into 100
classes would need per-number training data and still would not generalise to an
unseen number. Instead the `jersey` box is cropped, read by OCR, and voted
across frames by `src/jersey.py`. The annotation tool stores the number as a
side-channel file (`jersey_numbers.json`), never as a YOLO label.

Full labelling rules for all five classes, including how to draw each box and
how much of each to collect, are in **`docs/annotation_guide.md`**. The two that
bite hardest:

- Annotate `rim` **tightly and consistently**. Every made-shot margin is measured
  in units of the rim box, so a loose rim box silently widens all of them.
- `jersey` boxes only pay off when the number is actually legible. A box drawn
  around an unreadable back does not help OCR and adds noise.

---

## 3. Training and validation

```bash
# train (defaults: yolo26s.pt, 100 epochs, imgsz 960, device 0)
python train/train.py train --data train/basketball.yaml

# larger backbone / multi-GPU
python train/train.py train --data train/basketball.yaml --model yolo26m.pt --device 0,1

# validate, with per-class mAP
python train/train.py val \
  --data train/basketball.yaml \
  --weights runs/detect/train/basketball_baseline/weights/best.pt
```

`imgsz=960` is the default for a reason: ball and rim are small objects and
degrade badly at 640. **Inference `imgsz` must match training `imgsz`**, so if
you change one, change `detector.imgsz` and `tracker.imgsz` in
`configs/default.yaml` too.

Validation prints per-class mAP rather than one aggregate number, because
`ball` and `rim` are the classes most likely to be weak and an overall mAP
dominated by `player` would hide that:

```
class            mAP50    mAP50-95
all             0.7412      0.5130
player          0.9210      0.7050
ball            0.5120      0.2980      <- watch this one
rim             0.8830      0.6120
jersey          0.6490      0.4370
referee         0.7100      0.4800
```

Classes with no validation labels are listed separately, so an untrained class
never masquerades as a score of 0.

---

## 4. Court calibration

The camera is fixed, so the homography is solved **once** outside the frame loop.

```bash
cp configs/court_calibration.template.json configs/court_calibration.json
```

Then open one frame from your video and replace every `image_points` placeholder
with the clicked pixel coordinate of that landmark. Leave `court_points` alone.
At least 4 points are needed; more points, spread widely, fit better. Delete any
landmark you cannot see from **both** `image_points` and `court_points`.

> **The canvas is portrait 500x940 on purpose.** The `court_points` span
> x<=468, y<=908, and the homography is fitted to exactly those numbers. If you
> switch to a landscape canvas you must re-click the calibration points —
> changing `canvas` / `render.court` in the config alone will silently place
> every player in the wrong spot. `config.validate()` refuses to run when
> `canvas` and `render.court` disagree.

Player court positions use the **bottom-edge midpoint** of the player box (the
feet), never the box centre. The centre would place players metres up-court
under perspective.

---

## 5. Running the pipeline

```bash
python src/pipeline.py \
  --input /path/to/game.mp4 \
  --config configs/default.yaml \
  --output-dir outputs/game1

# override the tracker backend without editing the config
python src/pipeline.py --input game.mp4 --tracker botsort.yaml

# quick check on the first 300 frames
python src/pipeline.py --input game.mp4 --max-frames 300
```

Produces `annotated.mp4`, `topdown.mp4`, `shots.csv`, `shots.json` in the output
directory, and prints a summary with the realtime ratio, how many OCR calls ran
versus were skipped by the quality gate, and any shots that could not be
attributed to a player.

---

## 6. Choosing a tracker backend

Set `tracker.tracker` in `configs/default.yaml`, or pass `--tracker`.

| Backend | Character |
|---|---|
| `botsort.yaml` | Default general-purpose tracker; motion plus optional appearance cues. The only bundled backend verified here to support ReID. |
| `bytetrack.yaml` | Lightweight strong baseline. First choice at high frame rates. |
| `ocsort.yaml` | Observation-centric SORT variant; more robust motion association. |
| `deepocsort.yaml` | OC-SORT plus ReID appearance fusion and camera motion compensation. |
| `fasttrack.yaml` | Occlusion-aware ByteTrack variant with Kalman rollback. |
| `tracktrack.yaml` | **Default.** CVPR 2025; multi-cue iterative association and Track-Aware Initialization. Best under dense occlusion, and the heaviest. |
| `configs/trackers/bytetrack_sportcv.yaml` | Custom, carried over unchanged from the previous project. Same field names as the official ByteTrack config, with `track_high_thresh` / `new_track_thresh` raised to 0.35 and `track_buffer` raised to 40. |

**Do not choose from this table alone — measure.**

```bash
python scripts/compare_trackers.py --input clip.mp4 --weights yolo26n.pt --save-videos
```

This runs every backend over the same clip through the same code path the
pipeline uses, and prints:

- **unique track ids** — the on-court player count is fixed, so a total far
  above it means the backend is churning identities. Lower is better.
- **mean / median track lifetime** — longer is more stable.
- **realtime ratio** — processing time over video duration; `<= 1.00` is real time.
- **rim contamination** — how many distinct ids the backend handed to the
  stationary rim, and what share of player id changes happen right next to it.
  This is the test for the `restrict_tracker_to_players: false` default: if
  players passing the rim are disturbing association, flip it to `true` and pay
  the second forward pass.

It works **before** you have trained anything: with `--weights yolo26n.pt` it
tracks the COCO person class, which is enough to rank backends on identity
stability. Ball and rim columns are then shown as unavailable rather than as
misleading zeros.

### ReID

`tracker.with_reid` only takes effect on backends that actually declare a
`with_reid` key. Verified: `botsort.yaml` declares it, `bytetrack.yaml` does
not. On a backend without it the run **warns loudly and proceeds without ReID**
rather than silently ignoring the flag.

Temper expectations: basketball team-mates wear near-identical kits, so
appearance embeddings may barely separate them. Treat ReID as an experiment to
run, not a fix to assume.

---

## 7. Parameter calibration guide

Everything under `# === UNCALIBRATED ===` in `configs/default.yaml` is a first
guess. None of it came from working tuned code. Calibrate items 1-5 before
trusting any shot output.

Work with a **short clip containing 5-10 known shots** you have counted by hand.
Change **one parameter at a time** and re-run:

```bash
python src/pipeline.py --input clip.mp4 --output-dir outputs/tune --max-frames 900
```

Then compare `shots.csv` against your hand count.

---

### 1. `shot.possession_max_distance_ratio` — default `0.4`

*Ball-to-player distance for possession, in units of that player's box height.*

**What to look at:** the `jersey_number` / `track_id` columns in `shots.csv`.
Is each made shot credited to the player who actually took it?

| Symptom | Direction |
|---|---|
| Shots credited to a nearby defender, or attribution flips between two players in traffic | **Decrease** toward `0.25` |
| Many rows with an empty `track_id`, i.e. `unattributed_count > 0` in `shots.json` | **Increase** toward `0.6` |

**Overshooting low:** the ball is never "held" by anyone, every shot becomes
unattributed, and output 3 is empty of names.
**Overshooting high:** every player near the ball qualifies, the nearest-player
rule becomes a coin flip, and credit lands on whoever happens to be closest to
the camera.

**Why a ratio and not pixels:** a near-baseline player box may be 300px tall
while a far-baseline one is 100px. One absolute pixel threshold cannot be
correct at both ends of the court.

---

### 2. `shot.rim_below_margin` — default `0.5`

*How far below the rim, in rim-box heights, the ball must fall to confirm a make.*

**This is the main false-positive dial.**

**What to look at:** total made count in `shots.csv` versus your hand count.

| Symptom | Direction |
|---|---|
| More makes than really happened; misses that grazed the rim are being counted | **Increase** toward `0.8-1.0` |
| Real makes missing from the output | **Decrease** toward `0.3` |

**Overshooting high:** the ball leaves frame or gets occluded by players below
the rim before reaching the confirmation line, and genuine makes vanish.
**Overshooting low:** any ball touching the rim area counts, and the shot count
inflates badly.

---

### 3. `shot.min_downward_velocity` — default `1.0` px/frame

*Minimum descent required while crossing the rim. This is what rejects a ball
travelling upward through the rim plane.*

**What to look at:** false makes whose timestamps coincide with rebounds,
put-back attempts, or passes across the rim area.

| Symptom | Direction |
|---|---|
| Upward ball motion (rebound taps, passes over the rim) counted as makes | **Increase** to `3-8` |
| Slow, high-arc swishes being missed | **Decrease** toward `0.5` |

**Note:** the default is almost certainly **too permissive**. At 30fps a real
ball descending through the rim moves far more than 1 pixel per frame. Start by
measuring: log ball y-positions across a known make and read the actual
per-frame delta.

**Overshooting high:** only fast, hard shots register; soft shooters disappear
from the stats.

---

### 4. `shot.min_possession_frames` — default `5`

*Consecutive frames a player must hold the ball to be creditable as the shooter.*

At 30fps, `5` frames is about 0.17s.

| Symptom | Direction |
|---|---|
| Attribution jitters during contested rebounds; credit goes to whoever brushed the ball | **Increase** to `8-12` |
| Quick catch-and-shoot produces no attribution | **Decrease** to `3` |

**Overshooting high:** fast plays — the ones most worth measuring — yield no
shooter at all, and `unattributed_count` climbs.
**Overshooting low:** a defender deflecting the ball for two frames can be
credited with the basket.

**Interaction:** if the tracker is switching player IDs mid-possession, a run
gets split and may fall below this threshold. Fix the tracker first
(section 6) before raising this value.

---

### 5. `shot.rim_horizontal_margin` — default `0.15`

*Horizontal tolerance for "inside the rim span", as a fraction of rim width.*

| Symptom | Direction |
|---|---|
| Balls hitting the side of the backboard or passing beside the hoop counted | **Decrease** toward `0.05` |
| Makes missed because the ball's detected centre drifts outside a tight rim box | **Increase** toward `0.3` |

**Keep this consistent with your annotation style.** If you drew rim boxes
loosely during labelling, the detected box is already wide and this margin
should be small. Re-check a few rim detections visually in `annotated.mp4`
before touching this.

---

### Jersey number parameters

`jersey.stabilizer.number_vote_window` (15), `slot_lock_min_votes` (3),
`slot_reassign_patience` (8).

These three were read as `args.*` by the salvaged source but **never declared in
its argparse block**, so the original could not reach a jersey lock and no tuned
value survives anywhere. They are guesses.

**What to look at:** the number displayed above each player in `annotated.mp4`.

| Symptom | Change |
|---|---|
| Numbers flicker between values frame to frame | Increase `number_vote_window` to 25-30, and `slot_lock_min_votes` to 5 |
| Numbers take too long to appear | Decrease `slot_lock_min_votes` to 2 |
| A number locks onto the **wrong** value and never recovers | Decrease `slot_reassign_patience` to 3-4 |
| A correct number is abandoned too easily when a player turns away | Increase `slot_reassign_patience` to 15 |

`jersey.gate.min_box_width` / `min_box_height` control which crops OCR runs on
at all. The run summary breaks skips down **by reason**
(`too_small` / `too_blurry` / `no_owner` / `bad_crop`), so it names the threshold
that starved a run rather than leaving you to guess.

> Measured on 1080p 3x3 footage: at the old defaults of 24 x 32, **308 of 320
> candidates (96%) were gated out**, the vote quorum was never reachable, and no
> number could ever lock. Jersey boxes on a wide 3x3 camera are routinely
> 20 x 25 px. The defaults are now 12 x 14.

`jersey.gate.pad_ratio` (0.18) and `upscale_to_height` (96) decide what the
recognizer actually sees. The crop is **deliberately padded** beyond the
detector's box before recognition, then upscaled.

> **A tight crop is not what the recognizer wants.** The jersey detector is
> strong at *finding* the number (mAP50 ~0.97) but weak at pixel-exact boxes
> (mAP50-95 ~0.64) — which is normal, not a defect: on a 30 x 20 box, 3 px of
> edge error alone drops IoU to ~0.64, and annotation itself is only consistent
> to within 2-3 px. Padding absorbs that jitter so the weak high-IoU
> localization stops mattering. Do not set `pad_ratio` to 0 believing a tighter
> box reads better.

Setting `min_laplacian_variance` above 0 additionally rejects motion-blurred
crops at the cost of one Laplacian per crop. Leave it at 0 until numbers are
locking: an absolute threshold on generally-soft footage rejects everything,
which looks identical to "OCR is broken".

### PARSeq — the zero-training option (start here)

```bash
python src/pipeline.py --input game.mp4 --config configs/default.yaml \
  --output-dir outputs/parseq --jersey-backend parseq \
  --max-frames 1000 --dump-jersey-crops /tmp/crops
```

Nothing to train, nothing to install beyond torch/torchvision. Weights come down
through `torch.hub` on first use (~91 MB, cached). Koshkina & Elder's
jersey-number paper reports PARSeq working on jersey numbers *without*
fine-tuning, which matched what was measured here: **23/24 on synthetic 30–70 px
blurred jersey crops**, ~50 ms per crop on CPU (expect ~5–15 ms on GPU).

Three things worth knowing:

- **It emits letters.** PARSeq decodes over an alphanumeric charset, so it reads
  the digit `0` as the letter `O` — that was *both* failures in the first probe.
  `digits_from_scene_text` repairs the near-identical glyphs (`O/o/D/Q→0`,
  `I/l/i/|→1`, `Z/z→2`, `S/s→5`, `G→6`, `B→8`) and that alone took the run from
  22/24 to 23/24. The map is deliberately tight: `b→6` was removed after it made
  the junk string `"abc"` decode as `6`.
- **`pytorch_lightning` and `nltk` are not required.** PARSeq's repo imports both
  (a LightningModule base class and one validation metric), neither needed for
  inference. `src/jersey.py` installs inference-only stubs when they are missing
  *or broken* — the check actually imports rather than calling `find_spec`,
  because a package can be installed and still raise (pytorch_lightning →
  torchmetrics → transformers → sklearn is a real ABI-breakage chain).
- **`trust_repo=True` is mandatory.** Without it `torch.hub` prompts on stdin and
  a pipeline run hangs forever.

If GitHub is unreachable from the GPU box, clone once and switch to a local
source:

```yaml
jersey:
  parseq:
    repo: /path/to/parseq
    source: local
```

### Using a Roboflow-hosted model

```bash
pip install inference-sdk
export ROBOFLOW_API_KEY=your_key_here        # never put the key in the YAML

python src/pipeline.py --input game.mp4 --config configs/default.yaml \
  --output-dir outputs/rf --jersey-backend roboflow \
  --roboflow-model digits-x7d7z/10 --max-frames 300
```

Three things about Roboflow model ids that cost real time to discover:

1. **The id is `project-slug/version`. The workspace is not part of it.**
   `workspace/project/version` is rejected by the SDK and returns HTTP 405 from
   the REST endpoint.
2. **The project must have a DEPLOYED model.** A Roboflow project of type
   `text-image-pairs` is an OCR *dataset* with nothing served behind it, and
   every call returns `500 Service misconfiguration`. The widely-linked
   `basketball-jersey-numbers-ocr/7` is exactly that type and **cannot be used
   here at all** — no client configuration fixes it. Check a project's type with
   `GET https://api.roboflow.com/{workspace}/{project}?api_key=...` before
   wiring it in.
3. **Most jersey models are per-digit detectors**, not number classifiers: "23"
   comes back as two boxes, class `2` at x=44 and class `3` at x=111, in
   arbitrary order. `src/jersey.py` sorts by x and concatenates, and takes the
   *weakest* digit's confidence as the number's confidence. Classification
   models (where `predictions` is a dict) are handled too.

> **Latency is the real constraint.** The hosted endpoint measures **2.6–6.8 s
> per crop**. At `ocr_interval: 3` with six players, a 26k-frame game issues
> tens of thousands of calls — days of wall clock. Use the hosted endpoint only
> with `--max-frames`. For anything longer, serve the model locally:
>
> ```bash
> docker run -d --gpus all -p 9001:9001 roboflow/roboflow-inference-server-gpu
> ```
> then `--roboflow-url http://localhost:9001`. Same model, same API, ~20–50 ms
> per call, no quota, and no frames leaving your machine.

`jersey.skip_locked_tracks` (default true) stops calling OCR for a track once
its number has locked. Leave it on — it is the single biggest reduction in call
volume, and further calls can only re-confirm what is already known.
`jersey.max_ocr_calls` caps the total per run as a safety net for metered
backends.

### When no number appears at all

Run with `--dump-jersey-crops /tmp/crops` and **look at the images**. Nothing
else distinguishes the two failure modes, and they have opposite fixes:

| What the crops look like | Meaning | Fix |
|---|---|---|
| Directory is empty | Everything was gated out | Lower `min_box_width` / `min_box_height` |
| Digits legible to you, filenames say `none` | The recognizer is too weak | Switch `jersey.backend` to `paddleocr` |
| Digits illegible to you too | Below the resolution floor | No model can read these — zoom the camera, or fall back to ReID-based identity |

The run summary prints a `DIAGNOSIS` line naming which of these three it is.

---

## 8. Known limitations

These are measured-unknown, not fixed. They are documented rather than
compensated for, because writing compensation code for an unquantified problem
is speculation. Once you have real false-positive statistics, revisit them.

### Made-shot detection is 2D geometry only

- **A ball passing *behind* the rim traces the same 2D path as a ball dropping
  through it.** With a single fixed side camera there is no depth cue, so this
  cannot be separated by geometry. `rim_below_margin` reduces the rate but
  cannot eliminate it. **This is the most fundamental limitation of output 3.**
- **A ball that grazes the rim and continues straight down** can register as a
  make, for the same reason: its 2D trajectory matches a swish.
- **Rim rattle.** A shot oscillating on the rim can fire early. `cooldown_frames`
  prevents double-counting the same make, but does not prevent an early trigger.
- **Missed detections cause missed makes.** If the ball detector drops frames
  during the fast descent, the below-rim stage is never observed and a real make
  is lost. **This is expected to be the dominant error source** until the `ball`
  class is well trained — likely larger than all the false-positive modes above,
  which is precisely why no compensation has been written for them yet.
- **Multiple balls in frame** (warm-up shots on an adjacent hoop) are not
  disambiguated. `src/track.py` emits one ball per frame by nearest-neighbour and
  may latch onto the wrong one.

### Shooter attribution

- **Contested rebounds and screens** make the nearest-player rule flicker.
  `min_possession_frames` suppresses short flickers, but a genuinely contested
  ball may be credited to the wrong player.
- **The rule is 2D.** A defender between the camera and the shooter can be closer
  in image space than the actual shooter.
- **A tracker ID switch mid-possession** splits the run and may drop it below the
  threshold. This fails toward *no attribution* rather than *wrong attribution*,
  which is the intended failure direction.

### Recognition and tracking

- **Jersey OCR needs a legible number.** Players facing away, motion blur and
  small far-court boxes all yield nothing. The quality gate skips those crops
  rather than feeding the vote junk, so numbers simply stay unknown.
- **ReID is unproven here.** Team-mates in identical kits may not be separable by
  appearance at all.
- **`restrict_tracker_to_players: false`** means the tracker sees ball and rim
  boxes during association. This is the fast default and has **not** been
  measured on real footage — `scripts/compare_trackers.py` reports the rim
  contamination numbers that decide whether to change it.

### Not implemented

- **Missed shots are not recorded.** Output 3 contains made shots only; the
  `made` column exists for schema stability and is always `true`.
- **No team assignment.** The salvaged colour/embedding clustering was not
  carried over.
- **No pose estimation**, by design.

---

## 9. Project layout

```
basketball-v2/
  configs/
    default.yaml                      all thresholds, paths, switches
    court_calibration.template.json   copy -> fill in -> court_calibration.json
    trackers/
      bytetrack_sportcv.yaml          tuned custom backend, carried over verbatim
  src/
    detect.py     YOLO26 wrapper -> framework-agnostic list[Detection]
    track.py      Ultralytics native tracking, configurable backend, ball NN track
    jersey.py     OCR gate -> cross-frame voting -> number locking
    calib.py      homography solve + image-to-court projection
    shot.py       made-basket state machine + shooter attribution
    render.py     output 1 and output 2 drawing
    stats.py      output 3 writer (CSV + JSON)
    config.py     YAML -> typed dataclasses
    pipeline.py   main entry point
  scripts/
    check_env.py         run this first on the server
    smoke_test.py        end-to-end check, no trained weights needed
    compare_trackers.py  backend benchmark and rim-contamination report
  train/
    basketball.yaml   5-class dataset definition (source of truth)
    train.py          train / val with per-class mAP
  tests/
    test_jersey_logic.py   voting and locking, no torch/cv2 required
    test_shot_logic.py     basket geometry and attribution, no torch/cv2 required
  docs/
    annotation_guide.md    labelling rules for all five classes - read before labelling
  tools/
    annotation/            jersey-active-learning, vendored and extended to 5 classes
```

### Building your own dataset

`tools/annotation/` is a vendored copy of the jersey-active-learning tool,
extended to export this project's 5-class scheme. It reads
`train/basketball.yaml` directly and **refuses to export** if the `data.yaml` it
generates disagrees with it, so a class-scheme change cannot silently produce a
mislabelled dataset.

```bash
cd tools/annotation/backend
pip install -r requirements.txt
python -m uvicorn app.main:app --port 8000      # upload, annotate, export
```

It splits train/val **by video or by track**, never at random — see
`docs/annotation_guide.md` for why that distinction decides whether your
validation mAP means anything. Jersey numbers are exported to a separate
`jersey_numbers.json` as OCR ground truth, normalised with the very same
`clean_jersey_number` the pipeline votes with.

Run the logic tests anywhere, including a machine with no GPU and no torch:

```bash
python tests/test_jersey_logic.py
python tests/test_shot_logic.py
```

### Design rules this codebase follows

- **No absolute paths, no drive letters, no non-ASCII in code.** Every path comes
  from config or the command line and is resolved relative to the config file, so
  the same tree works on Windows and Linux.
- **No silent fallbacks.** Missing weights raise rather than dropping back to a
  COCO model, because COCO class 0 is `person` and would masquerade as `player`
  while ball, rim and jersey silently vanished.
- **Unknown config keys are an error**, not ignored — a typo in a threshold name
  would otherwise leave the default silently in place.
- **Ultralytics is contained.** Only `src/detect.py` and `src/track.py` touch it;
  no `Results` object escapes them. `track.py` uses the API directly because
  Ultralytics fuses detection and tracking in `model.track()`; that coupling is
  deliberate and documented.
