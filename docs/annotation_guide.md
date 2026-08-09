# Annotation guide — basketball-v2, 5 classes

This is the rulebook for labelling your own footage. It exists because most of
the ways annotation goes wrong here are **silent**: the dataset trains, the mAP
looks plausible, and the damage only shows up as bad shot counts weeks later.

Source of truth for the classes is `train/basketball.yaml`. The annotation tool
(`tools/annotation/`) reads it directly and refuses to export if its generated
`data.yaml` disagrees.

| id | name | what the box contains | feeds |
|---|---|---|---|
| 0 | `player` | whole player, head to feet | tracking, top-down, possession, shot attribution |
| 1 | `ball` | the basketball | made-shot geometry, possession |
| 2 | `rim` | the hoop | **every** made-shot threshold |
| 3 | `jersey` | the number patch on the shirt | OCR crop |
| 4 | `referee` | whole referee | **nothing — exclusion only** |

The **number text** (0–99) is not a class. It is typed into a side field and
exported to `jersey_numbers.json` as OCR ground truth. It never appears in a
`.txt` label file.

---

## Your footage: fixed wide camera

Everything below assumes what you actually have — a fixed wide shot, not a
broadcast feed. Two consequences that override normal instincts:

**Players are much smaller than in broadcast footage.** A far-baseline player
may be 40–60 px tall. That is *not* too small to label. Public basketball
datasets are broadcast-framed, so a model trained on them collapses on your
far end. Those small far-court boxes are the most valuable labels you will
draw, because they are the ones nothing else in the world will teach the model.

**Do not skip the far end.** The temptation is to label the near half where
everything is easy and legible. A dataset labelled that way teaches the
detector that small distant objects are background, and the far basket becomes
invisible. If a person is recognisably a person, box them.

Rule of thumb: label down to roughly **16 px** of box height for people and
**8 px** for the ball. Below that, skip — and skip *consistently*.

> **Consistency beats correctness at the margin.** A borderline object you
> sometimes box and sometimes do not is worse than one you always skip: you are
> teaching the detector that identical pixels mean two different things. Pick
> your cutoff, write it down, and hold it.

---

## Class 2 — `rim`

**Priority: highest. Get this right before anything else.**

**Rule: draw it tight, and draw it the same way every single time.**

Box the **orange hoop ring only**. Not the net, not the backboard, not the
support arm. If the rim is a foreshortened ellipse, box that ellipse.

### Why this one matters more than its size suggests

Every made-shot threshold in `src/shot.py` is measured in **units of the rim
box**, not in pixels:

| config value | measured in |
|---|---|
| `rim_above_margin` (0.5) | rim box **heights** |
| `rim_below_margin` (0.5) | rim box **heights** |
| `rim_horizontal_margin` (0.15) | rim box **widths** |

So if you habitually draw the rim box 40% too large — by including the net, say
— then every one of those margins silently becomes 40% larger. The confirmation
line drops further below the hoop, the horizontal tolerance widens, and balls
passing *near* the rim start registering as makes. **Nothing reports this.**
`shots.csv` just quietly gains false baskets, and you will spend a day tuning
`rim_below_margin` to compensate for a labelling habit.

Including the net is the single most damaging mistake in this guide, because
the net roughly doubles the box height and it is the most natural thing to do.

### Specifics

- **Both rims.** Label the far rim too, even though it is tiny.
- **Occluded by a player:** box the full rim where you know it to be. It is a
  static object; you can see it in neighbouring frames.
- **Net moving after a make:** ignore the net entirely, every time.
- **Backboard:** never included.
- If the rim is genuinely not visible at all, do not invent it.

**How much:** the camera is fixed, so the rim barely moves. You need far fewer
rim boxes than you would think — **150–300 per rim position** is plenty. What
you need is *consistency*, not volume. Label rims in one sitting so your hand
stays calibrated.

---

## Class 1 — `ball`

**Priority: highest. This is the hardest class and the one that limits everything.**

**Rule: box the ball in every state it appears in, including the ugly ones.**

### Why this is the critical class

From `README.md` §8: *missed ball detections during the fast descent are
expected to be the dominant error source* — larger than every false-positive
mode combined. The shot state machine needs to see the ball **above**, then
**inside**, then **below** the rim. Miss the descent and a real make is simply
never counted.

And the descent is exactly when the ball is hardest to detect: small, fast,
motion-blurred, often against a busy crowd background.

### The states you must cover

Label all of these. A dataset that only contains crisp, stationary balls
produces a detector that drops the ball precisely when it matters.

| state | why it must be in the set |
|---|---|
| **Held still** | easy baseline; do not let it dominate |
| **Being dribbled** | partly occluded by hand, close to the floor, often motion-blurred |
| **In flight — pass** | fast horizontal blur, elongated shape |
| **In flight — shot** | the arc. **The single most important state** |
| **Passing through the rim** | the frames the make decision depends on |
| **Partly occluded by hands/body** | box the *whole* ball including the hidden part |
| **Motion-blurred to an ellipse** | box the blur extent; do not skip it |
| **Against the crowd** | hardest background; needed or the far end fails |

### Rules

- Box the ball's **full circular extent**, including the parts behind a hand.
  A box around only the visible sliver teaches the wrong object size, and
  `src/shot.py` uses the ball **centre** — a clipped box moves that centre.
- **Blurred ball:** box the full smear. Do not try to guess the "real" position
  inside it.
- **Two balls in frame** (warm-up on an adjacent hoop): label both as `ball`.
  The pipeline picks one by nearest-neighbour; that is a known limitation
  (`README.md` §8), not something labelling can fix. Do not leave one unlabelled
  — an unlabelled visible ball is a false negative you are training in.
- **Ball completely hidden:** do not label. Never guess a position.

**How much:** **1500–3000 boxes minimum**, and deliberately weighted toward
flight and rim-passage frames. Count them: if fewer than ~300 of your ball boxes
are of a ball in the air near the rim, the state machine has not been taught the
case it exists to handle. This is where your annotation hours should go.

---

## Class 0 `player` vs class 4 `referee`

**Rule: box every person on the court, then decide which of the two they are.**

Both are whole-body boxes: top of head to feet, including raised arms. Same
drawing standard. The only question is the label.

### Telling them apart

| | `player` (0) | `referee` (4) |
|---|---|---|
| shirt | team kit, sleeveless, numbered | striped or plain grey/black referee shirt |
| shorts | team colours, longer | plain dark, shorter |
| motion | plays the ball | stays at the periphery, watching |
| position | in the action | baseline, sideline, trail |

In your wide fixed shot, **kit colour and shirt pattern read at far smaller
sizes than facial detail does**. Judge by the shirt, not the face.

### Why `referee` exists at all

It feeds **nothing**. `src/track.py` drops it during the role split; it is never
tracked, never owns a jersey number, never reaches `shots.csv`.

That is the entire point. Without a label for "person who is not a player", the
detector has nowhere to put a referee and learns to file them under `player`.
A referee then appears on the top-down court, joins the possession contest, and
can be credited with a basket. **Detecting them is how they get excluded.**

So: labelling referees is not optional busywork. Every referee you leave
unlabelled is a person the detector will guess about, and its best guess is
`player`.

### Edge cases — decide once, apply always

| situation | label |
|---|---|
| Coach standing on the sideline | **`referee`** — see note below |
| Substitutes on the bench | **`referee`** — see note below |
| Substitute kneeling at the scorer's table waiting to enter | **`referee`** |
| Player who just subbed out, still on court | `player` while on court |
| Scorekeeper / table officials | `referee` |
| Spectators in the stands | **do not label at all** |
| Photographer under the basket | `referee` |
| Person half out of frame at the edge | label if recognisably a person; box only the visible part |

> **Why coaches and bench players go under `referee`, not a new class.**
> Class 4 is really "person present but not playing". Its job is to give the
> detector somewhere to put every non-player human so they do not contaminate
> class 0 — and for that job, a coach and a referee are the same thing. Adding
> `coach` and `bench` classes would split scarce training data across three
> rare categories to make a distinction nothing downstream reads. If you later
> want them separated, they can be **appended** as classes 5 and 6 without
> touching indices 0–4.
>
> If that naming bothers you, read the class as "non-player person". The name
> stays `referee` because the index is already baked into labelled data.

**Spectators are the exception** and the reason is practical: there are
hundreds of them, they are static, they are behind the court, and they never
touch the ball. Labelling them would consume most of your annotation budget to
teach a distinction the detector can make from position alone. Leave them
unlabelled — but be aware this means the model sees crowd bodies as background,
so *do* label anyone who steps onto the court itself.

### Player-specific rules

- **Overlapping players:** box each one fully, including the occluded parts you
  can infer. Boxes overlapping heavily is normal and fine.
- **Feet matter more than heads.** Court position uses the **bottom-edge
  midpoint** of the player box (`README.md` §4). A box cut off at the ankles
  places that player metres up-court in `topdown.mp4`. When you have to be
  sloppy somewhere, be sloppy at the top.
- **Player jumping:** the box bottom is their feet in the air, not the floor.
- **Player partly out of frame:** box the visible part only.

**How much:** `player` is the easiest class and will fill up on its own —
**2000+ boxes** arrives quickly. `referee` is much rarer; aim for **300–500**
and deliberately seek out frames containing them, or the class will be too
sparse to learn and referees will keep leaking into `player`.

---

## Class 3 — `jersey`

**Rule: only box it when you can actually read the number yourself.**

The box is the **number patch**, not the whole shirt: a tight rectangle around
the digits, with a small margin. Back numbers and front numbers both count.

### The one rule that matters

**If you cannot read the number, do not draw the box.**

Not "box it and mark it unclear". Do not draw it.

The `jersey` class exists for exactly one purpose: producing a tight crop for
OCR. A box drawn around an unreadable back:

- teaches the detector that featureless fabric is a jersey, and
- at inference produces a confident detection feeding garbage into the vote
  pool in `src/jersey.py`, where it competes with real readings.

An unreadable number costs you nothing when unlabelled — the player still gets
tracked, and `shots.csv` falls back to the track id. It costs you a corrupted
number when labelled. **The asymmetry is the whole argument.**

So skip: players facing away with no visible number, motion-blurred torsos,
far-court players whose digits are a smudge, numbers folded into a crease,
numbers under an arm.

### Typing the number

The number goes in the tool's number field, **not** into the class. It is
normalized by `src/jersey.py`'s `clean_jersey_number` — the same function the
inference pipeline votes with:

- digits only
- at most 2 digits
- leading zeros stripped: **`01` and `1` are the same number**
- bare `0` is legal; `00` is rejected

Type what you see; the tool normalizes. If you type `01` it is stored as `1`.

### Track ids

For `player` and `jersey` boxes, set a **track id** — the same integer for the
same person across frames. It does not need to be their jersey number, just
consistent. Ball, rim and referee boxes do not need one.

This is what lets the dataset splitter keep one player's frames on one side of
the train/val boundary. Splitting **by video** works without track ids; only
`split_strategy=track` needs them. The tool warns if you ask for a track split
without them.

**How much:** **800–1500 legible jersey boxes**, spread across as many different
numbers and players as your footage contains. Ten numbers × 100 examples beats
2 numbers × 500 — OCR must generalise to digits it has not seen in that kit.

---

## Priority order

If you have limited hours, spend them in this order:

| # | class | target | rationale |
|---|---|---|---|
| 1 | `ball` | 1500–3000 | dominant error source; nothing else works without it |
| 2 | `rim` | 150–300 per rim | cheap, and every shot threshold is scaled by it |
| 3 | `player` | 2000+ | accumulates for free while labelling other classes |
| 4 | `jersey` | 800–1500 legible | needed for named output, not for shot counting |
| 5 | `referee` | 300–500 | rare but prevents non-players contaminating class 0 |

`ball` and `rim` before `jersey`: a pipeline that counts baskets correctly but
labels them `id:7` is useful. One that knows everyone's name but miscounts
baskets is not.

**Label whole frames, not whole classes.** For each frame you open, draw every
object of every class present, then move on. Passing through the footage five
times labelling one class each pass produces frames with missing labels — and in
YOLO, an unlabelled visible object is trained as **background**, actively
teaching the detector to miss it.

---

## Train/val split: by video, never at random

**Rule: `split_strategy=video`. Never a random split.**

The tool defaults to this and the exporter verifies it — a video appearing in
two splits is a hard error, not a warning.

### Why random splitting produces a meaningless number

Frames are sampled from continuous video. Frame 300 and frame 301 are nearly the
same image: same players, same positions, same lighting, same jersey creases.

Under a random split, frame 300 lands in train and frame 301 in val. The model
is then validated on images it has effectively memorised. Validation mAP comes
back high — 0.9+ is common — and it means **nothing**. It measures the model's
ability to recall frames it has seen, not to generalise to a new game.

You would discover this only when the model met genuinely new footage and
collapsed, with no idea why, because every number you had said it was fine.

Splitting by video keeps all frames of a clip on one side. Val then contains
players, lighting and camera framing the model has never seen — which is the
question you actually want answered: *will this work on next week's game?*

### Practical consequences

- **You need several distinct videos.** One video split by video cannot produce
  a val set at all. Collect **at least 4–5 clips**, ideally different sessions,
  teams and lighting.
- **Expect the honest number to be lower.** A by-video mAP of 0.6 is worth more
  than a random-split 0.95. The first is information; the second is an echo.
- **Same-game clips are not independent.** Two clips from one game share
  players, kit and lighting. Better than random, still optimistic. If you can,
  hold out an entire *session* for val.
- **`split_strategy=track`** additionally keeps one player's track on one side.
  It needs real track ids, and it protects against a subtler leak: the same
  player in two different clips. Use it once your track ids are reliable; until
  then `video` is the safe default.

---

## Quick checklist

Before exporting:

- [ ] Rim boxes tight on the ring — **no net, no backboard** — and consistent
- [ ] Ball labelled in flight and through the rim, not just when held
- [ ] Both baskets and both court ends labelled, including the small far end
- [ ] Every person on court is either `player` or `referee`; nobody skipped
- [ ] Jersey boxes only where **you** can read the number
- [ ] Track ids consistent for the same person within a clip
- [ ] Every frame you touched is labelled *completely*, all classes
- [ ] At least 4 distinct videos, so a by-video split can exist
- [ ] Export reports non-zero counts for all five classes

The export prints `annotation_counts_by_class` and lists
`classes_with_no_annotations`. Read it. A class sitting at zero means a training
run that reports mAP 0 for it and leaves you wondering whether the model or the
data is at fault.
