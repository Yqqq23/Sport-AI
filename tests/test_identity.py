"""Pure-logic tests for the tracker-id -> jersey-number identity layer.

Deliberately free of torch / cv2 / ultralytics so they run in the local Windows
editing environment. Run with:  python tests/test_identity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from identity import PlayerIdentityRegistry, SlotTracker, identity_color  # noqa: E402

failures: list[str] = []


def check(name: str, got, want) -> None:
    if got != want:
        failures.append(f"{name}: got {got!r}, want {want!r}")
    else:
        print(f"  ok  {name}")


# --- a track with no number is provisional -------------------------------
# THE LABEL IS THE SLOT, NEVER THE TRACKER ID. A registry that caps the number
# of identities but still prints "id:81" has bounded the wrong thing.
print("provisional identity")
reg = PlayerIdentityRegistry()
a = reg.resolve(3, None)
check("label is the slot, not the track id", a.label, "id:3")
check("slot assigned", a.slot, 3)
check("not confirmed", a.is_confirmed, False)
check("no number", a.number, None)

# --- locking a number promotes the identity ------------------------------
print("promotion on lock")
b = reg.resolve(3, "7")
check("label becomes the number", b.label, "id:3 #7")
check("confirmed", b.is_confirmed, True)
check("keyed on the number", b.key, "jersey:7")

# --- THE point of the sticky mapping -------------------------------------
# OCR runs every ocr_interval frames, so most frames supply number=None. The
# label must not flicker back to "id:3" on those frames.
print("sticky across frames without an OCR observation")
c = reg.resolve(3, None)
check("still shows the number", c.label, "id:3 #7")
check("same identity object", c.key, b.key)

# --- THE point of merging on number --------------------------------------
# The tracker loses the player behind an occlusion and re-acquires them under a
# brand new id. Same number means same person, so both ids collapse into one
# identity - which is what lets the top-down trail continue instead of restarting.
print("merge when a new track id locks the same number")
d = reg.resolve(91, "7")
check("merged into the existing identity", d.key, b.key)
check("both track ids recorded", sorted(d.track_ids), [3, 91])
check("new track is sticky too", reg.resolve(91, None).label, "id:3 #7")

# --- different numbers stay separate -------------------------------------
print("distinct numbers stay distinct")
e = reg.resolve(12, "23")
check("own identity", e.key, "jersey:23")
check("own label", e.label, "id:12 #23")
check("does not steal the other track", sorted(reg.resolve(3, None).track_ids), [3, 91])

# --- reporting ------------------------------------------------------------
print("reporting")
check("numbers in first-seen order", reg.confirmed_numbers(), ["7", "23"])
check("summary maps identity -> track ids", reg.summary()["jersey:7"], [3, 91])

# --- colour ---------------------------------------------------------------
print("colour")
check("confirmed identities differ in colour", identity_color(b) != identity_color(e), True)
check("colour is stable for one identity", identity_color(b), identity_color(reg.resolve(3, "7")))

# Provisional identities must be visibly duller than confirmed ones, so
# "not recognized yet" cannot be mistaken for a recognized player.
prov = PlayerIdentityRegistry().resolve(5, None)
conf = PlayerIdentityRegistry().resolve(5, "5")
check(
    "provisional is less saturated than confirmed",
    (max(identity_color(prov)) - min(identity_color(prov)))
    < (max(identity_color(conf)) - min(identity_color(conf))),
    True,
)

# Six players on a 3x3 court must all get well-separated hues. Spacing by
# appearance order (not by number value) is what guarantees this: hashing the
# number puts #7 and #23 within 0.12 of each other on the wheel.
print("six players get six distinguishable colours")
reg6 = PlayerIdentityRegistry()
colors = [
    identity_color(reg6.resolve(i + 1, n))
    for i, n in enumerate(["7", "23", "4", "11", "9", "1"])
]
check("all six unique", len(set(colors)), 6)
closest = min(
    sum(abs(x - y) for x, y in zip(c1, c2))
    for i, c1 in enumerate(colors)
    for c2 in colors[i + 1 :]
)
check("closest pair is well separated (L1 > 120)", closest > 120, True)

# --- roster cap -----------------------------------------------------------
# The complaint this solves: a tracker that churns ids produces dozens of
# "id:47" labels over a few minutes, on a court that only ever holds 6 players.
print("roster cap")
# The bound now lives in SlotTracker: it assigns ids from POSITION, because the
# MOT backend was returning boxes with no track ids at all - which erased the
# on-screen ids AND, since jersey ownership needed an owner id, discarded every
# jersey crop before OCR could run. Both symptoms, one cause.
tracker = SlotTracker(max_players=6)

boxes = [[100 * i, 400, 100 * i + 60, 580] for i in range(6)]
slots = tracker.assign(boxes, frame_idx=0)
check("six boxes -> six distinct slots", sorted(slots), [1, 2, 3, 4, 5, 6])

# Same players, slightly moved: slots must be STABLE frame to frame.
moved = [[b[0] + 8, b[1] + 4, b[2] + 8, b[3] + 4] for b in boxes]
check("slots stay with the same players", tracker.assign(moved, frame_idx=1), slots)

# Reordered input must not reshuffle identities - matching is by position.
shuffled = [moved[3], moved[0], moved[5], moved[1], moved[4], moved[2]]
want = [slots[3], slots[0], slots[5], slots[1], slots[4], slots[2]]
check("order of the input does not matter", tracker.assign(shuffled, frame_idx=2), want)

# A 7th box while all six are active gets no slot rather than inventing one.
seventh = moved + [[1500, 400, 1560, 580]]
check("7th box gets slot 0", tracker.assign(seventh, frame_idx=3)[6], 0)

# THE REGRESSION THAT MATTERED: ids must never leave 1..6, however long it runs.
long_run = SlotTracker(max_players=6)
seen = set()
for frame in range(300):
    drifting = [[100 * i + frame * 2, 400, 100 * i + 60 + frame * 2, 580] for i in range(6)]
    seen.update(long_run.assign(drifting, frame_idx=frame))
check("300 frames -> ids never leave 1..6", seen, {1, 2, 3, 4, 5, 6})

# And a registry fed those slots draws exactly them.
reg6 = PlayerIdentityRegistry(max_players=6)
labels = {reg6.resolve(s, None, frame_idx=0).label for s in range(1, 7)}
check("labels are id:1..id:6", labels, {f"id:{i}" for i in range(1, 7)})

# A recognized number outranks a stale provisional slot.
numbered = PlayerIdentityRegistry(max_players=2)
numbered.resolve(1, None, frame_idx=0)
numbered.resolve(2, None, frame_idx=0)
numbered.resolve(2, "7", frame_idx=10)
check("number promotes an existing slot", "jersey:7" in numbered.summary(), True)
# The roster bound now lives in SlotTracker, which only ever emits 1..max.
check("roster respected", len(numbered.summary()), 2)

check("unbounded by default", PlayerIdentityRegistry().max_players, None)

# THE PROMOTION BUG: recognition must not need a NEW slot. With a full roster of
# provisional players all visible on the same frame there is nothing free and
# nothing stale, so allocating instead of inheriting dropped the newly
# recognized player to slot 0 - overflow grey, unregistered, flickering label.
print("promotion inherits the slot")
# Slots now come from SlotTracker (1..max_players) and are passed straight in.
full = PlayerIdentityRegistry(max_players=6)
for slot in range(1, 7):
    full.resolve(slot, None, frame_idx=0)
check("slot is used verbatim", full.resolve(3, None, frame_idx=0).slot, 3)
promoted = full.resolve(3, "7", frame_idx=0)  # same frame: every slot active
check("promoted player keeps its slot", promoted.slot, 3)
check("label shows BOTH slot and number", promoted.label, "id:3 #7")
check("roster did not grow", len(full.summary()), 6)
check("provisional identity retired", "track:3" in full.summary(), False)
check("numbered identity registered", "jersey:7" in full.summary(), True)
check("colour is not the overflow grey", identity_color(promoted) != (150, 150, 150), True)
# and it must be sticky on the next frame
check("sticky after promotion", full.resolve(3, None, frame_idx=1).label, "id:3 #7")

# --- degenerate input -----------------------------------------------------
print("degenerate input")
check("None track id does not raise", reg.resolve(None, None).is_confirmed, False)

print()
if failures:
    print(f"FAILED ({len(failures)}):")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("All identity tests passed.")

