"""Pure-logic tests for made-basket detection and shooter attribution.

No torch / cv2 / ultralytics, so these run in the local Windows editing
environment. Run with:  python tests/test_shot_logic.py

These matter more than usual: src/shot.py has no reference implementation to
diff against, so this file is the only evidence the state machine behaves.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from shot import (  # noqa: E402
    BasketDetector,
    PossessionTracker,
    ShotAnalyzer,
    ShotConfig,
    make_player_box,
    point_to_bbox_distance,
)

failures: list[str] = []


def check(name: str, got, want) -> None:
    if got != want:
        failures.append(f"{name}: got {got!r}, want {want!r}")
    else:
        print(f"  ok  {name}")


# Rim spans x 900..1000, y 500..530  ->  height 30, width 100
# above_line = 500 - 0.5*30 = 485 ; below_line = 530 + 0.5*30 = 545
# horizontal span with 15% margin = 885 .. 1015
RIM = (900.0, 500.0, 1000.0, 530.0)
CFG = ShotConfig()


def run_ball_path(path, rim=RIM, config=None, fps=30.0):
    """Feed a list of (x, y) ball centres, one per frame. Return fired events."""
    detector = BasketDetector(config or ShotConfig(), fps=fps)
    events = []
    for idx, center in enumerate(path):
        event = detector.update(
            frame_idx=idx, timestamp_sec=idx / fps, ball_center=center, rim_bbox=rim
        )
        if event is not None:
            events.append(event)
    return events


print("geometry helper")
check("point inside box -> 0", point_to_bbox_distance((950, 515), RIM), 0.0)
check("point left of box", point_to_bbox_distance((890, 515), RIM), 10.0)
check("point above box", point_to_bbox_distance((950, 490), RIM), 10.0)

print("made basket: descends through the rim")
made = run_ball_path([(950, 400), (950, 450), (950, 510), (950, 560), (950, 600)])
check("one event fired", len(made), 1)
check("marked made", made[0].made if made else None, True)

print("no event: ball rises through the rim plane")
rising = run_ball_path([(950, 600), (950, 560), (950, 510), (950, 450), (950, 400)])
check("no event", len(rising), 0)

print("no event: ball descends outside the rim span (airball wide)")
wide = run_ball_path([(700, 400), (700, 450), (700, 510), (700, 560), (700, 600)])
check("no event", len(wide), 0)

print("no event: ball descends but stops above the rim")
short = run_ball_path([(950, 300), (950, 350), (950, 400), (950, 440)])
check("no event", len(short), 0)

print("no event: ball enters rim then bounces back up (rim-out)")
rim_out = run_ball_path([(950, 400), (950, 460), (950, 510), (950, 460), (950, 400)])
check("no event", len(rim_out), 0)

print("fast ball: skips the inside frame entirely, still counts")
fast = run_ball_path([(950, 400), (950, 420), (950, 600)])
check("one event", len(fast), 1)

print("cooldown: a single make cannot fire twice while the ball bounces")
cfg_cool = ShotConfig()
detector = BasketDetector(cfg_cool, fps=30.0)
fired = 0
# Two full crossings back to back, well inside the 45-frame cooldown.
path = [(950, 400), (950, 450), (950, 510), (950, 560)] * 2
for idx, center in enumerate(path):
    if detector.update(idx, idx / 30.0, center, RIM) is not None:
        fired += 1
check("fires once, not twice", fired, 1)

print("hovering above the rim then dropping through still counts")
# The ABOVE stage deliberately re-arms while the ball remains above the rim,
# so the timeout measures the descent rather than how long the ball loitered.
slow_cfg = ShotConfig(max_frames_above_to_below=3)
hover = run_ball_path(
    [(950, 400)] + [(950, 410)] * 10 + [(950, 510), (950, 560)], config=slow_cfg
)
check("event still fires", len(hover), 1)

print("timeout: ball rattling on the rim is eventually abandoned")
# y=510 sits between above_line(485) and below_line(545): inside the rim, not
# through it. After max_frames_above_to_below the sequence is given up on, so
# a later drop below does not retroactively count.
rattle = run_ball_path(
    [(950, 400)] + [(950, 510)] * 10 + [(950, 560)], config=slow_cfg
)
check("no event after rattle timeout", len(rattle), 0)

print("no rim detected -> never fires")
detector_norim = BasketDetector(ShotConfig(), fps=30.0)
none_fired = [
    detector_norim.update(i, i / 30.0, (950, y), None)
    for i, y in enumerate([400, 450, 510, 560])
]
check("all None", [e for e in none_fired if e is not None], [])

# ---------------------------------------------------------------- possession
print("shooter attribution: last sustained holder before release")
pt = PossessionTracker(ShotConfig(min_possession_frames=5), fps=30.0)
for f in range(10):  # player 7 holds frames 0-9
    pt.update(f, f / 30.0, (100.0, 100.0), [make_player_box(7, (60.0, 60.0, 140.0, 200.0))])
for f in range(10, 21):  # ball in flight, nobody near
    pt.update(f, f / 30.0, (900.0, 400.0), [make_player_box(7, (60.0, 60.0, 140.0, 200.0))])
tid, ground, release = pt.find_shooter(basket_frame_idx=20)
check("shooter is 7", tid, 7)
check("release frame is last held frame", release, 9)
check("release position is foot midpoint", ground, (100.0, 200.0))

print("shooter attribution: brief flicker does not steal credit")
pt2 = PossessionTracker(ShotConfig(min_possession_frames=5), fps=30.0)
near = (100.0, 100.0)
p7 = make_player_box(7, (60.0, 60.0, 140.0, 200.0))
p3 = make_player_box(3, (60.0, 60.0, 140.0, 200.0))
for f in range(10):  # player 7 holds for 10 frames
    pt2.update(f, f / 30.0, near, [p7])
pt2.update(10, 10 / 30.0, near, [p3])  # player 3 for a single frame
for f in range(11, 21):
    pt2.update(f, f / 30.0, (900.0, 400.0), [p7])
tid2, _, _ = pt2.find_shooter(basket_frame_idx=20)
check("credit stays with 7", tid2, 7)

print("shooter attribution: two sustained holders, latest wins")
pt3 = PossessionTracker(ShotConfig(min_possession_frames=5), fps=30.0)
for f in range(0, 6):
    pt3.update(f, f / 30.0, near, [p3])
for f in range(6, 14):
    pt3.update(f, f / 30.0, near, [p7])
for f in range(14, 21):
    pt3.update(f, f / 30.0, (900.0, 400.0), [p7])
tid3, _, rel3 = pt3.find_shooter(basket_frame_idx=20)
check("latest sustained holder wins", tid3, 7)
check("release frame", rel3, 13)

print("shooter attribution: nobody held long enough -> None")
pt4 = PossessionTracker(ShotConfig(min_possession_frames=5), fps=30.0)
for f in range(21):
    pt4.update(f, f / 30.0, (900.0, 400.0), [p7])  # ball never near a player
tid4, _, _ = pt4.find_shooter(basket_frame_idx=20)
check("no shooter", tid4, None)

print("possession requires proximity")
pt5 = PossessionTracker(ShotConfig(possession_max_distance=10.0), fps=30.0)
holder = pt5.update(0, 0.0, (500.0, 500.0), [make_player_box(7, (60.0, 60.0, 140.0, 200.0))])
check("far player is not the holder", holder, None)

# ------------------------------------------------------------- integration
print("ShotAnalyzer end to end: make is attributed to the shooter")
analyzer = ShotAnalyzer(ShotConfig(min_possession_frames=3), fps=30.0)
shooter_box = make_player_box(11, (900.0, 700.0, 1000.0, 900.0))
event = None
# Shooter holds the ball low, then the ball arcs down through the rim.
sequence = [(950.0, 750.0)] * 5 + [(950.0, 400.0), (950.0, 460.0), (950.0, 510.0), (950.0, 560.0)]
for idx, ball in enumerate(sequence):
    got = analyzer.update(idx, idx / 30.0, ball, RIM, [shooter_box])
    if got is not None:
        event = got
check("event produced", event is not None, True)
check("attributed to 11", event.shooter_track_id if event else None, 11)
check("shot position recorded", event.shot_position_xy if event else None, (950.0, 900.0))
check("analyzer stored the event", len(analyzer.events), 1)

print()
if failures:
    print(f"FAILED ({len(failures)}):")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("All shot logic tests passed.")
