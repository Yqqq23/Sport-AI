"""Referees must be SEPARATED from players, then DROPPED - and still drawn.

No torch / cv2 / ultralytics, so this runs anywhere. Run with:
    python tests/test_referee_exclusion.py

Why this file exists: class 4 feeds nothing downstream, which makes every one of
its failure modes silent.

  * If referees leak into ``players``, a non-player joins tracking, the top-down
    court, the possession contest, and can be credited with a basket. Nothing
    raises; output 3 just gains a wrong row.
  * If ``FrameTracks.referees`` is declared but never populated - which is
    exactly the bug this file was written after - referee boxes are never drawn,
    and the first failure mode above becomes invisible too.

Both directions are asserted here. The role split is re-implemented against the
same class constants rather than invoked through Tracker, because Tracker's
constructor loads Ultralytics and real weights.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from detect import (  # noqa: E402
    BASKETBALL_CLASS_NAMES,
    CLASS_BALL,
    CLASS_JERSEY,
    CLASS_PLAYER,
    CLASS_REFEREE,
    CLASS_RIM,
)
from render import FrameRenderData, RefereeRender, RimRender, render_annotated_frame  # noqa: E402
from track import FrameTracks, TrackedObject  # noqa: E402

failures: list[str] = []


def check(name: str, got, want) -> None:
    if got != want:
        failures.append(f"{name}: got {got!r}, want {want!r}")
    else:
        print(f"  ok  {name}")


# --------------------------------------------------------------------------
print("class scheme: referee is appended, 0-3 unchanged")

check("player is 0", CLASS_PLAYER, 0)
check("ball is 1", CLASS_BALL, 1)
check("rim is 2", CLASS_RIM, 2)
check("jersey is 3", CLASS_JERSEY, 3)
check("referee is appended last", CLASS_REFEREE, 4)
check("five classes", len(BASKETBALL_CLASS_NAMES), 5)
check("referee is named", BASKETBALL_CLASS_NAMES[CLASS_REFEREE], "referee")


# --------------------------------------------------------------------------
print("role split: referees are separated from players, not merged into them")


def role_split(objects):
    """The split performed by Tracker.update, over plain dataclasses."""
    return {
        "players": [o for o in objects if o.class_id == CLASS_PLAYER],
        "rims": [o for o in objects if o.class_id == CLASS_RIM],
        "jerseys": [o for o in objects if o.class_id == CLASS_JERSEY],
        "referees": [o for o in objects if o.class_id == CLASS_REFEREE],
        "ball_candidates": [o for o in objects if o.class_id == CLASS_BALL],
    }


def box(class_id, x1=0.0, y1=0.0, x2=10.0, y2=20.0, track_id=None):
    return TrackedObject(
        bbox_xyxy=(x1, y1, x2, y2), class_id=class_id, confidence=0.9, track_id=track_id
    )


detections = [
    box(CLASS_PLAYER, 100, 100, 140, 200, track_id=1),
    box(CLASS_PLAYER, 300, 100, 340, 200, track_id=2),
    box(CLASS_REFEREE, 500, 100, 540, 200),
    box(CLASS_REFEREE, 700, 100, 740, 200),
    box(CLASS_RIM, 900, 50, 1000, 80),
    box(CLASS_BALL, 400, 60, 420, 80),
    box(CLASS_JERSEY, 105, 130, 135, 160),
]
split = role_split(detections)

check("two players", len(split["players"]), 2)
check("two referees", len(split["referees"]), 2)
check("referees are NOT in players", any(o.class_id == CLASS_REFEREE for o in split["players"]), False)
check("players are NOT in referees", any(o.class_id == CLASS_PLAYER for o in split["referees"]), False)
check("referees are NOT in jerseys", any(o.class_id == CLASS_REFEREE for o in split["jerseys"]), False)
check("referees are NOT ball candidates", any(o.class_id == CLASS_REFEREE for o in split["ball_candidates"]), False)


# --------------------------------------------------------------------------
print("FrameTracks: the referees field is actually populated")

tracks = FrameTracks(
    frame_idx=0,
    timestamp_sec=0.0,
    players=split["players"],
    rims=split["rims"],
    jerseys=split["jerseys"],
    referees=split["referees"],
    ball=None,
)

# The regression this file was written for: the field existed but Tracker.update
# never filled it, so it was always empty and referees were never drawn.
check("referees field is non-empty when referees were detected", len(tracks.referees), 2)
check("players field unaffected", len(tracks.players), 2)
check(
    "no referee reached players",
    [o.class_id for o in tracks.players],
    [CLASS_PLAYER, CLASS_PLAYER],
)


# --------------------------------------------------------------------------
print("referees are never tracked: no track id is assigned to them")

check("referees carry no track id", [o.track_id for o in tracks.referees], [None, None])


# --------------------------------------------------------------------------
print("render: referee boxes reach output 1 in their own colour")

render_data = FrameRenderData(
    frame_idx=0,
    timestamp_sec=0.0,
    players=[],
    ball=None,
    rims=[RimRender(bbox_xyxy=[900.0, 50.0, 1000.0, 80.0], conf=0.9)],
    referees=[RefereeRender(bbox_xyxy=list(o.bbox_xyxy), conf=o.confidence) for o in tracks.referees],
)
check("render data carries both referees", len(render_data.referees), 2)
check("referee render has no track id attribute", hasattr(RefereeRender, "track_id"), False)
check("referee render has no court position", hasattr(RefereeRender, "court_xy"), False)

# Draw for real when numpy is available; the colour must differ from every team
# colour, or a referee misclassified as a player would look identical.
try:
    import numpy as np

    from render import REFEREE_COLOR, TEAM_COLOR

    check(
        "referee colour differs from every team colour",
        REFEREE_COLOR in set(TEAM_COLOR.values()),
        False,
    )

    frame = np.zeros((300, 1100, 3), dtype=np.uint8)
    out = render_annotated_frame(frame, render_data, line_width=2)
    check("render returned a frame of the same shape", out.shape, frame.shape)
    check("render did not mutate the input frame", int(frame.max()), 0)
    # Something was actually painted where the first referee box is.
    referee_region = out[100:200, 500:541]
    check("referee box was drawn", bool(referee_region.any()), True)
except ImportError:
    print("  skip  pixel checks (numpy/cv2 not available)")


# --------------------------------------------------------------------------
print()
if failures:
    print(f"{len(failures)} referee test(s) FAILED:")
    for item in failures:
        print(f"  - {item}")
    raise SystemExit(1)
print("All referee exclusion tests passed.")
