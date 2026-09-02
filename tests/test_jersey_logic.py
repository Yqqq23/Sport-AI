"""Pure-logic tests for the salvaged jersey voting/locking state machine.

Deliberately free of torch / cv2 / ultralytics so they run in the local Windows
editing environment. Run with:  python tests/test_jersey_logic.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from jersey import (  # noqa: E402
    JerseyNumberStabilizer,
    JerseyStabilizerConfig,
    OCRGateConfig,
    RoboflowJerseyOCR,
    clean_jersey_number,
    digits_from_scene_text,
    should_run_ocr,
)

failures: list[str] = []


def check(name: str, got, want) -> None:
    if got != want:
        failures.append(f"{name}: got {got!r}, want {want!r}")
    else:
        print(f"  ok  {name}")


# --- NEW behaviour: leading-zero normalization ---------------------------
print("clean_jersey_number")
check("'01' -> '1'", clean_jersey_number("01"), "1")
check("'1' -> '1'", clean_jersey_number("1"), "1")
check("'0' stays '0'", clean_jersey_number("0"), "0")
check("'07' -> '7'", clean_jersey_number("07"), "7")
check("'23' unchanged", clean_jersey_number("23"), "23")
check("'10' unchanged", clean_jersey_number("10"), "10")
# --- salvaged behaviour preserved ---------------------------------------
check("'00' rejected", clean_jersey_number("00"), None)
check("3+ digits rejected", clean_jersey_number("123"), None)
check("empty rejected", clean_jersey_number(""), None)
check("None rejected", clean_jersey_number(None), None)
check("letters stripped '#7'", clean_jersey_number("#7"), "7")

# --- NEW behaviour: OCR gate --------------------------------------------
print("should_run_ocr")
gate = OCRGateConfig(min_box_width=24, min_box_height=32, min_laplacian_variance=0.0)
check("large box passes", should_run_ocr([0, 0, 40, 60], gate), True)
check("narrow box rejected", should_run_ocr([0, 0, 10, 60], gate), False)
check("short box rejected", should_run_ocr([0, 0, 40, 10], gate), False)
check("exact threshold passes", should_run_ocr([0, 0, 24, 32], gate), True)

# --- salvaged voting/locking state machine ------------------------------
print("stabilizer: nothing decided before quorum")
cfg = JerseyStabilizerConfig(number_vote_window=15, slot_lock_min_votes=3, slot_reassign_patience=8)
s = JerseyNumberStabilizer(cfg)
# required_recent_votes = max(2, min(15, max(3, 3))) = 3
check("after 1 obs", s.update(1, "7", 0.9, 0), None)
check("after 2 obs", s.update(1, "7", 0.9, 1), None)
check("after 3 obs -> quorum", s.update(1, "7", 0.9, 2), "7")
check("not locked yet (1 lock frame)", s.is_locked(1), False)
s.update(1, "7", 0.9, 3)
s.update(1, "7", 0.9, 4)
check("locked after 3 lock frames", s.is_locked(1), True)
check("stable number", s.get_stable_number(1), "7")

print("stabilizer: low-confidence observations never vote")
s2 = JerseyNumberStabilizer(cfg)
for f in range(10):
    s2.update(2, "9", 0.50, f)  # below the salvaged 0.55 gate
check("no number from sub-0.55 obs", s2.get_stable_number(2), None)

print("stabilizer: '01' and '1' vote together (regression for the new fix)")
s3 = JerseyNumberStabilizer(cfg)
s3.update(3, "01", 0.9, 0)
s3.update(3, "1", 0.9, 1)
s3.update(3, "01", 0.9, 2)
check("mixed 01/1 reaches quorum as '1'", s3.get_stable_number(3), "1")

print("stabilizer: locked number resists a brief conflicting burst")
s4 = JerseyNumberStabilizer(cfg)
for f in range(6):
    s4.update(4, "5", 0.9, f)
check("locked on 5", s4.is_locked(4), True)
for f in range(6, 9):  # 3 conflicting obs, patience is 8
    s4.update(4, "6", 0.9, f)
check("still 5 during conflict", s4.get_stable_number(4), "5")

print("stabilizer: sustained conflict eventually reassigns")
s5 = JerseyNumberStabilizer(JerseyStabilizerConfig(slot_reassign_patience=2))
for f in range(6):
    s5.update(5, "5", 0.9, f)
check("locked on 5", s5.get_stable_number(5), "5")
for f in range(6, 20):
    s5.update(5, "6", 0.9, f)
check("reassigned to 6", s5.get_stable_number(5), "6")

print("stabilizer: tracks are independent")
s6 = JerseyNumberStabilizer(cfg)
for f in range(5):
    s6.update(10, "11", 0.9, f)
    s6.update(20, "22", 0.9, f)
check("track 10", s6.get_stable_number(10), "11")
check("track 20", s6.get_stable_number(20), "22")
check("unknown track", s6.get_stable_number(99), None)

# --- scene-text homoglyph repair -----------------------------------------
# PARSeq decodes over an alphanumeric charset, so it emits LETTERS for jersey
# digits. Measured: reading "0" as the letter "O" was the single largest error
# source (both failures in the first 24-crop probe). Repairing it took that run
# from 22/24 to 23/24.
print("digits_from_scene_text")
check("'O' -> '0' (the measured failure)", digits_from_scene_text("O"), "0")
check("'o' -> '0'", digits_from_scene_text("o"), "0")
check("'l1' -> '11'", digits_from_scene_text("l1"), "11")
check("'S' -> '5'", digits_from_scene_text("S"), "5")
check("'B' -> '8'", digits_from_scene_text("B"), "8")
check("'Z3' -> '23'", digits_from_scene_text("Z3"), "23")
check("plain digits untouched", digits_from_scene_text("23"), "23")
check("'#7' -> '7'", digits_from_scene_text("#7"), "7")
check("'O7' -> '7' (leading zero stripped)", digits_from_scene_text("O7"), "7")
# The map must NOT be so aggressive that junk decodes as a number. "b"->6 alone
# used to make "abc" read as "6"; that entry was removed after this test.
check("junk letters stay rejected", digits_from_scene_text("abc"), None)
check("'O0' is '00', which is rejected", digits_from_scene_text("O0"), None)
check("empty", digits_from_scene_text(""), None)
check("None", digits_from_scene_text(None), None)

# --- Roboflow response parsing -------------------------------------------
# Two shapes, decided by the project owner, not by us. Parsed offline here so
# the assembly logic is covered without spending a live API call per assertion.
print("roboflow response parsing")
rf = RoboflowJerseyOCR(model_id="test/1")


def det(cls, x, conf):
    return {"class": str(cls), "x": x, "y": 50, "width": 20, "height": 30, "confidence": conf}


# Per-digit detection. Roboflow returns digits in arbitrary order; the reader
# must sort by x. Given out of order here ON PURPOSE - this is the regression
# that would silently turn every "23" into "32".
check(
    "per-digit boxes assemble left to right",
    rf._parse({"predictions": [det(3, 111, 0.95), det(2, 44, 0.93)]}),
    ("23", 0.93),
)
check(
    "single digit",
    rf._parse({"predictions": [det(7, 60, 0.96)]}),
    ("7", 0.96),
)
# Confidence is the WEAKEST digit, not the mean: a barely-read digit makes the
# whole number barely read.
check(
    "confidence is the weakest digit",
    rf._parse({"predictions": [det(1, 40, 0.99), det(9, 90, 0.42)]}),
    ("19", 0.42),
)
check(
    "low-confidence digits are dropped before assembly",
    rf._parse({"predictions": [det(2, 44, 0.93), det(3, 111, 0.05)]}),
    ("2", 0.93),
)
check(
    "three digits are rejected by clean_jersey_number",
    rf._parse({"predictions": [det(1, 20, 0.9), det(2, 50, 0.9), det(3, 80, 0.9)]}),
    (None, 0.0),
)
check("no predictions", rf._parse({"predictions": []}), (None, 0.0))
check("junk response", rf._parse("not a dict"), (None, 0.0))

# Classification shape: predictions is a DICT keyed by class name.
check(
    "classification picks the top class",
    rf._parse({"predictions": {"7": {"confidence": 0.3}, "23": {"confidence": 0.88}}}),
    ("23", 0.88),
)
check(
    "classification ignores unparseable classes",
    rf._parse({"predictions": {"unknown": {"confidence": 0.99}, "5": {"confidence": 0.4}}}),
    ("5", 0.4),
)

print()
if failures:
    print(f"FAILED ({len(failures)}):")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("All jersey logic tests passed.")
