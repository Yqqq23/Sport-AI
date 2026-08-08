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
    clean_jersey_number,
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

print()
if failures:
    print(f"FAILED ({len(failures)}):")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("All jersey logic tests passed.")
