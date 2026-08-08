"""Jersey number recognition and cross-frame stabilization.

Salvaged from:
  SRC_A/fixed7_jersey_tracking_ocr7/run_basketball_pose_tracking_fixed7_ocr7_stable.py
    - clean_jersey_number                  (lines 89-97)
    - trim_deque                           (lines 183-185)
    - majority_number_from_recent_history  (lines 188-200)
    - update_slot_number_state             (lines 1583-1625)
    - NumberRecognizer tesseract branch    (lines 798-816)

The voting / locking numeric behaviour is preserved verbatim. Every literal
constant below (0.55 confidence gate, the 0.85 vote-weight split, the
``max(2, min(window, max(3, lock_min_votes)))`` quorum, and the
``min(1.0, 0.55 + 0.1 * count + 0.2 * conf)`` lock-confidence formula) comes
from the source and must not be retuned without an experiment.

Two behaviours are NEW (they did not exist in any salvaged variant and were
explicitly requested):
  1. leading-zero normalization, so "01" and "1" collapse to one number
  2. an OCR input gate, so OCR only runs on boxes large/sharp enough to read

``cv2`` is imported lazily inside the OCR backend and the sharpness gate so
that the pure-Python voting logic in this module stays importable and unit
testable in an environment without OpenCV.
"""

from __future__ import annotations

import re
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Protocol

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np


# --------------------------------------------------------------------------
# Number cleaning / normalization
# --------------------------------------------------------------------------


def clean_jersey_number(text: Optional[str]) -> Optional[str]:
    """Reduce raw OCR text to a canonical jersey number, or None.

    Salvaged verbatim from the ``_stable`` variant (lines 89-97): keep digits
    only, reject anything longer than two digits, reject "00".

    NEW (not in any source variant): strip leading zeros so that "01" and "1"
    are treated as the same player. "0" itself stays a legal number. The source
    only ever did ``re.sub(r"\\D", "", ...)`` plus a ``[:2]`` truncation, so
    "01" and "1" voted as two separate numbers and could split a track's votes.
    """
    if not text:
        return None
    digits = re.sub(r"\D", "", str(text))
    if not digits or len(digits) > 2:
        return None
    if digits == "00":
        return None
    # --- NEW: leading-zero normalization ("01" -> "1", "0" stays "0") ---
    normalized = digits.lstrip("0")
    return normalized if normalized else "0"


# --------------------------------------------------------------------------
# OCR backend (pluggable)
# --------------------------------------------------------------------------


class JerseyOCRBackend(Protocol):
    """Any jersey OCR implementation. Returns (text, confidence)."""

    def recognize(self, crop: "np.ndarray") -> tuple[Optional[str], float]:
        ...


class PytesseractJerseyOCR:
    """Tesseract digit OCR.

    Preprocessing chain and the fixed 0.7 confidence are salvaged verbatim from
    the ``_stable`` variant, lines 798-816. Tesseract exposes no usable score
    here, so the source returned a constant 0.7; that is kept because the
    downstream vote weighting is calibrated against it (see
    ``JerseyNumberStabilizer.update``: < 0.85 counts as weight 1).
    """

    def __init__(self) -> None:
        self._available: Optional[bool] = None

    def is_available(self) -> bool:
        if self._available is None:
            try:
                import pytesseract  # noqa: F401

                pytesseract.get_tesseract_version()
                self._available = True
            except Exception:
                self._available = False
        return bool(self._available)

    def recognize(self, crop: "np.ndarray") -> tuple[Optional[str], float]:
        if crop is None or getattr(crop, "size", 0) == 0:
            return None, 0.0
        if not self.is_available():
            return None, 0.0
        try:
            import cv2
            import pytesseract

            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            text = pytesseract.image_to_string(
                bw, config="--psm 7 -c tessedit_char_whitelist=0123456789"
            )
            digits = re.findall(r"\d+", text)
            if not digits:
                return None, 0.0
            candidate = max(digits, key=len)
            candidate = clean_jersey_number(candidate)
            if not candidate:
                return None, 0.0
            return candidate, 0.7
        except Exception:
            return None, 0.0


# --------------------------------------------------------------------------
# OCR input gate  (NEW - no equivalent in any salvaged variant)
# --------------------------------------------------------------------------


@dataclass
class OCRGateConfig:
    """Thresholds deciding whether a crop is worth running OCR on.

    NEW in this project. The salvaged pipeline ran OCR on every matched box on
    every Nth frame regardless of how small or motion-blurred it was, which is
    the main source of junk votes. All values are config-driven; the defaults
    below are starting points, NOT tuned values.
    """

    min_box_width: int = 24
    min_box_height: int = 32
    # Set to 0.0 to disable the sharpness test (it costs one Laplacian per crop).
    min_laplacian_variance: float = 0.0


def laplacian_variance(crop: "np.ndarray") -> float:
    """Focus measure: variance of the Laplacian. Higher means sharper."""
    import cv2

    if crop is None or getattr(crop, "size", 0) == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def should_run_ocr(
    bbox_xyxy: list[float],
    cfg: OCRGateConfig,
    crop: Optional["np.ndarray"] = None,
) -> bool:
    """True when the box is big enough (and sharp enough) to attempt OCR.

    The size test needs only the bbox. The sharpness test is skipped unless a
    crop is supplied and ``min_laplacian_variance`` is positive.
    """
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    if (x2 - x1) < float(cfg.min_box_width):
        return False
    if (y2 - y1) < float(cfg.min_box_height):
        return False
    if cfg.min_laplacian_variance > 0.0 and crop is not None:
        if laplacian_variance(crop) < float(cfg.min_laplacian_variance):
            return False
    return True


# --------------------------------------------------------------------------
# Cross-frame stabilization
# --------------------------------------------------------------------------


@dataclass
class JerseyStabilizerConfig:
    """Voting / locking parameters.

    WARNING - see the Phase A report. ``number_vote_window``,
    ``slot_lock_min_votes`` and ``slot_reassign_patience`` are read as
    ``args.*`` by the source (lines 1593-1624) but are NEVER declared in its
    argparse block (lines 2268-2360) and never set via setattr/set_defaults.
    The source therefore raises AttributeError before it can lock a number, and
    no tuned value for these three survives anywhere in SRC_A. The values below
    are DEFAULTS CHOSEN HERE, not recovered constants, and need calibration.
    """

    number_vote_window: int = 15
    slot_lock_min_votes: int = 3
    slot_reassign_patience: int = 8
    # Salvaged verbatim (line 1590): observations weaker than this never vote.
    min_observation_confidence: float = 0.55
    # Salvaged verbatim (line 1595): >= this confidence counts as 2 votes.
    strong_vote_confidence: float = 0.85


@dataclass
class TrackNumberState:
    """Per-track jersey state.

    The number-related subset of ``FixedPlayerSlot`` (source lines 116-133).
    The bbox/centre/slot-assignment fields are intentionally not carried over:
    they belong to the fixed-slot assignment algorithm, not to number voting,
    and this project keys state on tracker IDs instead of fixed slots.
    """

    track_id: int
    jersey_number: Optional[str] = None
    number_votes: Counter = field(default_factory=Counter)
    is_locked: bool = False
    lock_number: Optional[str] = None
    lock_confidence: float = 0.0
    lock_frame_count: int = 0
    conflict_frame_count: int = 0
    recent_number_history: deque = field(default_factory=deque)
    recent_number_confidences: deque = field(default_factory=deque)
    last_seen_frame: int = -1


def trim_deque(values: deque, max_len: int) -> None:
    """Salvaged verbatim from source lines 183-185."""
    while len(values) > max_len:
        values.popleft()


def majority_number_from_recent_history(
    state: TrackNumberState, window: int
) -> tuple[Optional[str], int, float]:
    """Salvaged verbatim from source lines 188-200 (``slot`` renamed ``state``)."""
    if window <= 0 or not state.recent_number_history:
        return None, 0, 0.0
    recent_numbers = list(state.recent_number_history)[-window:]
    recent_confs = list(state.recent_number_confidences)[-window:]
    if not recent_numbers:
        return None, 0, 0.0

    counts = Counter(recent_numbers)
    best_number, best_count = counts.most_common(1)[0]
    confs = [recent_confs[i] for i, number in enumerate(recent_numbers) if number == best_number]
    mean_conf = float(sum(confs) / max(1, len(confs))) if confs else 0.0
    return best_number, int(best_count), mean_conf


class JerseyNumberStabilizer:
    """Aggregates per-frame OCR guesses into one stable number per track.

    The state machine in ``update`` is a direct transcription of
    ``update_slot_number_state`` (source lines 1583-1625). Only the plumbing
    changed: state is keyed by ``track_id`` and looked up here rather than
    being passed in as a pre-resolved slot, and parameters come from a config
    dataclass instead of an argparse namespace.
    """

    def __init__(self, config: Optional[JerseyStabilizerConfig] = None) -> None:
        self.config = config or JerseyStabilizerConfig()
        self.states: dict[int, TrackNumberState] = {}

    def _state(self, track_id: int) -> TrackNumberState:
        state = self.states.get(int(track_id))
        if state is None:
            state = TrackNumberState(track_id=int(track_id))
            self.states[int(track_id)] = state
        return state

    def update(
        self,
        track_id: int,
        raw_text: Optional[str],
        confidence: float,
        frame_idx: int,
    ) -> Optional[str]:
        """Feed one OCR observation for one track. Returns the current best number.

        ``raw_text`` may be None (OCR skipped or failed): the source still
        re-evaluates the lock on every call, so a skipped observation can still
        confirm a pending lock. ``frame_idx`` is recorded for observability and
        deliberately does not affect the voting decision, matching the source.
        """
        cfg = self.config
        state = self._state(track_id)
        state.last_seen_frame = int(frame_idx)

        observed_number = clean_jersey_number(raw_text)
        observed_conf = float(confidence)

        if observed_number is not None and observed_conf >= cfg.min_observation_confidence:
            state.recent_number_history.append(observed_number)
            state.recent_number_confidences.append(observed_conf)
            trim_deque(state.recent_number_history, int(cfg.number_vote_window))
            trim_deque(state.recent_number_confidences, int(cfg.number_vote_window))
            state.number_votes[observed_number] += (
                1 if observed_conf < cfg.strong_vote_confidence else 2
            )

        recent_number, recent_count, recent_conf = majority_number_from_recent_history(
            state, int(cfg.number_vote_window)
        )
        required_recent_votes = max(
            2, min(int(cfg.number_vote_window), max(3, int(cfg.slot_lock_min_votes)))
        )

        if recent_number is None or recent_count < required_recent_votes:
            return self.get_stable_number(track_id)

        if state.is_locked and state.lock_number and recent_number != state.lock_number:
            state.conflict_frame_count += 1
            if state.conflict_frame_count >= int(cfg.slot_reassign_patience):
                state.is_locked = False
                state.lock_number = None
                state.lock_confidence = 0.0
                state.lock_frame_count = 0
                state.conflict_frame_count = 0
            else:
                return self.get_stable_number(track_id)
        else:
            state.conflict_frame_count = 0

        state.jersey_number = recent_number
        if state.lock_number == recent_number or state.lock_number is None:
            state.lock_frame_count += 1
        else:
            state.lock_frame_count = 1
        state.lock_number = recent_number
        state.lock_confidence = min(1.0, 0.55 + 0.1 * recent_count + 0.2 * recent_conf)

        if state.lock_frame_count >= int(cfg.slot_lock_min_votes):
            state.is_locked = True

        return self.get_stable_number(track_id)

    def get_stable_number(self, track_id: int) -> Optional[str]:
        """The number to display for this track, or None if not yet decided.

        Prefers the locked number, mirroring the source's precedence
        (``slot.lock_number or slot.jersey_number``, line 1641).
        """
        state = self.states.get(int(track_id))
        if state is None:
            return None
        if state.is_locked and state.lock_number:
            return state.lock_number
        return state.jersey_number

    def get_confidence(self, track_id: int) -> float:
        state = self.states.get(int(track_id))
        return float(state.lock_confidence) if state is not None else 0.0

    def is_locked(self, track_id: int) -> bool:
        state = self.states.get(int(track_id))
        return bool(state.is_locked) if state is not None else False

    def snapshot(self) -> dict[int, dict[str, Any]]:
        """Debug view of every track's jersey state."""
        return {
            tid: {
                "jersey_number": s.jersey_number,
                "lock_number": s.lock_number,
                "is_locked": s.is_locked,
                "lock_confidence": s.lock_confidence,
                "lock_frame_count": s.lock_frame_count,
                "conflict_frame_count": s.conflict_frame_count,
                "last_seen_frame": s.last_seen_frame,
                "votes": dict(s.number_votes),
            }
            for tid, s in self.states.items()
        }
