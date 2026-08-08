"""Output 3: per-player made-shot records, written as CSV and JSON.

Consumes ShotEvent objects from src/shot.py after the pipeline has enriched
them with a jersey number (src/jersey.py) and court coordinates (src/calib.py).

Deliberately dependency-free: only the standard library. This module must stay
importable and testable without torch, cv2 or ultralytics.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

LOGGER = logging.getLogger(__name__)


#: Column order for the CSV. The first seven are the fields required by the
#: brief; the rest are provenance that makes a record auditable after the fact.
SHOT_RECORD_FIELDS: tuple[str, ...] = (
    "jersey_number",
    "track_id",
    "timestamp_sec",
    "frame_idx",
    "court_x",
    "court_y",
    "made",
    "release_frame_idx",
    "shot_position_x",
    "shot_position_y",
)


@dataclass
class ShotRecord:
    """One made shot, flattened for serialisation.

    ``jersey_number`` and ``track_id`` are both optional and independently so:
    a shot can be attributed to a track whose number never stabilised (number
    None, track set), and in principle a basket may have no attributable
    shooter at all (both None). Neither case is invented away - see
    ``unattributed_count``.
    """

    jersey_number: Optional[str]
    track_id: Optional[int]
    timestamp_sec: float
    frame_idx: int
    court_x: Optional[float]
    court_y: Optional[float]
    made: bool
    release_frame_idx: Optional[int] = None
    shot_position_x: Optional[float] = None
    shot_position_y: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "jersey_number": self.jersey_number,
            "track_id": self.track_id,
            "timestamp_sec": round(float(self.timestamp_sec), 4),
            "frame_idx": int(self.frame_idx),
            "court_x": None if self.court_x is None else round(float(self.court_x), 3),
            "court_y": None if self.court_y is None else round(float(self.court_y), 3),
            "made": bool(self.made),
            "release_frame_idx": self.release_frame_idx,
            "shot_position_x": (
                None if self.shot_position_x is None else round(float(self.shot_position_x), 2)
            ),
            "shot_position_y": (
                None if self.shot_position_y is None else round(float(self.shot_position_y), 2)
            ),
        }


def record_from_shot_event(event: Any) -> ShotRecord:
    """Flatten a src.shot.ShotEvent into a ShotRecord.

    Takes ``Any`` rather than importing ShotEvent so this module keeps zero
    internal imports and stays trivially testable with a stub.
    """
    court = getattr(event, "court_xy", None)
    position = getattr(event, "shot_position_xy", None)
    return ShotRecord(
        jersey_number=getattr(event, "shooter_jersey_number", None),
        track_id=getattr(event, "shooter_track_id", None),
        timestamp_sec=float(getattr(event, "timestamp_sec", 0.0)),
        frame_idx=int(getattr(event, "frame_idx", 0)),
        court_x=None if court is None else float(court[0]),
        court_y=None if court is None else float(court[1]),
        made=bool(getattr(event, "made", True)),
        release_frame_idx=getattr(event, "release_frame_idx", None),
        shot_position_x=None if position is None else float(position[0]),
        shot_position_y=None if position is None else float(position[1]),
    )


@dataclass
class PlayerShotSummary:
    """Per-player roll-up. Keyed by jersey number when known, else by track."""

    key: str
    jersey_number: Optional[str]
    track_ids: list[int]
    made_count: int
    timestamps_sec: list[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "jersey_number": self.jersey_number,
            "track_ids": sorted(self.track_ids),
            "made_count": self.made_count,
            "timestamps_sec": [round(t, 4) for t in self.timestamps_sec],
        }


class ShotRecorder:
    """Accumulates shot records and writes output 3.

    Streaming friendly: ``add`` may be called as events arrive, and ``write``
    once at the end.
    """

    def __init__(self) -> None:
        self.records: list[ShotRecord] = []

    # -- accumulation ----------------------------------------------------

    def add(self, record: ShotRecord) -> None:
        self.records.append(record)

    def add_event(self, event: Any) -> ShotRecord:
        record = record_from_shot_event(event)
        self.add(record)
        return record

    def extend_events(self, events: Iterable[Any]) -> None:
        for event in events:
            self.add_event(event)

    # -- derived views ---------------------------------------------------

    @property
    def made_count(self) -> int:
        return sum(1 for r in self.records if r.made)

    @property
    def unattributed_count(self) -> int:
        """Made shots with no shooter track. Reported, never hidden."""
        return sum(1 for r in self.records if r.made and r.track_id is None)

    @property
    def unnumbered_count(self) -> int:
        """Made shots attributed to a track whose jersey number never locked."""
        return sum(1 for r in self.records if r.made and r.track_id is not None and not r.jersey_number)

    def per_player(self) -> list[PlayerShotSummary]:
        """Group made shots per player.

        Grouped by jersey number when available, since one player may span
        several track ids after an ID switch. Falls back to the track id, and
        finally to a single "unattributed" bucket.
        """
        buckets: dict[str, PlayerShotSummary] = {}
        for record in self.records:
            if not record.made:
                continue
            if record.jersey_number:
                key = f"#{record.jersey_number}"
            elif record.track_id is not None:
                key = f"track_{record.track_id}"
            else:
                key = "unattributed"

            summary = buckets.get(key)
            if summary is None:
                summary = PlayerShotSummary(
                    key=key,
                    jersey_number=record.jersey_number,
                    track_ids=[],
                    made_count=0,
                    timestamps_sec=[],
                )
                buckets[key] = summary

            summary.made_count += 1
            summary.timestamps_sec.append(float(record.timestamp_sec))
            if record.track_id is not None and record.track_id not in summary.track_ids:
                summary.track_ids.append(int(record.track_id))

        return sorted(buckets.values(), key=lambda s: (-s.made_count, s.key))

    # -- writing ---------------------------------------------------------

    def write_csv(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # newline="" is required for the csv module to avoid blank rows on Windows.
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(SHOT_RECORD_FIELDS))
            writer.writeheader()
            for record in self.records:
                writer.writerow(record.to_dict())
        LOGGER.info("Wrote %s shot record(s) to %s", len(self.records), path)
        return path

    def write_json(self, path: Path, meta: Optional[dict[str, Any]] = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta": {
                **(meta or {}),
                "made_count": self.made_count,
                "unattributed_count": self.unattributed_count,
                "unnumbered_count": self.unnumbered_count,
            },
            "shots": [record.to_dict() for record in self.records],
            "per_player": [summary.to_dict() for summary in self.per_player()],
        }
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        LOGGER.info("Wrote shot JSON to %s", path)
        return path

    def write(
        self,
        output_dir: Path,
        stem: str = "shots",
        meta: Optional[dict[str, Any]] = None,
    ) -> tuple[Path, Path]:
        """Write both formats. Returns (csv_path, json_path)."""
        output_dir = Path(output_dir)
        return (
            self.write_csv(output_dir / f"{stem}.csv"),
            self.write_json(output_dir / f"{stem}.json", meta=meta),
        )

    def format_summary(self) -> str:
        """Human-readable roll-up for stdout at the end of a run."""
        lines = [
            "",
            "=" * 58,
            f"{'MADE SHOTS':<28}{self.made_count:>30}",
            "-" * 58,
        ]
        summaries = self.per_player()
        if not summaries:
            lines.append("  no made shots detected")
        else:
            lines.append(f"  {'player':<16}{'made':>8}   timestamps (s)")
            for summary in summaries:
                stamps = ", ".join(f"{t:.1f}" for t in summary.timestamps_sec[:8])
                if len(summary.timestamps_sec) > 8:
                    stamps += ", ..."
                lines.append(f"  {summary.key:<16}{summary.made_count:>8}   {stamps}")

        if self.unattributed_count:
            lines.append(
                f"  NOTE: {self.unattributed_count} made shot(s) had no attributable shooter"
            )
        if self.unnumbered_count:
            lines.append(
                f"  NOTE: {self.unnumbered_count} made shot(s) have a track but no stable jersey number"
            )
        lines.append("=" * 58)
        return "\n".join(lines)


def write_shot_outputs(
    events: Sequence[Any],
    output_dir: Path,
    stem: str = "shots",
    meta: Optional[dict[str, Any]] = None,
) -> tuple[Path, Path]:
    """One-shot convenience wrapper for a finished list of ShotEvents."""
    recorder = ShotRecorder()
    recorder.extend_events(events)
    return recorder.write(output_dir=output_dir, stem=stem, meta=meta)
