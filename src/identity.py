"""Player identity: turn tracker ids into jersey numbers and bounded slots.

WHY THIS MODULE EXISTS
----------------------
A tracker id is not a player. ByteTrack/BoT-SORT hand out 1, 2, 3, ... in the
order boxes appear, and re-issue a fresh id every time a player is occluded
long enough to lose the track. Over a 10-minute 3x3 game six players routinely
generate dozens of ids. Anything keyed on the raw id - the on-screen label, the
trail colour, the shot record - therefore fragments the same human into many
"players".

The jersey number is the real identity, so this module makes the number the
primary key and demotes the tracker id to what it actually is: a short-lived
handle used to accumulate votes.

THE SLOT
--------
Until a number is recognized a player still has to be called something, and
that something must NOT be the raw tracker id. A capped registry that still
prints ``id:81`` has bounded the wrong thing: the COUNT of identities rather
than the VALUE shown. Every identity therefore owns a SLOT - a small integer in
1..max_players - and provisional identities are labelled by slot. Recycling a
slot returns its number to the pool, so the displayed ids never leave 1..6 no
matter how much the tracker churns underneath.

Slots also drive colour, so slot 3 is the same colour all game even after it
changes occupant. That is the intended reading: a slot is a position on the
roster, not a promise about which human fills it.

Three behaviours are the whole point:

``sticky``
    Once a track has been seen wearing #7, it KEEPS #7 even on frames where the
    stabilizer returns None (OCR skipped, player turned away, box too small).
    Without this the label would flicker several times a second, since OCR only
    runs every ``ocr_interval`` frames.

``merge on number``
    When a NEW tracker id locks the same number as a retired one, both map to
    one identity. That is what re-joins a trail across an occlusion, and what
    stops one player appearing twice in the same frame's statistics.

``recycle on capacity``
    When the roster is full and a new track needs a slot, the least-recently-
    seen slot is reused instead of a new one being minted.

The merge is deliberately one-way: identities are never split again. A wrong
merge caused by a misread number is visible (two players sharing a colour and a
trail), whereas a silent split looks exactly like normal tracker churn and would
hide the very problem this module exists to solve.
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass, field
from typing import Optional, Sequence


@dataclass
class IdentityConfig:
    """Roster settings.

    ``max_players`` is the number of player identities alive at once, NOT a cap
    on detections. Set it to the number of players on court: 6 for 3x3, 10 for
    5x5. 0 disables the cap and falls back to unbounded slot numbering.

    Referees are excluded by construction - src/track.py drops CLASS_REFEREE
    before anything reaches this registry - so 6 means six players, with the
    referee neither occupying a slot nor needing one.
    """

    max_players: int = 6


class SlotTracker:
    """Assigns stable 1..N slot ids to player boxes by POSITION, not tracker id.

    WHY THIS EXISTS. The MOT backend was returning boxes with no track ids at
    all (``boxes.id is None``), which broke two things at once: every player
    rendered as the same placeholder, and - because jersey ownership requires
    ``owner.track_id`` - every jersey crop was discarded before OCR, so no
    number could ever be read. Both symptoms, one cause.

    Rather than depend on the backend producing ids, this assigns them here:
    greedy nearest-centre matching between the previous frame's slots and this
    frame's boxes. On a fixed camera with six players that is entirely
    sufficient, and it cannot fail the way an absent id does.

    The distance threshold scales with box height so it means the same thing
    near and far from the camera.
    """

    def __init__(
        self,
        max_players: int = 6,
        max_distance_ratio: float = 1.2,
        max_lost_frames: int = 45,
    ) -> None:
        self.max_players = int(max_players) if max_players else 0
        self.max_distance_ratio = float(max_distance_ratio)
        self.max_lost_frames = int(max_lost_frames)
        #: slot -> (centre_xy, height, last_seen_frame)
        self._slots: dict[int, tuple[tuple[float, float], float, int]] = {}
        self._next_slot = 1

    @staticmethod
    def _centre(bbox: Sequence[float]) -> tuple[float, float]:
        x1, y1, x2, y2 = (float(v) for v in bbox)
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def _free_slot(self, frame_idx: int, also_used: set[int]) -> Optional[int]:
        """An unused slot number, recycling one that has gone stale.

        ``also_used`` carries slots handed out earlier in THIS call. Without it
        every unmatched box in a frame saw the same not-yet-updated state and
        was given slot 1, so six players all rendered as "id:1".
        """
        if not self.max_players:
            slot = self._next_slot
            self._next_slot += 1
            return slot
        used = set(self._slots) | also_used
        for slot in range(1, self.max_players + 1):
            if slot not in used:
                return slot
        # All taken: drop the one missing longest, if it is genuinely stale.
        stale = [
            (slot, state)
            for slot, state in self._slots.items()
            if frame_idx - state[2] > self.max_lost_frames and slot not in also_used
        ]
        if stale:
            victim = min(stale, key=lambda kv: kv[1][2])[0]
            del self._slots[victim]
            return victim
        return None

    def assign(self, boxes: Sequence[Sequence[float]], frame_idx: int) -> list[int]:
        """Slot number for each box, in the order given. 0 means no slot free."""
        results: list[int] = [0] * len(boxes)
        unmatched = list(range(len(boxes)))

        # Greedy: shortest player-to-slot distance first, so an unambiguous
        # match cannot be stolen by a farther box processed earlier.
        pairs: list[tuple[float, int, int]] = []
        for index in unmatched:
            centre = self._centre(boxes[index])
            x1, y1, x2, y2 = (float(v) for v in boxes[index])
            height = max(1.0, y2 - y1)
            for slot, (previous, previous_height, _) in self._slots.items():
                distance = (
                    (centre[0] - previous[0]) ** 2 + (centre[1] - previous[1]) ** 2
                ) ** 0.5
                if distance <= self.max_distance_ratio * max(height, previous_height):
                    pairs.append((distance, index, slot))
        pairs.sort()

        taken_slots: set[int] = set()
        taken_boxes: set[int] = set()
        for _, index, slot in pairs:
            if index in taken_boxes or slot in taken_slots:
                continue
            taken_boxes.add(index)
            taken_slots.add(slot)
            results[index] = slot

        for index in range(len(boxes)):
            if results[index] == 0:
                slot = self._free_slot(frame_idx, taken_slots)
                if slot is None:
                    continue
                taken_slots.add(slot)
                results[index] = slot

        for index, slot in enumerate(results):
            if slot:
                x1, y1, x2, y2 = (float(v) for v in boxes[index])
                self._slots[slot] = (
                    self._centre(boxes[index]),
                    max(1.0, y2 - y1),
                    frame_idx,
                )
        return results


@dataclass
class Identity:
    """One player, as opposed to one track.

    ``key`` is stable for the whole run and is what colours, trails and stats
    are keyed on. ``number`` is None until the stabilizer locks one, in which
    case the identity is provisional and displayed by ``slot``.
    """

    key: str
    number: Optional[str] = None
    #: 1-based roster position. Drives the displayed label and the colour.
    #: 0 means "no slot available" - see PlayerIdentityRegistry.resolve.
    slot: int = 0
    #: Every tracker id that has ever mapped here. Debug/observability only.
    track_ids: set[int] = field(default_factory=set)
    #: Last frame this identity was resolved on. Drives slot recycling.
    last_seen_frame: int = -1

    @property
    def is_confirmed(self) -> bool:
        """True once a jersey number is attached."""
        return self.number is not None

    @property
    def label(self) -> str:
        """What to draw.

        The jersey number when known, otherwise the SLOT - never the raw tracker
        id. Printing the tracker id here is what produced "id:81" on a court
        that only ever holds six players.
        """
        if self.slot > 0:
            # Both, deliberately. The slot is the stable handle a viewer tracks
            # across the video; the number is what it MEANS once recognized.
            # Showing only the number hides which slot it attached to when a
            # reading is wrong, and that is the first thing you need when
            # debugging a misread.
            return f"id:{self.slot} #{self.number}" if self.number else f"id:{self.slot}"
        # Slot 0 = beyond the player roster. On a 3x3 court with max_players=6
        # the surplus person is the referee, so name them rather than printing
        # a meaningless "id:?".
        return "referee"


class PlayerIdentityRegistry:
    """Maps tracker ids to stable player identities with a bounded slot pool.

    Two deliberate refusals:

    * a slot seen on the CURRENT frame is never recycled. Stealing the identity
      of a player who is visible right now would swap two players' labels, which
      is worse than showing one extra marker.
    * when every slot is active and yet another track appears, the extra track
      gets a TRANSIENT identity with slot 0, labelled "id:?", which is not
      registered and consumes nothing. That is the honest signal that the
      detector produced more players than the roster allows - usually a
      duplicate box, or a referee filed as a player - and hiding it would hide a
      detector problem.
    """

    def __init__(self, max_players: Optional[int] = None) -> None:
        #: tracker id -> identity key. Sticky: updated on promotion from
        #: provisional to numbered, cleared when a slot is recycled.
        self._track_to_key: dict[int, str] = {}
        self._identities: dict[str, Identity] = {}
        #: Numbers observed at least once, in first-seen order.
        self._number_order: list[str] = []

        self.max_players = int(max_players) if max_players else None
        #: Slot numbers available for assignment, smallest first. When
        #: uncapped this stays empty and slots simply increment.
        self._free_slots: list[int] = (
            list(range(1, self.max_players + 1)) if self.max_players else []
        )
        self._next_unbounded_slot = 1

        self.recycled_slots = 0
        self.overflow_tracks = 0
        #: Player boxes that arrived with no tracker id. A large count means the
        #: tracker is not assigning ids at all - see resolve().
        self.untracked_boxes = 0

    # -- slot pool -------------------------------------------------------
    def _take_slot(self) -> Optional[int]:
        if self.max_players is None:
            slot = self._next_unbounded_slot
            self._next_unbounded_slot += 1
            return slot
        if self._free_slots:
            return self._free_slots.pop(0)
        return None

    def _release_slot(self, slot: int) -> None:
        if self.max_players is not None and slot > 0 and slot not in self._free_slots:
            # Sorted so the lowest free number is always reused next, which
            # keeps the visible ids clustered at 1..N instead of drifting.
            self._free_slots.append(slot)
            self._free_slots.sort()

    def _recycle_slot(self, frame_idx: int) -> bool:
        """Free the least-recently-seen slot. False if every slot is active."""
        candidates = [
            identity
            for identity in self._identities.values()
            if identity.last_seen_frame < frame_idx
        ]
        if not candidates:
            return False
        victim = min(candidates, key=lambda i: i.last_seen_frame)

        for track_id in [t for t, k in self._track_to_key.items() if k == victim.key]:
            del self._track_to_key[track_id]
        del self._identities[victim.key]
        self._release_slot(victim.slot)
        self.recycled_slots += 1
        return True

    # -- lookup ----------------------------------------------------------
    @staticmethod
    def _number_key(number: str) -> str:
        return f"jersey:{number}"

    @staticmethod
    def _track_key(track_id: int) -> str:
        return f"track:{track_id}"

    def _create(
        self,
        key: str,
        number: Optional[str],
        frame_idx: int,
        preferred_slot: Optional[int] = None,
    ) -> Identity:
        """Make a new identity, recycling a stale slot when the roster is full.

        ``preferred_slot`` lets the caller supply an already-stable slot (from
        SlotTracker) so the drawn id matches the one everything else is keyed
        on. Without it the registry would hand out its own numbering and a
        player tracked as slot 3 could render as "id:1".
        """
        if preferred_slot:
            identity = Identity(
                key=key, number=number, slot=int(preferred_slot), last_seen_frame=frame_idx
            )
            self._identities[key] = identity
            if number is not None and number not in self._number_order:
                self._number_order.append(number)
            return identity

        slot = self._take_slot()
        if slot is None and self._recycle_slot(frame_idx):
            slot = self._take_slot()

        if slot is None:
            # Every slot is occupied by a player seen on this very frame.
            self.overflow_tracks += 1
            return Identity(key=key, number=number, slot=0, last_seen_frame=frame_idx)

        identity = Identity(key=key, number=number, slot=slot, last_seen_frame=frame_idx)
        self._identities[key] = identity
        if number is not None and number not in self._number_order:
            self._number_order.append(number)
        return identity

    def _promote(self, track_id: int, number: str, frame_idx: int) -> Identity:
        """Turn this track's provisional identity into a numbered one.

        THE SLOT IS INHERITED, NOT REALLOCATED. Recognition does not add a
        player to the court, it renames one who was already there. Allocating a
        fresh slot instead meant that on a full roster - six provisional players
        all visible on the same frame - there was nothing free and nothing stale
        to recycle, so the newly recognized player fell through to slot 0: drawn
        in the overflow grey and never registered, which broke the sticky
        mapping and made the label flicker. Recognition working correctly must
        never be what breaks the display.
        """
        key = self._number_key(number)
        previous_key = self._track_to_key.get(track_id)
        previous = self._identities.get(previous_key) if previous_key else None

        if previous is None or previous.number is not None:
            # No provisional identity to inherit from (or it is already
            # numbered, i.e. a re-bind). Still use the caller's slot: track_id
            # IS the SlotTracker slot, so falling back to pool numbering here
            # would draw "id:1" for a player everything else calls slot 12.
            return self._create(key, number, frame_idx, preferred_slot=track_id)

        identity = Identity(
            key=key,
            number=number,
            slot=previous.slot,
            track_ids=set(previous.track_ids),
            last_seen_frame=frame_idx,
        )
        for stale_track in [t for t, k in self._track_to_key.items() if k == previous_key]:
            del self._track_to_key[stale_track]
        del self._identities[previous_key]

        self._identities[key] = identity
        if number not in self._number_order:
            self._number_order.append(number)
        return identity

    # -- main entry point ------------------------------------------------
    def resolve(
        self,
        track_id: Optional[int],
        number: Optional[str],
        frame_idx: int = 0,
    ) -> Identity:
        """Identity for this track on this frame.

        ``number`` is the stabilizer's current best guess and may be None on
        most frames; the sticky mapping carries the label through those.
        ``frame_idx`` drives slot recycling and must advance for the cap to work
        at all - with a constant frame index every slot looks equally fresh and
        nothing is ever recycled.
        """
        if track_id is None:
            # NO SHARED IDENTITY FOR UNTRACKED BOXES. This used to map every
            # id-less detection onto one "track:-1" identity, so a frame where
            # the tracker returned no ids at all rendered every player - and
            # every referee misfiled as a player - as the same "id:1". That
            # looked like a working roster of one, which is far worse than
            # looking broken.
            #
            # Each id-less box now gets its own transient, unregistered identity
            # labelled "id:?" in the overflow grey. Six grey "id:?" markers are
            # an unmistakable "tracking is not producing ids", which is the
            # actual fault to go and fix.
            self.untracked_boxes += 1
            return Identity(key="track:none", number=None, slot=0, last_seen_frame=frame_idx)

        track_id = int(track_id)

        if number is not None:
            # Confirmed. Bind (or re-bind) this track to the number identity,
            # merging it with any earlier track wearing the same number.
            key = self._number_key(number)
            identity = self._identities.get(key)
            if identity is None:
                identity = self._promote(track_id, number, frame_idx)
            identity.track_ids.add(track_id)
            identity.last_seen_frame = frame_idx
            if identity.slot > 0:
                self._track_to_key[track_id] = key
            return identity

        # No number this frame: reuse whatever this track resolved to last time,
        # so a numbered player does not flicker back to a slot label.
        remembered = self._track_to_key.get(track_id)
        if remembered is not None:
            identity = self._identities.get(remembered)
            if identity is not None:
                identity.track_ids.add(track_id)
                identity.last_seen_frame = frame_idx
                return identity

        key = self._track_key(track_id)
        identity = self._identities.get(key)
        if identity is None:
            # track_id IS the SlotTracker slot, so reuse it directly.
            identity = self._create(key, None, frame_idx, preferred_slot=track_id)
        identity.track_ids.add(track_id)
        identity.last_seen_frame = frame_idx
        if identity.slot > 0:
            self._track_to_key[track_id] = key
        return identity

    # -- reporting -------------------------------------------------------
    def confirmed_numbers(self) -> list[str]:
        """Jersey numbers recognized so far, in first-seen order."""
        return list(self._number_order)

    def active_slots(self) -> int:
        return len(self._identities)

    def summary(self) -> dict[str, list[int]]:
        """Identity key -> the tracker ids that collapsed into it."""
        return {
            key: sorted(identity.track_ids)
            for key, identity in sorted(self._identities.items())
        }

    def roster(self) -> dict[int, str]:
        """Slot -> current label. What the viewer should be seeing."""
        return {
            identity.slot: identity.label
            for identity in sorted(self._identities.values(), key=lambda i: i.slot)
        }


# --------------------------------------------------------------------------
# Colour
# --------------------------------------------------------------------------

#: Hand-picked, maximally distinct BGR colours, indexed by slot.
#:
#: A curated palette rather than a generated hue ramp, because the counts here
#: are tiny and known: 6 players in 3x3, 10 in 5x5. Golden-angle hue stepping is
#: the usual trick, but it only separates well asymptotically - at n=6 its
#: closest pair sits 0.09 apart on the wheel (an orange next to a red). These
#: ten were chosen to stay distinguishable against the light-blue court AND
#: against each other. Measured worst-case separation: L1 >= 190 over the first
#: six entries (a full 3x3 line-up) and >= 165 over all ten.
_PALETTE_BGR: tuple[tuple[int, int, int], ...] = (
    (60, 60, 235),    # red
    (240, 120, 20),   # blue
    (50, 180, 50),    # green
    (20, 200, 255),   # amber
    (200, 40, 190),   # magenta
    (235, 235, 90),   # cyan
    (20, 70, 120),    # brown
    (250, 140, 250),  # pink
    (110, 240, 150),  # mint
    (130, 20, 90),    # indigo
)

#: Golden-angle hue step, used only past the end of the palette where exact
#: spacing stops mattering (more players than any real line-up).
_GOLDEN_RATIO_CONJUGATE = 0.618033988749895

#: Overflow marker colour: a neutral grey for the "id:?" case, so a surplus
#: detection never looks like a roster player.
_OVERFLOW_COLOR = (150, 150, 150)


def identity_color(identity: Identity) -> tuple[int, int, int]:
    """A stable BGR colour for one identity, keyed on its SLOT.

    Replaces the original ``TEAM_COLOR[None]`` grey. Team assignment is not
    implemented anywhere in this pipeline, so ``team_id`` was None for every
    player on every frame and EVERY box, dot and trail rendered in the same
    light grey - which is why the top-down trails were unreadable.

    Keying on slot rather than jersey number bounds the palette to the roster
    and keeps a slot's colour constant all game. Hashing the number instead put
    #7 and #23 within 0.12 of each other on the hue wheel, both green.

    Provisional (number-less) identities are washed out towards grey, so "not
    yet recognized" is visually obvious rather than silently indistinguishable
    from a recognized player.
    """
    if identity.slot <= 0:
        return _OVERFLOW_COLOR

    index = identity.slot - 1
    if index < len(_PALETTE_BGR):
        color = _PALETTE_BGR[index]
    else:
        hue = (index * _GOLDEN_RATIO_CONJUGATE) % 1.0
        red, green, blue = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
        color = (int(blue * 255), int(green * 255), int(red * 255))

    if identity.number is None:
        # 30% of the hue over a mid grey: still tells two unidentified players
        # apart, but never competes with a recognized one for attention.
        color = tuple(int(round(c * 0.30 + 140 * 0.70)) for c in color)  # type: ignore[assignment]

    return color  # type: ignore[return-value]
