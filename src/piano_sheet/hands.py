"""Splitting notes between the right hand (treble staff) and left hand (bass staff)."""

from __future__ import annotations

from collections import deque
from itertools import groupby
from typing import NamedTuple, Sequence

LEFT, RIGHT = 0, 1


class TimedPitch(NamedTuple):
    """A note placed in beats; notes struck together share the same ``start``."""

    start: float
    end: float
    pitch: int

_MAX_SPAN = 14  # Widest stretch for one hand, in semitones.
_MAX_NOTES = 5
# A hand's position is given by the notes of its last few attacks...
_ANCHOR_GROUPS = 3
# ...unless it has been silent for this many beats, in which case its
# average position is used instead.
_ANCHOR_MEMORY = 4


def _span_cost(pitches: Sequence[int]) -> float:
    span = pitches[-1] - pitches[0]
    # Stretches beyond a fifth get gradually less comfortable; beyond an
    # octave they are rare, beyond a ninth practically impossible.
    if span > _MAX_SPAN:
        return 10.0 + (span - _MAX_SPAN)
    if span > 12:
        return 0.75 + 0.6 * (span - 12)
    if span > 7:
        return 0.15 * (span - 7)
    return 0.0


def _hand_cost(pitches: Sequence[int], anchors: Sequence[float], center: float, hand: int) -> float:
    if not pitches:
        return 0.0
    cost = _span_cost(pitches)
    if len(pitches) > _MAX_NOTES:
        cost += 10.0
    for p in pitches:
        # Continuity: hands move little between consecutive notes.
        cost += min(3.0, min(abs(p - a) for a in anchors) / 4)
        cost += abs(p - center) / 12
    # Each hand prefers its own side of the keyboard.
    if hand == LEFT:
        cost += sum(max(0, p - 64) for p in pitches) * 0.15
    else:
        cost += sum(max(0, 57 - p) for p in pitches) * 0.15
    return cost


def assign_hands(notes: Sequence[TimedPitch]) -> list[int]:
    """Return ``LEFT`` or ``RIGHT`` for each note.

    Notes struck together are split at the pitch where both hand positions
    are most plausible, given hand stretches and the notes each hand played
    just before (so that a melody dipping below middle C stays in the right
    hand while the left hand keeps its accompaniment).
    """
    hands = [RIGHT] * len(notes)
    centers = {LEFT: 50.0, RIGHT: 71.0}
    anchors = {LEFT: deque(maxlen=_ANCHOR_GROUPS), RIGHT: deque(maxlen=_ANCHOR_GROUPS)}
    last_end = {LEFT: None, RIGHT: None}
    order = sorted(range(len(notes)), key=lambda i: (notes[i].start, notes[i].pitch))
    for start, group in groupby(order, key=lambda i: notes[i].start):
        indices = list(group)
        pitches = [notes[i].pitch for i in indices]
        current = {}
        for hand in (LEFT, RIGHT):
            recent = last_end[hand] is not None and start - last_end[hand] <= _ANCHOR_MEMORY
            current[hand] = [p for attack in anchors[hand] for p in attack] if recent else [centers[hand]]

        best_cost, best_split = None, 0
        for split in range(len(pitches) + 1):
            left, right = pitches[:split], pitches[split:]
            cost = (_hand_cost(left, current[LEFT], centers[LEFT], LEFT)
                    + _hand_cost(right, current[RIGHT], centers[RIGHT], RIGHT))
            if best_cost is None or cost < best_cost:
                best_cost, best_split = cost, split

        for position, index in enumerate(indices):
            hands[index] = LEFT if position < best_split else RIGHT
        for hand, chosen in ((LEFT, pitches[:best_split]), (RIGHT, pitches[best_split:])):
            if not chosen:
                continue
            anchors[hand].append(tuple(chosen))
            last_end[hand] = max(notes[i].end for i in indices if hands[i] == hand)
            centers[hand] = 0.8 * centers[hand] + 0.2 * (sum(chosen) / len(chosen))
        # Keep the hands from drifting onto each other.
        if centers[RIGHT] - centers[LEFT] < 10:
            middle = (centers[RIGHT] + centers[LEFT]) / 2
            centers[LEFT], centers[RIGHT] = middle - 5, middle + 5
    return hands
