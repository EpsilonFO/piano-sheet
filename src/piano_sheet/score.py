"""Engraving: quantized notes -> a two-staff piano score (music21)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from itertools import groupby
from typing import Mapping, Sequence

from music21 import (
    chord,
    clef,
    duration,
    dynamics,
    instrument,
    layout,
    metadata,
    meter,
    note,
    spanner,
    stream,
    tempo,
    tie,
)
from music21 import key as m21key

from .hands import LEFT, RIGHT
from .keys import PitchSpeller
from .rhythm import Meter, QuantizedNote, QuantizedPerformance

# Single-symbol durations (in quarter notes), longest first: whole, dotted
# half, half, dotted quarter, quarter, dotted eighth, eighth, dotted
# sixteenth, sixteenth, thirty-second.
_PLAIN_LENGTHS = tuple(Fraction(x) for x in ("4", "3", "2", "3/2", "1", "3/4", "1/2", "3/8", "1/4", "1/8"))

# A silence between a note's release and the next attack in the same hand is
# not written as a rest when it is shorter than this (in beats)...
_LEGATO_GAP = Fraction(1, 4)
# ...or when the note lasted at least half the time to that next attack and
# the silence is at most a beat: detached chords ("oom-pah-pah") are written
# as quarter notes, not as eighth notes followed by eighth rests.
_DETACHED_GAP = Fraction(1)


@dataclass
class ScoreOptions:
    title: str = "Partition"
    composer: str | None = None
    dynamics: bool = True
    clef_changes: bool = True


@dataclass(frozen=True)
class _Event:
    """A chord (or a rest when ``pitches`` is empty), in beats."""

    start: Fraction
    end: Fraction
    pitches: tuple[int, ...]
    velocity: int = 0


def _staff_events(notes: Sequence[QuantizedNote], total: Fraction) -> list[_Event]:
    """Turn a hand's notes into a gapless sequence of chords and rests.

    Notes struck together become a chord lasting until the next attack in
    the same hand (or the release, if that comes clearly earlier), which
    keeps each staff to a single, readable voice.
    """
    groups = [(start, list(group)) for start, group in groupby(sorted(notes, key=lambda n: n.start), key=lambda n: n.start)]
    events: list[_Event] = []
    cursor = Fraction(0)
    for i, (start, group) in enumerate(groups):
        end = max(n.end for n in group)
        if i + 1 < len(groups):
            following = groups[i + 1][0]
            gap = following - end
            if gap <= _LEGATO_GAP or (gap <= _DETACHED_GAP and gap * 2 <= following - start):
                end = following
        if start > cursor:
            events.append(_Event(cursor, start, ()))
        pitches = tuple(sorted({n.pitch for n in group}))
        events.append(_Event(start, end, pitches, max(n.velocity for n in group)))
        cursor = end
    if cursor < total:
        events.append(_Event(cursor, total, ()))
    return events


def _largest_plain(length: Fraction) -> Fraction:
    for candidate in _PLAIN_LENGTHS:
        if candidate <= length:
            return candidate
    return length


def _split_plain(start: Fraction, end: Fraction, bar_start: Fraction, m: Meter) -> list[tuple[Fraction, Fraction]]:
    """Split a duple span inside one bar into conventionally notated pieces."""
    beat = m.beat_ql
    pieces = []
    while start < end:
        remaining = end - start
        position = start - bar_start
        if position % beat == 0:
            length = _largest_plain(remaining)
        else:
            next_beat = bar_start + math.ceil(position / beat) * beat
            syncopation = (
                not m.compound
                and position % (beat / 2) == 0
                and remaining in (beat, beat * 3 / 2)
            )
            if end <= next_beat or syncopation:
                length = _largest_plain(remaining) if not syncopation else remaining
            else:
                # Off the beat: show where the next beat falls with a tie.
                length = _largest_plain(next_beat - start)
        pieces.append((start, length))
        start += length
    return pieces


def _split_tuplet(start: Fraction, end: Fraction, beat_start: Fraction, beat: Fraction) -> list[tuple[Fraction, Fraction]]:
    """Split a span inside a triplet beat into triplet values.

    Uses triplet quarters and eighths (thirds of the beat) where aligned,
    triplet sixteenths (sixths of the beat) otherwise.
    """
    third, sixth = beat / 3, beat / 6
    pieces = []
    while start < end:
        position, remaining = start - beat_start, end - start
        if position % third == 0 and remaining >= 2 * third:
            length = 2 * third
        elif position % third == 0 and remaining >= third:
            length = third
        elif remaining >= sixth and position % sixth == 0:
            length = sixth
        else:
            length = remaining  # Off the triplet grid: cannot do better.
        pieces.append((start, length))
        start += length
    return pieces


def _split_span(start: Fraction, end: Fraction, m: Meter, tuplet_beats: dict[int, int]) -> list[tuple[Fraction, Fraction]]:
    """Split ``[start, end)`` (quarter notes) into notatable pieces.

    Pieces never cross a bar line, and the part of a span inside a triplet
    beat that does not fill the whole beat is written with triplet values.
    """
    beat, bar = m.beat_ql, m.bar_ql
    cuts = {start, end}
    for k in range(math.ceil(start / bar), math.floor(end / bar) + 1):
        if start < k * bar < end:
            cuts.add(k * bar)
    for k in range(math.floor(start / beat), math.ceil(end / beat)):
        if k not in tuplet_beats:
            continue
        beat_start, beat_end = k * beat, (k + 1) * beat
        if beat_start < start < beat_end:
            cuts.add(beat_end)
        if beat_start < end < beat_end:
            cuts.add(beat_start)
    bounds = sorted(c for c in cuts if start <= c <= end)
    pieces: list[tuple[Fraction, Fraction]] = []
    for a, b in zip(bounds, bounds[1:]):
        k = math.floor(a / beat)
        in_tuplet_beat = k in tuplet_beats and b <= (k + 1) * beat
        if in_tuplet_beat and not (a % beat == 0 and b % beat == 0):
            pieces.extend(_split_tuplet(a, b, k * beat, beat))
        else:
            pieces.extend(_split_plain(a, b, math.floor(a / bar) * bar, m))
    return pieces


def _dynamic_marks(events: Sequence[_Event], m: Meter, bars: int) -> dict[int, str]:
    """Dynamics from note velocities, written only where the level changes."""
    sums, counts = [0.0] * bars, [0] * bars
    for event in events:
        if event.pitches:
            index = min(bars - 1, int(event.start // m.beats_per_bar))
            sums[index] += event.velocity * len(event.pitches)
            counts[index] += len(event.pitches)
    levels: list[str | None] = []
    for i in range(bars):
        window = range(max(0, i - 1), min(bars, i + 2))
        total, count = sum(sums[j] for j in window), sum(counts[j] for j in window)
        levels.append(_velocity_to_dynamic(total / count) if counts[i] and count else None)
    marks: dict[int, str] = {}
    current = None
    for i, level in enumerate(levels):
        if level is None or level == current:
            continue
        # Require the new level to hold for two bars to avoid flickering marks.
        if current is not None and i + 1 < bars and levels[i + 1] not in (level, None):
            continue
        marks[i] = level
        current = level
    return marks


def _velocity_to_dynamic(velocity: float) -> str:
    for threshold, mark in ((30, "pp"), (45, "p"), (60, "mp"), (75, "mf"), (90, "f")):
        if velocity < threshold:
            return mark
    return "ff"


def _clef_plan(events: Sequence[_Event], m: Meter, bars: int, hand: int) -> dict[int, clef.Clef]:
    """Switch clefs for passages where a hand goes far from its usual range."""
    pitches_by_bar: list[list[int]] = [[] for _ in range(bars)]
    for event in events:
        if event.pitches:
            pitches_by_bar[min(bars - 1, int(event.start // m.beats_per_bar))].extend(event.pitches)
    current = "treble" if hand == RIGHT else "bass"
    changes: dict[int, clef.Clef] = {}
    for bar, pitches in enumerate(pitches_by_bar):
        if not pitches:
            continue
        mean, low, high = sum(pitches) / len(pitches), min(pitches), max(pitches)
        wanted = current
        if current == "bass" and mean >= 65 and low >= 55:
            wanted = "treble"
        elif current == "treble" and mean <= 54 and high <= 64:
            wanted = "bass"
        elif current == "treble" and hand == LEFT and (mean < 60 or low < 52):
            wanted = "bass"
        elif current == "bass" and hand == RIGHT and (mean > 58 or high > 67):
            wanted = "treble"
        if wanted != current:
            changes[bar] = clef.TrebleClef() if wanted == "treble" else clef.BassClef()
            current = wanted
    return changes


def _ottava_runs(events: Sequence[_Event]) -> set[int]:
    """Indices of right-hand events to write an octave lower under an "8va" line.

    Used for passages that stay high above the staff, which would otherwise
    need a forest of ledger lines.
    """
    selected: set[int] = set()
    run: list[int] = []

    def close() -> None:
        notes = [i for i in run if events[i].pitches]
        if len(notes) >= 3 and sum(max(events[i].pitches) for i in notes) / len(notes) >= 86:
            selected.update(run)
        run.clear()

    for index, event in enumerate(events):
        if not event.pitches:
            if run and event.end - event.start <= 2:
                run.append(index)  # Short rests do not interrupt the passage.
            else:
                close()
        elif min(event.pitches) >= 79:
            run.append(index)
        else:
            close()
    close()
    # Do not start or end the line on a rest.
    for edge in (min, max):
        while selected and not events[edge(selected)].pitches:
            selected.remove(edge(selected))
    return selected


def build_score(
    staves: Mapping[int, QuantizedPerformance],
    m: Meter,
    k: m21key.Key,
    bpm: float,
    options: ScoreOptions | None = None,
) -> stream.Score:
    """Engrave a grand staff from each hand's quantized notes (``RIGHT``/``LEFT``)."""
    options = options or ScoreOptions()
    speller = PitchSpeller(k)

    last_end = max((n.end for staff in staves.values() for n in staff.notes), default=Fraction(0))
    bars = max(1, math.ceil(last_end / m.beats_per_bar))
    total = Fraction(bars * m.beats_per_bar)

    staff_events = {hand: _staff_events(staves[hand].notes, total) for hand in (RIGHT, LEFT)}

    score = stream.Score()
    score.metadata = metadata.Metadata()
    # Only the movement title: setting the work title too makes renderers
    # print the name twice (as title and subtitle).
    score.metadata.movementName = options.title
    score.metadata.composer = options.composer or "Transcription automatique"

    parts = {}
    for hand in (RIGHT, LEFT):
        part = stream.PartStaff(id="RH" if hand == RIGHT else "LH")
        part.partName, part.partAbbreviation = "Piano", "Pno."
        if hand == RIGHT:
            part.insert(0, instrument.Piano())
            part.insert(0, tempo.MetronomeMark(number=round(bpm), referent=duration.Duration(m.beat_ql)))
        part.insert(0, clef.TrebleClef() if hand == RIGHT else clef.BassClef())
        part.insert(0, m21key.KeySignature(k.sharps))
        part.insert(0, meter.TimeSignature(str(m)))

        if options.clef_changes:
            for bar, new_clef in _clef_plan(staff_events[hand], m, bars, hand).items():
                if bar > 0:
                    part.insert(bar * m.bar_ql, new_clef)

        ottava_events = _ottava_runs(staff_events[hand]) if hand == RIGHT else set()
        ottava = None
        for event_index, event in enumerate(staff_events[hand]):
            in_ottava = event_index in ottava_events
            if not in_ottava:
                ottava = None
            elif ottava is None:
                ottava = spanner.Ottava(type="8va", transposing=True)
                part.insert(0, ottava)
            pieces = _split_span(event.start * m.beat_ql, event.end * m.beat_ql, m, staves[hand].tuplet_beats)
            for index, (offset, length) in enumerate(pieces):
                # Under an 8va line notes are written an octave below their pitch.
                pitches = [speller.spell(p - 12 if in_ottava else p) for p in event.pitches]
                if not pitches:
                    element = note.Rest()
                elif len(pitches) == 1:
                    element = note.Note(pitches[0])
                else:
                    element = chord.Chord(pitches)
                if ottava is not None:
                    ottava.addSpannedElements(element)
                element.duration = duration.Duration(length)
                if event.pitches:
                    element.volume.velocity = max(1, min(127, event.velocity))
                    if len(pieces) > 1:
                        kind = "start" if index == 0 else "stop" if index == len(pieces) - 1 else "continue"
                        element.tie = tie.Tie(kind)
                part.insert(offset, element)
        parts[hand] = part

    if options.dynamics:
        all_events = staff_events[RIGHT] + staff_events[LEFT]
        for bar, mark in _dynamic_marks(all_events, m, bars).items():
            parts[RIGHT].insert(bar * m.bar_ql, dynamics.Dynamic(mark))

    score.insert(0, parts[RIGHT])
    score.insert(0, parts[LEFT])
    score.insert(0, layout.StaffGroup([parts[RIGHT], parts[LEFT]], name="Piano", abbreviation="Pno.", symbol="brace"))
    score.makeNotation(inPlace=True)
    return score
