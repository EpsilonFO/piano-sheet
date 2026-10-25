"""From seconds to notated rhythm: beat tracking, meter and quantization.

A human performance never follows a metronome exactly, so notes are not
quantized against a fixed tempo. Beats are tracked through the piece instead
(following rubato), every note time is expressed as a fractional beat
position, and those positions are snapped to a rhythmic grid chosen beat by
beat (eighths, sixteenths, triplets...).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

import numpy as np

from .notes import NoteEvent

SUPPORTED_TIME_SIGNATURES = ("2/4", "3/4", "4/4", "6/8", "9/8", "12/8")

# Onset envelope resolution used for beat tracking (librosa's defaults).
_SR, _HOP = 22050, 512
# Notes starting less than this apart are considered played together.
_CHORD_TOLERANCE = 0.035


@dataclass(frozen=True)
class Meter:
    numerator: int
    denominator: int

    @classmethod
    def parse(cls, text: str) -> "Meter":
        if text.strip() not in SUPPORTED_TIME_SIGNATURES:
            raise ValueError(
                f"Mesure non prise en charge : {text!r} (choix : {', '.join(SUPPORTED_TIME_SIGNATURES)})"
            )
        numerator, denominator = text.strip().split("/")
        return cls(int(numerator), int(denominator))

    @classmethod
    def from_beats(cls, beats_per_bar: int, compound: bool) -> "Meter":
        return cls(beats_per_bar * 3, 8) if compound else cls(beats_per_bar, 4)

    @property
    def compound(self) -> bool:
        """6/8, 9/8, 12/8: beats are dotted quarters divided in three."""
        return self.denominator == 8 and self.numerator % 3 == 0

    @property
    def beats_per_bar(self) -> int:
        return self.numerator // 3 if self.compound else self.numerator

    @property
    def beat_ql(self) -> Fraction:
        """Length of one beat in quarter notes."""
        return Fraction(3, 2) if self.compound else Fraction(4, self.denominator)

    @property
    def bar_ql(self) -> Fraction:
        return self.beats_per_bar * self.beat_ql

    def __str__(self) -> str:
        return f"{self.numerator}/{self.denominator}"


class BeatGrid:
    """Maps between seconds and (fractional) beat indices.

    Between tracked beats time is interpolated linearly; outside, it is
    extrapolated with the local beat period.
    """

    def __init__(self, times: Sequence[float]) -> None:
        times = np.asarray(times, dtype=float)
        if len(times) < 2 or np.any(np.diff(times) <= 0):
            raise ValueError("A beat grid needs at least two increasing beat times")
        self.times = times
        intervals = np.diff(times)
        self._head_period = float(np.median(intervals[:4]))
        self._tail_period = float(np.median(intervals[-4:]))

    @classmethod
    def constant(cls, bpm: float, start: float, end: float) -> "BeatGrid":
        period = 60.0 / bpm
        count = max(2, int(math.ceil((end - start) / period)) + 1)
        return cls(start + period * np.arange(count))

    @property
    def bpm(self) -> float:
        return 60.0 / float(np.median(np.diff(self.times)))

    def to_beats(self, seconds: float | np.ndarray) -> np.ndarray:
        t = np.asarray(seconds, dtype=float)
        last = len(self.times) - 1
        inside = np.interp(t, self.times, np.arange(len(self.times), dtype=float))
        before = (t - self.times[0]) / self._head_period
        after = last + (t - self.times[-1]) / self._tail_period
        return np.where(t < self.times[0], before, np.where(t > self.times[-1], after, inside))

    def to_seconds(self, beats: float | np.ndarray) -> np.ndarray:
        b = np.asarray(beats, dtype=float)
        last = len(self.times) - 1
        inside = np.interp(b, np.arange(len(self.times), dtype=float), self.times)
        before = self.times[0] + b * self._head_period
        after = self.times[-1] + (b - last) * self._tail_period
        return np.where(b < 0, before, np.where(b > last, after, inside))


# ---------------------------------------------------------------------------
# Beat tracking


def _onset_envelope(notes: Sequence[NoteEvent]) -> np.ndarray:
    from scipy.ndimage import gaussian_filter1d

    fps = _SR / _HOP
    end = max(n.offset for n in notes)
    envelope = np.zeros(int(end * fps) + 8)
    for note in notes:
        envelope[int(round(note.onset * fps))] += 0.3 + note.velocity / 127
    return gaussian_filter1d(envelope, 1.0)


def _accents(envelope: np.ndarray, beat_times: np.ndarray) -> np.ndarray:
    fps = _SR / _HOP
    frames = np.clip(np.round(beat_times * fps).astype(int), 0, len(envelope) - 1)
    return np.array([envelope[max(0, f - 2): f + 3].max() for f in frames])


def estimate_beats(notes: Sequence[NoteEvent], bpm: float | None = None) -> BeatGrid:
    """Track beats through the performance.

    ``bpm``, when given, is used as the tempo instead of being estimated.
    Without it the beat level is kept between 50 and 160 BPM, the range in
    which a quarter (or dotted quarter) beat is most readable.
    """
    if not notes:
        return BeatGrid.constant(bpm or 100.0, 0.0, 1.0)
    first = min(n.onset for n in notes)
    last = max(n.offset for n in notes)
    distinct_onsets = len({round(n.onset / _CHORD_TOLERANCE) for n in notes})
    if distinct_onsets < 6:
        return BeatGrid.constant(bpm or 100.0, first, last)

    import librosa

    envelope = _onset_envelope(notes)
    # A looser tightness than librosa's default (100) lets the beats follow
    # the rubato of real performances.
    kwargs = dict(onset_envelope=envelope, sr=_SR, hop_length=_HOP, units="time", trim=False, tightness=50)
    if bpm:
        _, beats = librosa.beat.beat_track(bpm=float(bpm), **kwargs)
    else:
        _, beats = librosa.beat.beat_track(start_bpm=110.0, **kwargs)
    beats = np.unique(np.asarray(beats, dtype=float))
    if len(beats) < 4:
        return BeatGrid.constant(bpm or 100.0, first, last)

    if not bpm:
        beats = _normalize_tempo(beats, envelope)
    return BeatGrid(beats)


def _normalize_tempo(beats: np.ndarray, envelope: np.ndarray) -> np.ndarray:
    for _ in range(3):
        tempo = 60.0 / float(np.median(np.diff(beats)))
        if tempo > 160 and len(beats) >= 8:
            # Too fast: the tracker locked onto subdivisions. Group them by two
            # or three, whichever lines up best with the accents.
            accents = _accents(envelope, beats)
            best = None
            for group, prior in ((2, 1.0), (3, 0.92)):
                for phase in range(group):
                    contrast = accents[phase::group].mean() / (accents.mean() + 1e-9) * prior
                    if best is None or contrast > best[0]:
                        best = (contrast, group, phase)
            _, group, phase = best
            beats = beats[phase::group]
        elif tempo < 50:
            midpoints = (beats[:-1] + beats[1:]) / 2
            beats = np.sort(np.concatenate([beats, midpoints]))
        else:
            break
    return beats


# ---------------------------------------------------------------------------
# Meter


@dataclass(frozen=True)
class _Cluster:
    """Notes struck together."""

    position: float  # beat position
    notes: tuple[int, ...]  # indices into the note list


def _clusters(notes: Sequence[NoteEvent], positions: np.ndarray) -> list[_Cluster]:
    order = sorted(range(len(notes)), key=lambda i: notes[i].onset)
    clusters: list[list[int]] = []
    for i in order:
        if clusters and notes[i].onset - notes[clusters[-1][0]].onset < _CHORD_TOLERANCE:
            clusters[-1].append(i)
        else:
            clusters.append([i])
    return [_Cluster(float(np.median(positions[c])), tuple(c)) for c in clusters]


def _is_ternary(clusters: list[_Cluster]) -> bool:
    duple_votes = ternary_votes = 0
    for cluster in clusters:
        frac = cluster.position % 1.0
        if min(frac, 1 - frac) < 0.1:
            continue  # On the beat: says nothing about the subdivision.
        to_duple = abs(frac * 4 - round(frac * 4)) / 4
        to_ternary = abs(frac * 3 - round(frac * 3)) / 3
        if to_ternary + 0.02 < to_duple:
            ternary_votes += 1
        elif to_duple + 0.02 < to_ternary:
            duple_votes += 1
    return ternary_votes >= 8 and ternary_votes > 1.5 * duple_votes


def _beat_features(notes: Sequence[NoteEvent], grid: BeatGrid, first_beat: int, last_beat: int) -> np.ndarray:
    """Describe the texture of each beat (z-scored): how many notes, how low, how long, how loud."""
    onsets = grid.to_beats([n.onset for n in notes])
    lengths = grid.to_beats([n.offset for n in notes]) - onsets
    rows = []
    for beat in range(first_beat, last_beat + 1):
        in_beat = (onsets >= beat - 0.2) & (onsets < beat + 0.8)
        on_beat = np.flatnonzero((onsets >= beat - 0.2) & (onsets < beat + 0.2))
        rows.append([
            in_beat.sum(),
            len(on_beat),
            in_beat.sum() - len(on_beat),
            min((notes[i].pitch for i in on_beat), default=90),
            max((min(lengths[i], 3.0) for i in on_beat), default=0.0),
            sum(notes[i].velocity for i in on_beat) / 127,
        ])
    features = np.array(rows, dtype=float)
    return (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-9)


def _bar_similarity(features: np.ndarray, beats_per_bar: int) -> float:
    """Mean distance between beats one or more bars apart (lower: more alike)."""
    lags = [lag for lag in range(beats_per_bar, 9, beats_per_bar) if lag < len(features)]
    if not lags:
        return 0.0
    return float(np.mean([np.abs(features[lag:] - features[:-lag]).sum(axis=1).mean() for lag in lags]))


def estimate_meter(
    notes: Sequence[NoteEvent],
    grid: BeatGrid,
    meter: Meter | None = None,
) -> tuple[Meter, int]:
    """Guess the time signature and where bar lines fall.

    Returns the meter and the beat index (in ``grid`` coordinates) of the
    first bar line, which is at or just before the first note. When ``meter``
    is given only the bar line position is estimated.
    """
    if not notes:
        return meter or Meter(4, 4), 0
    positions = grid.to_beats([n.onset for n in notes])
    clusters = _clusters(notes, positions)

    first_beat = int(math.floor(positions.min() + 0.25))
    last_beat = int(math.floor(positions.max() + 0.5))
    # Two accent profiles per beat. The bar length shows in how much is played
    # (loud, thick, long notes recur once per bar), but the downbeat is better
    # told by the bass: in an oom-pah-pah the "pahs" are the thicker chords.
    weight = np.zeros(last_beat - first_beat + 1)
    bass = np.zeros_like(weight)
    lowest = [min(notes[i].pitch for i in cluster.notes) for cluster in clusters]
    cluster_positions = np.array([cluster.position for cluster in clusters])
    for index, cluster in enumerate(clusters):
        nearest = round(cluster.position)
        if abs(cluster.position - nearest) > 0.15 or not first_beat <= nearest <= last_beat:
            continue
        members = [notes[i] for i in cluster.notes]
        longest = max(float(grid.to_beats(n.offset) - grid.to_beats(n.onset)) for n in members)
        beat = nearest - first_beat
        weight[beat] += sum(0.2 + n.velocity / 127 for n in members) + 0.3 * min(longest, 2.0)
        if lowest[index] < 55:
            weight[beat] += 0.5
        # The lowest note around, held: the bass moving on a strong beat.
        neighbours = np.flatnonzero(np.abs(cluster_positions - cluster.position) <= 1.0)
        if lowest[index] <= min(lowest[j] for j in neighbours):
            bass[beat] += 1.0 + 0.5 * min(longest, 2.0)

    if meter is not None:
        candidates = [(meter.beats_per_bar, 0.0)]
        compound = meter.compound
    else:
        compound = _is_ternary(clusters)
        # Priors favour the most common meters (4/4, then 3/4; 6/8 for compound).
        candidates = [(2, 0.03), (4, 0.0), (3, -0.03)] if compound else [(4, 0.03), (3, 0.0), (2, -0.03)]

    def contrast(accents: np.ndarray, beats_per_bar: int, phase: int) -> float:
        selected = accents[phase::beats_per_bar]
        return float(selected.mean() / (accents.mean() + 1e-9)) if len(selected) else 0.0

    # Bar length: beats a bar apart look alike (self-similarity), and some
    # beat of the bar stands out (accent contrast). Neither cue is reliable
    # on its own, so both are added up.
    features = _beat_features(notes, grid, first_beat, last_beat)
    similarity = {m: _bar_similarity(features, m) for m, _ in candidates}
    mean_similarity = np.mean(list(similarity.values())) + 1e-9

    def evidence(beats_per_bar: int, prior: float) -> float:
        accent = max(contrast(weight, beats_per_bar, phase) for phase in range(beats_per_bar))
        return (1 - similarity[beats_per_bar] / mean_similarity) + (accent - 1) + prior

    opening = int(round(positions.min())) - first_beat
    beats_per_bar = max(candidates, key=lambda candidate: evidence(*candidate))[0]
    combined = weight / (weight.mean() + 1e-9) + 2 * bass / (bass.mean() + 1e-9)
    phase = max(
        range(beats_per_bar),
        # Pieces usually start on a downbeat.
        key=lambda p: contrast(combined, beats_per_bar, p) * (1.05 if opening % beats_per_bar == p else 1.0),
    )
    chosen = meter or Meter.from_beats(beats_per_bar, compound)
    downbeat = first_beat + phase
    # First bar line at or before the first note (allowing a slightly early
    # first note to belong to the next bar).
    first_position = float(positions.min())
    start = downbeat - beats_per_bar * math.ceil((downbeat - first_position - 0.25) / beats_per_bar)
    return chosen, int(start)


def chord_positions(notes: Sequence[NoteEvent], grid: BeatGrid, first_bar_beat: int) -> list[tuple[float, float]]:
    """``(start, end)`` of each note in beats from the first bar line.

    Notes struck together (within a few milliseconds) get exactly the same
    start, so that they can be grouped before quantization.
    """
    onsets = grid.to_beats([n.onset for n in notes]) - first_bar_beat
    ends = grid.to_beats([n.offset for n in notes]) - first_bar_beat
    starts = np.empty(len(notes))
    for cluster in _clusters(notes, onsets):
        starts[list(cluster.notes)] = cluster.position
    return [(float(s), float(e)) for s, e in zip(starts, ends)]


# ---------------------------------------------------------------------------
# Quantization


@dataclass(frozen=True)
class QuantizedNote:
    """A note on the rhythmic grid, in beats from the first bar line."""

    start: Fraction
    end: Fraction
    pitch: int
    velocity: int


@dataclass
class QuantizedPerformance:
    notes: list[QuantizedNote]
    meter: Meter
    # Beats (from the first bar line) divided in triplets, mapped to their
    # subdivision (3: triplet eighths, 6: triplet sixteenths). Only used in
    # simple meters; compound meters are ternary by nature.
    tuplet_beats: dict[int, int]


# (subdivisions per beat, cost per note, family). A finer grid always fits
# better, so each one has a cost that must be earned by a better fit.
_SIMPLE_GRIDS = (
    (2, 0.0, "duple"), (4, 0.03, "duple"), (8, 0.09, "duple"),
    (3, 0.04, "triplet"), (6, 0.08, "triplet"),
)
_COMPOUND_GRIDS = ((3, 0.0, "duple"), (6, 0.03, "duple"), (12, 0.09, "duple"))
_COLLISION_COST = 0.3
_AMBIGUITY = 0.06


def _grid_cost(fracs: np.ndarray, division: int, per_note: float) -> float:
    snapped = np.round(fracs * division)
    error = float(np.abs(fracs - snapped / division).sum())
    collisions = len(snapped) - len(np.unique(snapped))
    return error + per_note * len(fracs) + _COLLISION_COST * collisions


def quantize(
    notes: Sequence[NoteEvent],
    grid: BeatGrid,
    meter: Meter,
    first_bar_beat: int,
) -> QuantizedPerformance:
    """Snap notes to a rhythmic grid chosen independently for each beat."""
    if not notes:
        return QuantizedPerformance([], meter, {})
    grids = _COMPOUND_GRIDS if meter.compound else _SIMPLE_GRIDS
    coarse = {"duple": grids[0][0], "triplet": 3}

    onsets = grid.to_beats([n.onset for n in notes]) - first_bar_beat
    offsets = grid.to_beats([n.offset for n in notes]) - first_bar_beat
    clusters = _clusters(notes, onsets)

    # Group clusters by beat; a cluster slightly before a beat belongs to it.
    by_beat: dict[int, list[_Cluster]] = {}
    for cluster in clusters:
        by_beat.setdefault(int(math.floor(cluster.position + 0.06)), []).append(cluster)

    # Best grid of each family for each beat.
    choices: dict[int, dict[str, tuple[float, int]]] = {}
    for beat, members in by_beat.items():
        fracs = np.array([c.position - beat for c in members])
        options: dict[str, tuple[float, int]] = {}
        for division, per_note, family in grids:
            cost = _grid_cost(fracs, division, per_note)
            if family not in options or cost < options[family][0]:
                options[family] = (cost, division)
        choices[beat] = options

    # Beats that fit both families about as well follow their neighbours.
    decided: dict[int, str] = {}
    for beat, options in choices.items():
        if len(options) == 1:
            decided[beat] = next(iter(options))
        elif abs(options["duple"][0] - options["triplet"][0]) >= _AMBIGUITY:
            decided[beat] = min(options, key=lambda f: options[f][0])

    def family_of(beat: int) -> str:
        if beat in decided:
            return decided[beat]
        votes = [decided[b] for b in range(beat - 4, beat + 5) if b in decided]
        return "triplet" if votes.count("triplet") > votes.count("duple") else "duple"

    division_of: dict[int, int] = {}
    for beat, options in choices.items():
        family = family_of(beat)
        division_of[beat] = options[family][1] if family in options else coarse[family]

    starts: dict[int, Fraction] = {}
    for beat, members in by_beat.items():
        division = division_of[beat]
        for cluster in members:
            step = round((cluster.position - beat) * division)
            start = max(Fraction(0), beat + Fraction(step, division))
            for index in cluster.notes:
                starts[index] = start

    quantized: dict[tuple[int, Fraction], QuantizedNote] = {}
    for index, note in enumerate(notes):
        start = starts[index]
        end_position = float(offsets[index])
        end_beat = int(math.floor(end_position))
        if Fraction(end_beat) <= start:
            # Ends within the beat it starts in: keep that beat's resolution.
            division = division_of.get(end_beat, coarse[family_of(end_beat)])
        else:
            # Releases are less precise than attacks: use a coarse grid.
            division = coarse[family_of(end_beat)]
        end = end_beat + Fraction(round((end_position - end_beat) * division), division)
        if end <= start:
            start_beat = int(math.floor(start))
            end = start + Fraction(1, division_of.get(start_beat, coarse[family_of(start_beat)]))
        key = (note.pitch, start)
        previous = quantized.get(key)
        if previous is None or end > previous.end:
            velocity = max(note.velocity, previous.velocity) if previous else note.velocity
            quantized[key] = QuantizedNote(start, end, note.pitch, velocity)

    # A key cannot be struck again before being released.
    result = sorted(quantized.values(), key=lambda n: (n.pitch, n.start))
    for i in range(len(result) - 1):
        current, following = result[i], result[i + 1]
        if current.pitch == following.pitch and current.end > following.start:
            result[i] = QuantizedNote(current.start, following.start, current.pitch, current.velocity)
    result.sort(key=lambda n: (n.start, n.pitch))

    tuplet_beats: dict[int, int] = {}
    if not meter.compound:
        last = int(math.ceil(max(n.end for n in result)))
        tuplet_beats = {
            b: division_of.get(b, coarse["triplet"]) for b in range(last + 1) if family_of(b) == "triplet"
        }
    return QuantizedPerformance(result, meter, tuplet_beats)
