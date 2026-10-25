from fractions import Fraction

import numpy as np
import pytest

from piano_sheet.notes import NoteEvent
from piano_sheet.rhythm import BeatGrid, Meter, estimate_beats, estimate_meter, quantize

from conftest import perform


def test_meter_properties():
    assert Meter.parse("4/4").beat_ql == 1
    assert Meter.parse("3/4").bar_ql == 3
    six_eight = Meter.parse("6/8")
    assert six_eight.compound
    assert six_eight.beats_per_bar == 2
    assert six_eight.beat_ql == Fraction(3, 2)
    assert six_eight.bar_ql == 3
    assert str(Meter.from_beats(4, compound=True)) == "12/8"
    with pytest.raises(ValueError):
        Meter.parse("5/4")


def test_beat_grid_round_trip_and_extrapolation():
    grid = BeatGrid([1.0, 1.5, 2.1, 2.6, 3.2])
    seconds = np.array([0.0, 1.0, 1.8, 3.2, 4.0])
    beats = grid.to_beats(seconds)
    assert beats[1] == pytest.approx(0.0)
    assert beats[3] == pytest.approx(4.0)
    assert beats[0] < 0  # Before the first beat: extrapolated.
    assert beats[4] > 4  # After the last beat: extrapolated.
    np.testing.assert_allclose(grid.to_seconds(beats), seconds)


@pytest.mark.parametrize("bpm", [72, 100, 132])
def test_estimate_beats_finds_tempo(bpm):
    # Eighth notes in the right hand, quarter-note bass.
    score = [(i / 2, 0.5, 72 + (i % 5)) for i in range(96)] + [(i, 1, 48 + (i % 4) * 2) for i in range(48)]
    notes = perform(score, bpm=bpm, jitter=0.01, seed=1)
    grid = estimate_beats(notes)
    ratio = grid.bpm / bpm
    # The beat level may legitimately be the eighth note for slow pieces.
    assert min(abs(ratio - 1), abs(ratio - 2), abs(ratio - 0.5)) < 0.04


def test_estimate_beats_uses_given_tempo():
    notes = perform([(i, 1, 60) for i in range(32)], bpm=90)
    assert estimate_beats(notes, bpm=90).bpm == pytest.approx(90, rel=0.03)


def test_estimate_beats_with_few_notes_uses_constant_grid():
    notes = [NoteEvent(0.5, 1.0, 60, 80), NoteEvent(1.1, 1.5, 64, 80)]
    grid = estimate_beats(notes)
    assert grid.bpm == pytest.approx(100)


def test_waltz_is_three_four(waltz):
    notes = perform(waltz, bpm=120, jitter=0.008, seed=3)
    grid = estimate_beats(notes)
    meter, first_bar = estimate_meter(notes, grid)
    assert str(meter) == "3/4"
    # The first note (a downbeat) starts the first bar.
    assert grid.to_beats(notes[0].onset) - first_bar == pytest.approx(0, abs=0.2)


def test_compound_meter_detected_from_ternary_subdivisions():
    score = []
    for beat in range(32):
        score.append((beat, 1.5, 45 if beat % 2 == 0 else 52))  # Dotted-quarter bass.
        score.extend((beat + k / 3, 1 / 3, 69 + k * 3) for k in range(3))
    notes = perform(score, bpm=70, jitter=0.006, seed=2)
    grid = estimate_beats(notes, bpm=70)
    meter, _ = estimate_meter(notes, grid)
    assert meter.compound


def test_quantize_snaps_to_sixteenths():
    pattern = [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.75, 3]
    score = [(bar * 4 + p, 0.25, 60 + i) for bar in range(4) for i, p in enumerate(pattern)]
    notes = perform(score, bpm=80, jitter=0.008, seed=5)
    grid = BeatGrid.constant(80, 0.5, 20)
    performance = quantize(notes, grid, Meter(4, 4), 0)
    starts = sorted({n.start for n in performance.notes})
    expected = sorted({Fraction(bar * 4) + Fraction(p).limit_denominator(4) for bar in range(4) for p in pattern})
    assert starts == expected
    assert performance.tuplet_beats == {}


def test_quantize_detects_triplets_per_beat():
    score = []
    for beat in range(8):
        if beat < 4:
            score.extend((beat + k / 3, 1 / 3, 64 + k) for k in range(3))
        else:
            score.extend((beat + k / 2, 1 / 2, 64 + k) for k in range(2))
    notes = perform(score, bpm=90, jitter=0.006, seed=7)
    grid = BeatGrid.constant(90, 0.5, 10)
    performance = quantize(notes, grid, Meter(4, 4), 0)
    assert {Fraction(1, 3), Fraction(2, 3)} <= {n.start % 1 for n in performance.notes if n.start < 4}
    assert all(n.start % Fraction(1, 2) == 0 for n in performance.notes if n.start >= 4)
    assert set(performance.tuplet_beats) >= {0, 1, 2, 3}
    assert not set(performance.tuplet_beats) & {4, 5, 6, 7}


def test_quantize_never_overlaps_a_repeated_key():
    notes = [NoteEvent(0.5, 1.4, 60, 80), NoteEvent(1.0, 1.5, 60, 80)]
    performance = quantize(notes, BeatGrid.constant(120, 0.5, 3), Meter(4, 4), 0)
    first, second = sorted(performance.notes, key=lambda n: n.start)
    assert first.end <= second.start


def test_slow_held_arpeggios_are_not_counted_at_double_speed():
    # A ballad at 70 BPM: each bar, a chord is arpeggiated in eighth notes and
    # held with the pedal. The arpeggio notes are evenly spaced, so a tracker
    # may take them for beats at 140 BPM.
    chords = [(56, 63, 68, 73), (59, 63, 68, 73), (54, 63, 68, 73), (53, 63, 68, 73)]
    score = []
    for bar in range(16):
        for i, pitch in enumerate(chords[bar % 4]):
            score.append((bar * 4 + i / 2, 4 - i / 2, pitch))
    notes = perform(score, bpm=70, jitter=0.01, seed=8, legato=0.95)
    assert estimate_beats(notes).bpm == pytest.approx(70, rel=0.05)
