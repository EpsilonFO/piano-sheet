import random

import pytest

from piano_sheet.notes import NoteEvent


def perform(score, bpm=100.0, jitter=0.0, seed=0, start=0.5, legato=0.9, velocity=70):
    """Turn ``(beat, length_in_beats, pitch)`` triples into a performance in seconds."""
    rng = random.Random(seed)
    seconds_per_beat = 60.0 / bpm
    notes = []
    for beat, length, pitch in score:
        onset = start + beat * seconds_per_beat + rng.gauss(0, jitter)
        offset = start + (beat + length * legato) * seconds_per_beat + rng.gauss(0, jitter)
        notes.append(NoteEvent(max(0.0, onset), max(onset + 0.03, offset), pitch, velocity))
    return notes


@pytest.fixture
def waltz():
    """16 bars of 3/4: bass on the downbeat, chords on beats 2 and 3, a melody on top."""
    score = []
    melody = [76, 79, 77, 74, 72, 76, 74, 71]
    for bar in range(16):
        beat = bar * 3
        score.append((beat, 1, 48 if bar % 2 == 0 else 43))
        for chord_beat in (1, 2):
            score.extend((beat + chord_beat, 1, p) for p in (55, 60, 64))
        score.append((beat, 2, melody[bar % len(melody)]))
        score.append((beat + 2, 1, melody[(bar + 1) % len(melody)] - 2))
    return score
