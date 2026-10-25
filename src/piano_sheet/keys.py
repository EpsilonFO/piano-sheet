"""Key estimation and pitch spelling (C# or Db?)."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from music21 import key as m21key
from music21 import pitch as m21pitch

# Krumhansl-Kessler key profiles, indexed from the tonic.
_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# Tonic spellings that give conventional key signatures (at most 6 accidentals,
# F# major rather than Gb major, etc.).
_MAJOR_TONICS = ["C", "D-", "D", "E-", "E", "F", "F#", "G", "A-", "A", "B-", "B"]
_MINOR_TONICS = ["c", "c#", "d", "e-", "e", "f", "f#", "g", "g#", "a", "b-", "b"]

_LETTER_PCS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

SUPPORTED_KEYS = _MAJOR_TONICS + _MINOR_TONICS

_FRENCH_STEPS = {"C": "Do", "D": "Ré", "E": "Mi", "F": "Fa", "G": "Sol", "A": "La", "B": "Si"}


def estimate_key(notes: Iterable[tuple[int, float]]) -> m21key.Key:
    """Krumhansl-Schmuckler key finding.

    ``notes`` yields ``(midi_pitch, weight)`` pairs, the weight typically being
    the note duration.
    """
    histogram = np.zeros(12)
    for midi_pitch, weight in notes:
        histogram[midi_pitch % 12] += max(weight, 0.0)
    if not histogram.any():
        return m21key.Key("C")

    best_score, best_key = -np.inf, "C"
    for tonic in range(12):
        rotated = np.roll(histogram, -tonic)
        for profile, names in ((_MAJOR_PROFILE, _MAJOR_TONICS), (_MINOR_PROFILE, _MINOR_TONICS)):
            score = np.corrcoef(rotated, profile)[0, 1]
            if score > best_score:
                best_score, best_key = score, names[tonic]
    return m21key.Key(best_key)


def parse_key(name: str) -> m21key.Key:
    """Parse a user-supplied key such as ``"Eb"``, ``"f#"``, ``"Bb minor"``.

    Upper-case tonic means major, lower-case means minor, unless the mode is
    given explicitly.
    """
    text = name.strip()
    mode = None
    lowered = text.lower()
    for suffix, suffix_mode in ((" major", "major"), (" majeur", "major"), (" minor", "minor"),
                                (" mineur", "minor"), ("m", "minor")):
        if lowered.endswith(suffix) and len(text) > len(suffix):
            text, mode = text[: -len(suffix)].strip(), suffix_mode
            break
    if not text or text[0].upper() not in _LETTER_PCS:
        raise ValueError(f"Tonalité inconnue : {name!r}")
    tonic = text[0].upper() + text[1:].replace("b", "-").replace("♭", "-").replace("♯", "#")
    if mode is None:
        mode = "major" if text[0].isupper() else "minor"
    try:
        return m21key.Key(tonic, mode)
    except Exception as error:
        raise ValueError(f"Tonalité inconnue : {name!r}") from error


def describe_key(k: m21key.Key) -> str:
    """French name of a key, e.g. ``"Mi♭ majeur"``."""
    tonic = k.tonic
    name = _FRENCH_STEPS[tonic.step]
    if tonic.accidental is not None:
        name += {"sharp": "♯", "flat": "♭", "double-sharp": "𝄪", "double-flat": "𝄫"}.get(tonic.accidental.name, "")
    return f"{name} {'majeur' if k.mode == 'major' else 'mineur'}"


class PitchSpeller:
    """Chooses note names for MIDI pitches according to a key."""

    def __init__(self, k: m21key.Key) -> None:
        self.key = k
        names: dict[int, tuple[str, int]] = {}
        scale = k.getPitches()[:7]
        for p in scale:
            names[p.pitchClass] = (p.step, int(p.alter))
        if k.mode == "minor":
            # Melodic/harmonic minor: raised 6th and 7th degrees keep their
            # letter (G# in A minor), except where that would take a double
            # sharp (F## in G# minor): a plain G is far easier to read.
            for degree in (scale[5], scale[6]):
                if degree.alter < 1:
                    names.setdefault((degree.pitchClass + 1) % 12, (degree.step, int(degree.alter) + 1))
        sharps = k.sharps
        for pc in range(12):
            if pc in names:
                continue
            natural = next((step for step, value in _LETTER_PCS.items() if value == pc), None)
            if natural is not None:
                names[pc] = (natural, 0)
            elif sharps > 0 or (sharps == 0 and pc in (1, 6, 8)):
                names[pc] = (_step_for((pc - 1) % 12), 1)
            else:
                names[pc] = (_step_for((pc + 1) % 12), -1)
        self._names = names

    def spell(self, midi_pitch: int) -> m21pitch.Pitch:
        step, alter = self._names[midi_pitch % 12]
        octave = (midi_pitch - _LETTER_PCS[step] - alter) // 12 - 1
        p = m21pitch.Pitch(step=step, octave=octave)
        if alter:
            p.accidental = m21pitch.Accidental(alter)
        return p


def _step_for(pc: int) -> str:
    return next(step for step, value in _LETTER_PCS.items() if value == pc)
