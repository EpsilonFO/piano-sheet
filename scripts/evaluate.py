"""Benchmark: known scores -> synthesized MP3 -> piano-sheet -> comparison.

Renders pieces from the music21 corpus with FluidSynth (optionally with a
humanized performance: rubato, timing jitter, varying dynamics), runs the full
pipeline on the resulting MP3 and reports how close the produced score is to
the original.

Requires ``fluidsynth``, a General MIDI SoundFont and ``ffmpeg``:

    python scripts/evaluate.py --soundfont /usr/share/sounds/sf2/FluidR3_GM.sf2
"""

from __future__ import annotations

import argparse
import random
import subprocess
import sys
import tempfile
from collections import Counter
from fractions import Fraction
from pathlib import Path

from music21 import chord, converter, corpus, note, stream

from piano_sheet.notes import NoteEvent, Transcription, read_midi, write_midi
from piano_sheet.pipeline import Options, arrange, run, write_outputs
from piano_sheet.rhythm import Meter

PIECES = {
    # name: (corpus path, measures, quarter-note BPM, piano parts (right, left), expected meter and key)
    "bach-prelude-c": ("bach/bwv846", (1, 12), 72, (0, 1), "4/4", "Do majeur"),
    "chopin-mazurka-6-2": ("chopin/mazurka06-2.krn", (1, 24), 138, (0, 1), "3/4", "Do♯ mineur"),
    "schubert-lindenbaum": ("schubert/Lindenbaum.xml", (1, 24), 84, (1, 2), "3/4", "Mi majeur"),
    "schumann-dichterliebe-2": ("schumann_robert/dichterliebe_no2.xml", (1, 16), 66, (1, 2), "2/4", "La majeur"),
}


def performed_notes(parts: list[stream.Part], bpm: float, humanize: bool, seed: int):
    """Notes of ``parts`` as a performance: ``(event, notated offset, part index)``."""
    rng = random.Random(seed)
    seconds_per_quarter = 60.0 / bpm
    raw = []
    # A pickup bar is stored shorter than a full bar: shift everything so that
    # offsets count from a (virtual) full first bar and bar lines fall on
    # multiples of the bar length.
    first_measure = parts[0].getElementsByClass("Measure").first()
    padding = Fraction(first_measure.paddingLeft).limit_denominator(48) if first_measure else Fraction(0)
    for hand, part in enumerate(parts):
        for element in part.flatten().notes:
            if element.duration.isGrace:
                continue
            for p in element.pitches:
                raw.append((Fraction(element.offset).limit_denominator(48) + padding,
                            Fraction(element.quarterLength).limit_denominator(48), p.midi, element, hand))
    # Tied continuations extend the first note instead of re-striking it.
    merged: dict[tuple[int, Fraction], list] = {}
    ends: dict[tuple[int, Fraction], tuple[int, Fraction]] = {}
    for offset, length, midi, element, hand in sorted(raw, key=lambda r: (r[0], r[2])):
        tie = element.tie if isinstance(element, note.Note) else next(
            (n.tie for n in element.notes if n.pitch.midi == midi), None)
        key = (midi, offset)
        if tie is not None and tie.type in ("stop", "continue") and key in ends:
            origin = ends.pop(key)
            merged[origin][1] += length
            ends[(midi, offset + length)] = origin
            continue
        merged[key] = [offset, length, midi, hand]
        ends[(midi, offset + length)] = key

    def to_seconds(ql: Fraction) -> float:
        q = float(ql)
        if not humanize:
            return q * seconds_per_quarter
        # Gentle rubato: tempo drifts sinusoidally by +/-6%.
        import math
        return q * seconds_per_quarter + 0.06 * seconds_per_quarter * 4 * math.sin(q / 8)

    events = []
    for offset, length, midi, hand in merged.values():
        onset = to_seconds(offset)
        offset_s = to_seconds(offset + length) - 0.03
        if humanize:
            onset += rng.gauss(0, 0.012)
            offset_s += rng.gauss(0, 0.02)
        velocity = rng.randint(50, 85) if humanize else 70
        events.append((NoteEvent(max(0.0, onset) + 0.5, max(onset + 0.05, offset_s) + 0.5, midi, velocity), offset, hand))
    return events


def render(events, soundfont: str, workdir: Path, name: str) -> Path:
    midi = workdir / f"{name}.mid"
    write_midi(Transcription([e for e, _, _ in events]), midi)
    wav = workdir / f"{name}.wav"
    mp3 = workdir / f"{name}.mp3"
    subprocess.run(["fluidsynth", "-ni", "-g", "0.8", "-r", "44100", "-F", str(wav), soundfont, str(midi)],
                   check=True, capture_output=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), "-b:a", "192k", str(mp3)], check=True)
    return mp3


def score_notes(path: Path) -> list[tuple[Fraction, int, int]]:
    """``(offset, pitch, staff)`` of every attacked note of a two-staff score."""
    parsed = converter.parse(str(path))
    result = []
    for staff, part in enumerate(parsed.parts):
        for element in part.flatten().notes:
            for n in (element.notes if isinstance(element, chord.Chord) else [element]):
                if n.tie is not None and n.tie.type in ("stop", "continue"):
                    continue
                result.append((Fraction(element.offset).limit_denominator(48), n.pitch.midi, staff))
    return sorted(result)


def compare(expected: list[tuple[Fraction, int, int]], produced: list[tuple[Fraction, int, int]],
            bar_length: Fraction) -> dict:
    """Pitch/onset/staff agreement, allowing a global shift and a tempo factor of 1/2, 1 or 2."""
    best = None
    expected_onsets = Counter((o, p) for o, p, _ in expected)
    for factor in (Fraction(1, 2), Fraction(1), Fraction(2)):
        scaled = [(o * factor, p) for o, p, _ in produced]
        shifts = Counter(e[0] - s[0] for e in expected[:40] for s in scaled[:60] if e[1] == s[1])
        for shift, _ in shifts.most_common(5):
            matched = sum((Counter((o + shift, p) for o, p in scaled) & expected_onsets).values())
            if best is None or matched > best[0]:
                best = (matched, factor, shift)
    matched, factor, shift = best
    expected_hands = {(o, p): h for o, p, h in expected}
    same_hand = sum(1 for o, p, h in produced if expected_hands.get((o * factor + shift, p)) == h)
    pitch_matched = sum((Counter(p for _, p, _ in expected) & Counter(p for _, p, _ in produced)).values())
    return {
        "notes_expected": len(expected),
        "notes_produced": len(produced),
        "pitch_recall": pitch_matched / max(1, len(expected)),
        "rhythm_precision": matched / max(1, len(produced)),
        "rhythm_recall": matched / max(1, len(expected)),
        "hand_accuracy": same_hand / max(1, matched),
        "tempo_factor": str(factor),
        # Bar lines fall where the composer put them (given the same meter).
        "downbeats_aligned": shift % bar_length == 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--soundfont", default="/usr/share/sounds/sf2/FluidR3_GM.sf2")
    parser.add_argument("--pieces", nargs="*", default=list(PIECES))
    parser.add_argument("--no-humanize", action="store_true")
    parser.add_argument("--keep", type=Path, help="Keep generated files in this directory")
    parser.add_argument("--reuse", action="store_true",
                        help="Reuse transcriptions already in the --keep directory (only re-engrave)")
    args = parser.parse_args()

    workdir = args.keep or Path(tempfile.mkdtemp(prefix="piano-sheet-eval-"))
    workdir.mkdir(parents=True, exist_ok=True)
    for name in args.pieces:
        path, (first, last), bpm, part_indices, expected_meter, expected_key = PIECES[name]
        source = corpus.parse(path)
        parts = [source.parts[i].measures(first, last) for i in part_indices]
        events = performed_notes(parts, bpm, humanize=not args.no_humanize, seed=1)
        cached = workdir / f"{name}-out" / f"{name}.transcription.mid"
        if args.reuse and cached.exists():
            transcription = read_midi(cached)
            score, analysis = arrange(transcription, Options(title=name))
            files = write_outputs(transcription, score, workdir / f"{name}-out", name)
        else:
            mp3 = render(events, args.soundfont, workdir, name)
            analysis, files = run(mp3, workdir / f"{name}-out")
        expected = sorted({(offset, e.pitch, hand) for e, offset, hand in events})
        metrics = compare(expected, score_notes(files.musicxml), Meter.parse(expected_meter).bar_ql)
        print(f"== {name}")
        print(f"   detected: {analysis.tempo_bpm} BPM, {analysis.time_signature} (expected {expected_meter}), "
              f"{analysis.key_name} (expected {expected_key}), {analysis.measures} measures")
        for key, value in metrics.items():
            print(f"   {key}: {value:.3f}" if isinstance(value, float) else f"   {key}: {value}")
        print(f"   files: {files.musicxml}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
