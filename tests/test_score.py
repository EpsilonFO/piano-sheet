import random
from fractions import Fraction

import pytest
from music21 import converter

from piano_sheet.notes import NoteEvent, Transcription
from piano_sheet.pipeline import Options, arrange
from piano_sheet.rhythm import SUPPORTED_TIME_SIGNATURES, Meter
from piano_sheet.score import _split_span

from conftest import perform

F = Fraction


def musicxml_round_trip(score, tmp_path):
    path = tmp_path / "score.musicxml"
    score.write("musicxml", fp=str(path))
    return converter.parse(str(path))


def assert_bars_are_full(parsed, meter_text):
    meter = Meter.parse(meter_text)
    for part in parsed.parts:
        measures = list(part.getElementsByClass("Measure"))
        assert measures, "no measures"
        for measure in measures:
            assert measure.duration.quarterLength == pytest.approx(float(meter.bar_ql)), (
                f"measure {measure.number} of {part.id} lasts {measure.duration.quarterLength}"
            )


@pytest.mark.parametrize(
    "start, end, expected",
    [
        (F(0), F(4), [(F(0), F(4))]),  # Whole note.
        (F(0), F(3, 2), [(F(0), F(3, 2))]),  # Dotted quarter.
        (F(1, 2), F(3, 2), [(F(1, 2), F(1))]),  # Syncopated quarter.
        (F(3), F(5), [(F(3), F(1)), (F(4), F(1))]),  # Across the bar line: tied.
        (F(1, 4), F(2), [(F(1, 4), F(3, 4)), (F(1), F(1))]),  # Off the beat: tie at the beat.
        (F(0), F(5, 4), [(F(0), F(1)), (F(1), F(1, 4))]),
    ],
)
def test_split_span_plain(start, end, expected):
    assert _split_span(start, end, Meter(4, 4), {}) == expected


def test_split_span_in_triplet_beat():
    pieces = _split_span(F(1, 3), F(2), Meter(4, 4), {0: 3})
    assert pieces == [(F(1, 3), F(2, 3)), (F(1), F(1))]
    pieces = _split_span(F(1, 6), F(1), Meter(4, 4), {0: 6})
    assert pieces == [(F(1, 6), F(1, 6)), (F(1, 3), F(2, 3))]


def test_arrange_waltz(waltz, tmp_path):
    notes = perform(waltz, bpm=120, jitter=0.006, seed=4)
    score, analysis = arrange(Transcription(notes), Options(title="Valse"))
    assert analysis.time_signature == "3/4"
    assert analysis.key_name == "Do majeur"
    assert analysis.measures == 16
    assert 115 <= analysis.tempo_bpm <= 125

    parsed = musicxml_round_trip(score, tmp_path)
    assert len(parsed.parts) == 2  # One grand staff, two staves.
    assert_bars_are_full(parsed, "3/4")
    right, left = parsed.parts
    right_pitches = {p.midi for p in right.flatten().pitches}
    left_pitches = {p.midi for p in left.flatten().pitches}
    assert {71, 72, 74, 76, 77, 79} <= right_pitches  # The melody...
    assert {43, 48} <= left_pitches  # ...over the bass.
    assert not right_pitches & {43, 48} and not left_pitches & {71, 72, 74, 76, 77, 79}


def test_user_settings_override_detection(waltz, tmp_path):
    notes = perform(waltz, bpm=120, jitter=0.006, seed=4)
    options = Options(time_signature="6/8", key="a", bpm=60, dynamics=False, title="Test")
    score, analysis = arrange(Transcription(notes), options)
    assert analysis.time_signature == "6/8"
    assert analysis.key_name == "La mineur"
    assert analysis.tempo_bpm == 60
    parsed = musicxml_round_trip(score, tmp_path)
    assert_bars_are_full(parsed, "6/8")
    assert not list(parsed.flatten().getElementsByClass("Dynamic"))


def test_empty_transcription_gives_empty_score(tmp_path):
    score, analysis = arrange(Transcription([]), Options())
    assert analysis.notes == 0
    assert analysis.warnings
    musicxml_round_trip(score, tmp_path)


def random_performance(seed):
    """Messy, unmusical input: random chords, rhythms, tempo and lengths."""
    rng = random.Random(seed)
    notes = []
    time = rng.uniform(0, 1)
    for _ in range(rng.randint(20, 120)):
        for _ in range(rng.choice([1, 1, 1, 2, 3, 4])):
            pitch = rng.randint(24, 100)
            notes.append(NoteEvent(time + rng.uniform(0, 0.03), time + rng.uniform(0.05, 2.5), pitch,
                                   rng.randint(15, 120)))
        time += rng.choice([0.08, 0.12, 0.17, 0.2, 0.25, 0.33, 0.4, 0.5, 0.75, 1.0, 1.6])
    return Transcription(notes)


@pytest.mark.parametrize("seed", range(12))
def test_any_input_gives_a_valid_score(seed, tmp_path):
    """Whatever the transcription, the score must export and every bar must be complete."""
    meter = [None, *SUPPORTED_TIME_SIGNATURES][seed % (len(SUPPORTED_TIME_SIGNATURES) + 1)]
    score, analysis = arrange(random_performance(seed), Options(time_signature=meter))
    parsed = musicxml_round_trip(score, tmp_path)
    assert_bars_are_full(parsed, analysis.time_signature)
    attacked = sum(
        len(getattr(n, "pitches", ())) for n in parsed.flatten().notes
        if n.tie is None or n.tie.type == "start"
    )
    assert attacked == analysis.notes


def test_high_passage_gets_an_ottava_and_midi_keeps_real_pitches(tmp_path):
    import mido

    from piano_sheet.pipeline import write_outputs

    high = [(beat / 2, 0.5, 88 + (beat % 5)) for beat in range(16)]  # E6 and above.
    bass = [(beat, 1, 48) for beat in range(8)]
    transcription = Transcription(perform(high + bass, bpm=100, seed=3))
    score, _ = arrange(transcription, Options(time_signature="4/4"))
    files = write_outputs(transcription, score, tmp_path, "high")
    assert "octave-shift" in files.musicxml.read_text()
    played = sorted({m.note for track in mido.MidiFile(files.score_midi).tracks for m in track
                     if m.type == "note_on" and m.velocity > 0})
    assert played == sorted({p for _, _, p in high + bass})


@pytest.mark.parametrize(
    "name, expected",
    [("Black_Hole_Sun.transcription", "Black_Hole_Sun_transcription"), ("Ma valse préférée", "Ma_valse_préférée"),
     ("a/../b", "a_b"), ("***", "partition")],
)
def test_safe_basename(name, expected):
    from piano_sheet.pipeline import safe_basename

    assert safe_basename(name) == expected
