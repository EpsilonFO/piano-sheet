import pytest

from piano_sheet.keys import PitchSpeller, describe_key, estimate_key, parse_key


def weighted(pitches):
    return [(p, 1.0) for p in pitches]


def test_estimates_major_key():
    e_flat_major = [63, 65, 67, 68, 70, 72, 74, 75, 63, 67, 70, 63, 70]
    k = estimate_key(weighted(e_flat_major))
    assert (k.tonic.name, k.mode) == ("E-", "major")


def test_estimates_minor_key():
    a_minor = [57, 59, 60, 62, 64, 65, 68, 69, 57, 60, 64, 57, 64, 68, 57]
    k = estimate_key(weighted(a_minor))
    assert (k.tonic.name, k.mode) == ("A", "minor")


def test_empty_input_defaults_to_c_major():
    assert estimate_key([]).tonic.name == "C"


@pytest.mark.parametrize(
    "key, midi, expected",
    [
        ("E-", 68, "A-4"),
        ("E-", 66, "G-4"),
        ("G", 66, "F#4"),
        ("d", 61, "C#4"),  # Leading tone of D minor.
        ("d", 70, "B-4"),
        ("C", 70, "B-4"),
        ("C", 66, "F#4"),
        ("c#", 60, "B#3"),  # Leading tone of C# minor.
        ("F#", 65, "E#4"),
    ],
)
def test_pitch_spelling(key, midi, expected):
    pitch = PitchSpeller(parse_key(key)).spell(midi)
    assert pitch.nameWithOctave == expected
    assert pitch.midi == midi


@pytest.mark.parametrize(
    "text, tonic, mode",
    [("Eb", "E-", "major"), ("f#", "F#", "minor"), ("Bb minor", "B-", "minor"), ("Am", "A", "minor"),
     ("C majeur", "C", "major"), ("d", "D", "minor")],
)
def test_parse_key(text, tonic, mode):
    k = parse_key(text)
    assert (k.tonic.name, k.mode) == (tonic, mode)


def test_parse_key_rejects_garbage():
    with pytest.raises(ValueError):
        parse_key("H#")


def test_describe_key_in_french():
    assert describe_key(parse_key("Eb")) == "Mi♭ majeur"
    assert describe_key(parse_key("c#")) == "Do♯ mineur"
