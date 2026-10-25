from fractions import Fraction

from piano_sheet.hands import LEFT, RIGHT, TimedPitch, assign_hands


def notes_from(events):
    return [TimedPitch(Fraction(start), Fraction(start) + 1, pitch) for start, pitch in events]


def test_melody_and_bass_are_separated():
    events = [(0, 48), (0, 72), (1, 55), (1, 74), (2, 52), (2, 76), (3, 55), (3, 72)]
    hands = assign_hands(notes_from(events))
    for (start, pitch), hand in zip(events, hands):
        assert hand == (RIGHT if pitch >= 60 else LEFT), (start, pitch)


def test_melody_dipping_below_middle_c_stays_in_right_hand():
    # Right hand: E4 D4 C4 B3 C4 while the left hand repeats a low C/G figure.
    melody = [(0, 64), (1, 62), (2, 60), (3, 59), (4, 60)]
    bass = [(t / 2, 36 if t % 2 == 0 else 43) for t in range(10)]
    events = sorted(melody + bass)
    hands = assign_hands(notes_from(events))
    for (start, pitch), hand in zip(events, hands):
        assert hand == (RIGHT if (start, pitch) in melody else LEFT), (start, pitch)


def test_wide_chord_is_split_between_hands():
    events = [(0, 36), (0, 48), (0, 64), (0, 67), (0, 72)]
    hands = assign_hands(notes_from(events))
    assert hands == [LEFT, LEFT, RIGHT, RIGHT, RIGHT]


def test_high_passage_only_uses_right_hand():
    events = [(i, 72 + i % 5) for i in range(8)]
    assert set(assign_hands(notes_from(events))) == {RIGHT}
