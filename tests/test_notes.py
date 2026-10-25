from piano_sheet.notes import NoteEvent, PedalEvent, Transcription, read_midi, write_midi


def test_midi_round_trip(tmp_path):
    original = Transcription(
        notes=[NoteEvent(0.5, 1.0, 60, 80), NoteEvent(0.5, 2.0, 64, 70), NoteEvent(1.0, 1.25, 60, 90)],
        pedals=[PedalEvent(0.4, 1.9)],
    )
    path = tmp_path / "out.mid"
    write_midi(original, path)
    loaded = read_midi(path)
    assert len(loaded.notes) == 3
    for a, b in zip(original.notes, loaded.notes):
        assert (a.pitch, a.velocity) == (b.pitch, b.velocity)
        assert abs(a.onset - b.onset) < 0.002 and abs(a.offset - b.offset) < 0.002
    assert len(loaded.pedals) == 1
    assert abs(loaded.pedals[0].onset - 0.4) < 0.002
    assert loaded.duration >= 2.0
