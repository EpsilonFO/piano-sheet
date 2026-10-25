"""Performance-level note data (seconds) and MIDI input/output."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mido


@dataclass(frozen=True)
class NoteEvent:
    """A played note, timed in seconds."""

    onset: float
    offset: float
    pitch: int
    velocity: int

    @property
    def duration(self) -> float:
        return self.offset - self.onset


@dataclass(frozen=True)
class PedalEvent:
    """A sustain-pedal press, timed in seconds."""

    onset: float
    offset: float


@dataclass
class Transcription:
    """Everything that was heard in a recording."""

    notes: list[NoteEvent]
    pedals: list[PedalEvent] = field(default_factory=list)
    duration: float = 0.0

    def __post_init__(self) -> None:
        self.notes = sorted(self.notes, key=lambda n: (n.onset, n.pitch))
        self.pedals = sorted(self.pedals, key=lambda p: p.onset)
        if self.notes:
            self.duration = max(self.duration, max(n.offset for n in self.notes))


_TICKS_PER_BEAT = 480
_TEMPO = 500_000  # 120 BPM: one beat = 0.5 s, so ticks map linearly to seconds.


def _seconds_to_ticks(seconds: float) -> int:
    return int(round(seconds * _TICKS_PER_BEAT * 1_000_000 / _TEMPO))


def write_midi(transcription: Transcription, path: str | Path) -> None:
    """Write the raw (unquantized) performance as a MIDI file."""
    messages: list[tuple[int, int, mido.Message]] = []
    # The middle element orders simultaneous events: note-offs before note-ons
    # so that repeated notes are not cut short.
    for note in transcription.notes:
        messages.append((_seconds_to_ticks(note.onset), 2,
                         mido.Message("note_on", note=note.pitch, velocity=max(1, min(127, note.velocity)))))
        messages.append((_seconds_to_ticks(note.offset), 0,
                         mido.Message("note_off", note=note.pitch, velocity=0)))
    for pedal in transcription.pedals:
        messages.append((_seconds_to_ticks(pedal.onset), 1,
                         mido.Message("control_change", control=64, value=127)))
        messages.append((_seconds_to_ticks(pedal.offset), 1,
                         mido.Message("control_change", control=64, value=0)))
    messages.sort(key=lambda m: (m[0], m[1]))

    track = mido.MidiTrack()
    track.append(mido.MetaMessage("track_name", name="Piano", time=0))
    track.append(mido.MetaMessage("set_tempo", tempo=_TEMPO, time=0))
    track.append(mido.Message("program_change", program=0, time=0))
    now = 0
    for tick, _, message in messages:
        track.append(message.copy(time=tick - now))
        now = tick
    track.append(mido.MetaMessage("end_of_track", time=0))

    midi = mido.MidiFile(ticks_per_beat=_TICKS_PER_BEAT)
    midi.tracks.append(track)
    midi.save(str(path))


def read_midi(path: str | Path) -> Transcription:
    """Read every note of a MIDI file (all tracks and channels merged)."""
    midi = mido.MidiFile(str(path))
    notes: list[NoteEvent] = []
    pedals: list[PedalEvent] = []
    active: dict[tuple[int, int], list[tuple[float, int]]] = {}
    pedal_down: dict[int, float] = {}
    now = 0.0
    # Iterating a MidiFile yields messages merged across tracks with delta
    # times already converted to seconds (tempo changes included).
    for message in midi:
        now += message.time
        if message.is_meta:
            continue
        if message.type == "note_on" and message.velocity > 0:
            active.setdefault((message.channel, message.note), []).append((now, message.velocity))
        elif message.type in ("note_off", "note_on"):
            started = active.get((message.channel, message.note))
            if started:
                onset, velocity = started.pop(0)
                if now > onset:
                    notes.append(NoteEvent(onset, now, message.note, velocity))
        elif message.type == "control_change" and message.control == 64:
            if message.value >= 64 and message.channel not in pedal_down:
                pedal_down[message.channel] = now
            elif message.value < 64 and message.channel in pedal_down:
                onset = pedal_down.pop(message.channel)
                if now > onset:
                    pedals.append(PedalEvent(onset, now))
    # Close anything left hanging at the end of the file.
    for (_, pitch), started in active.items():
        for onset, velocity in started:
            if now > onset:
                notes.append(NoteEvent(onset, now, pitch, velocity))
    for onset in pedal_down.values():
        if now > onset:
            pedals.append(PedalEvent(onset, now))
    return Transcription(notes=notes, pedals=pedals, duration=now)
