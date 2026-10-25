"""End-to-end pipeline: recording -> notes -> score files."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Callable

from .audio import AUDIO_EXTENSIONS, MIDI_EXTENSIONS, load_audio
from .hands import LEFT, RIGHT, TimedPitch, assign_hands
from .keys import describe_key, estimate_key, parse_key
from .notes import Transcription, read_midi, write_midi
from .rhythm import Meter, chord_positions, estimate_beats, estimate_meter, quantize
from .score import ScoreOptions, build_score

# Called with a stage identifier and the progress (0-1) within that stage.
StageCallback = Callable[[str, float], None]

SUPPORTED_EXTENSIONS = AUDIO_EXTENSIONS | MIDI_EXTENSIONS


@dataclass
class Options:
    """User-tunable settings. ``None`` means "detect automatically"."""

    time_signature: str | None = None
    bpm: float | None = None
    key: str | None = None
    title: str | None = None
    dynamics: bool = True
    # Notes quieter than this are dropped (mostly spurious detections).
    min_velocity: int = 12

    def validate(self) -> None:
        if self.time_signature:
            Meter.parse(self.time_signature)
        if self.key:
            parse_key(self.key)
        if self.bpm is not None and not 20 <= self.bpm <= 300:
            raise ValueError("Le tempo doit être compris entre 20 et 300 BPM.")
        if not 0 <= self.min_velocity <= 127:
            raise ValueError("La vélocité minimale doit être comprise entre 0 et 127.")


@dataclass
class Analysis:
    """What was detected, for display."""

    tempo_bpm: int
    time_signature: str
    key: str
    key_name: str
    measures: int
    notes: int
    duration_seconds: float
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class OutputFiles:
    musicxml: Path
    score_midi: Path
    transcription_midi: Path


def title_from_filename(path: str | Path) -> str:
    stem = Path(path).stem
    title = re.sub(r"[_]+", " ", stem).strip()
    return title or "Partition"


def transcribe_file(
    path: str | Path,
    transcriber=None,
    on_stage: StageCallback | None = None,
    max_seconds: float | None = None,
) -> Transcription:
    """Notes played in an audio file (or read from a MIDI file)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in MIDI_EXTENSIONS:
        return read_midi(path)
    if transcriber is None:
        from .transcriber import PianoTranscriber

        transcriber = PianoTranscriber()
    if on_stage:
        on_stage("decode", 0.0)
    audio = load_audio(path, transcriber.sample_rate)
    seconds = len(audio) / transcriber.sample_rate
    if max_seconds is not None and seconds > max_seconds:
        raise ValueError(
            f"L'enregistrement dure {seconds / 60:.0f} min, la limite est de {max_seconds / 60:.0f} min."
        )
    if on_stage:
        on_stage("model", 0.0)
        transcriber.load()
        on_stage("transcribe", 0.0)
    return transcriber.transcribe(audio, progress=(lambda f: on_stage("transcribe", f)) if on_stage else None)


def arrange(transcription: Transcription, options: Options):
    """Turn performed notes into a score. Returns ``(score, analysis)``."""
    options.validate()
    notes = [n for n in transcription.notes if n.velocity >= options.min_velocity and n.duration > 0.01]
    warnings = []
    if not notes:
        warnings.append("Aucune note de piano n'a été détectée dans cet enregistrement.")
    if len(notes) and len(notes) < 8:
        warnings.append("Très peu de notes détectées : le rythme et la tonalité sont approximatifs.")

    grid = estimate_beats(notes, bpm=options.bpm)
    requested_meter = Meter.parse(options.time_signature) if options.time_signature else None
    meter, first_bar = estimate_meter(notes, grid, requested_meter)

    # Hands are separated before quantizing so that each hand gets its own
    # rhythmic grid: eighths in one hand against triplets in the other are
    # common in piano music.
    positions = chord_positions(notes, grid, first_bar)
    hands = assign_hands([TimedPitch(start, end, n.pitch) for (start, end), n in zip(positions, notes)])
    staves = {
        hand: quantize([n for n, h in zip(notes, hands) if h == hand], grid, meter, first_bar)
        for hand in (RIGHT, LEFT)
    }
    quantized = [n for staff in staves.values() for n in staff.notes]

    if options.key:
        k = parse_key(options.key)
    else:
        k = estimate_key((n.pitch, float(n.end - n.start)) for n in quantized)

    # The notated tempo refers to the beat (dotted quarter in compound meters).
    bpm = options.bpm or grid.bpm
    score = build_score(
        staves,
        meter,
        k,
        bpm,
        ScoreOptions(title=options.title or "Partition", dynamics=options.dynamics),
    )
    analysis = Analysis(
        tempo_bpm=int(round(bpm)),
        time_signature=str(meter),
        key=k.tonicPitchNameWithCase,
        key_name=describe_key(k),
        measures=len(score.parts[0].getElementsByClass("Measure")) if score.parts else 0,
        notes=len(quantized),
        duration_seconds=round(transcription.duration, 2),
        warnings=warnings,
    )
    return score, analysis


def write_outputs(transcription: Transcription, score, output_dir: str | Path, basename: str) -> OutputFiles:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = OutputFiles(
        musicxml=output_dir / f"{basename}.musicxml",
        score_midi=output_dir / f"{basename}.mid",
        transcription_midi=output_dir / f"{basename}.transcription.mid",
    )
    score.write("musicxml", fp=str(files.musicxml))
    # Notes under "8va" lines are stored as written: play them where they sound.
    score.toSoundingPitch().write("midi", fp=str(files.score_midi))
    write_midi(transcription, files.transcription_midi)
    return files


def run(
    input_path: str | Path,
    output_dir: str | Path,
    options: Options | None = None,
    transcriber=None,
    on_stage: StageCallback | None = None,
) -> tuple[Analysis, OutputFiles]:
    """Convert one recording into score files in ``output_dir``."""
    input_path = Path(input_path)
    options = options or Options()
    if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Format non pris en charge : {input_path.suffix or input_path.name}")
    options.validate()
    if not options.title:
        options = replace(options, title=title_from_filename(input_path))
    transcription = transcribe_file(input_path, transcriber, on_stage)
    if on_stage:
        on_stage("arrange", 0.0)
    score, analysis = arrange(transcription, options)
    files = write_outputs(transcription, score, output_dir, safe_basename(input_path.stem))
    return analysis, files


def safe_basename(name: str) -> str:
    cleaned = re.sub(r"[^\w\- ]+", "", name, flags=re.UNICODE).strip().replace(" ", "_")
    return cleaned[:80] or "partition"

