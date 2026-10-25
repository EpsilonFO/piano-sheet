"""Audio -> notes, with the high-resolution piano transcription model.

The model is the "Note_pedal" CRNN of Kong et al., "High-resolution Piano
Transcription with Pedals by Regressing Onset and Offset Times" (2021),
trained on the MAESTRO dataset. Its weights (~165 MB, CC-BY-4.0) are
downloaded from Zenodo on first use.
"""

from __future__ import annotations

import hashlib
import os
import threading
import urllib.request
from pathlib import Path
from typing import Callable

import numpy as np

from .notes import NoteEvent, PedalEvent, Transcription

ProgressCallback = Callable[[float], None]

SAMPLE_RATE = 16_000
CHECKPOINT_NAME = "note_F1=0.9677_pedal_F1=0.9186.pth"
CHECKPOINT_URL = (
    "https://zenodo.org/records/4034264/files/"
    "CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth?download=1"
)
CHECKPOINT_MD5 = "22b961b77c1878239fec963362097045"
CHECKPOINT_SIZE = 171_966_578

_SEGMENT_SECONDS = 10


class ModelUnavailableError(RuntimeError):
    """The model checkpoint is missing and could not be downloaded."""


def default_model_dir() -> Path:
    configured = os.environ.get("PIANO_SHEET_MODEL_DIR")
    if configured:
        return Path(configured).expanduser()
    cache = os.environ.get("XDG_CACHE_HOME")
    return (Path(cache) if cache else Path.home() / ".cache") / "piano-sheet"


def find_checkpoint(model_dir: str | Path | None = None) -> Path | None:
    """Return a complete checkpoint already on disk, if any."""
    candidates = [Path(model_dir or default_model_dir()) / CHECKPOINT_NAME]
    # Location used by the upstream piano_transcription_inference package.
    candidates.append(Path.home() / "piano_transcription_inference_data" / CHECKPOINT_NAME)
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size == CHECKPOINT_SIZE:
            return candidate
    return None


def ensure_checkpoint(
    model_dir: str | Path | None = None,
    progress: ProgressCallback | None = None,
) -> Path:
    """Return the checkpoint path, downloading and verifying it if needed."""
    existing = find_checkpoint(model_dir)
    if existing:
        return existing

    target = Path(model_dir or default_model_dir()) / CHECKPOINT_NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    digest = hashlib.md5()
    received = 0
    try:
        with urllib.request.urlopen(CHECKPOINT_URL, timeout=60) as response, open(partial, "wb") as out:
            while chunk := response.read(1 << 20):
                out.write(chunk)
                digest.update(chunk)
                received += len(chunk)
                if progress:
                    progress(min(1.0, received / CHECKPOINT_SIZE))
    except OSError as error:
        partial.unlink(missing_ok=True)
        raise ModelUnavailableError(
            f"Téléchargement du modèle impossible depuis {CHECKPOINT_URL} ({error}). "
            f"Vous pouvez le télécharger manuellement et le placer dans {target}."
        ) from error
    if digest.hexdigest() != CHECKPOINT_MD5:
        partial.unlink(missing_ok=True)
        raise ModelUnavailableError("Le modèle téléchargé est corrompu (somme de contrôle invalide).")
    partial.replace(target)
    return target


class PianoTranscriber:
    """Thread-safe, lazily loaded wrapper around the transcription model."""

    sample_rate = SAMPLE_RATE

    def __init__(self, checkpoint: str | Path | None = None, device: str | None = None) -> None:
        self._checkpoint = Path(checkpoint) if checkpoint else None
        self._device_name = device
        self._model = None
        self._lock = threading.Lock()

    def load(self) -> None:
        with self._lock:
            self._load_locked()

    def _load_locked(self) -> None:
        if self._model is not None:
            return
        import torch
        from piano_transcription_inference import config
        from piano_transcription_inference.models import Note_pedal

        device = torch.device(self._device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint = self._checkpoint or ensure_checkpoint()
        model = Note_pedal(frames_per_second=config.frames_per_second, classes_num=config.classes_num)
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state["model"], strict=False)
        model.to(device)
        model.eval()
        self._model = model
        self._device = device

    def transcribe(self, audio: np.ndarray, progress: ProgressCallback | None = None) -> Transcription:
        """Transcribe a mono signal sampled at :attr:`sample_rate`."""
        import torch
        from piano_transcription_inference import config
        from piano_transcription_inference.utilities import RegressionPostProcessor

        with self._lock:
            self._load_locked()
            audio = np.asarray(audio, dtype=np.float32)
            segments = _enframe(audio, _SEGMENT_SECONDS * SAMPLE_RATE)
            outputs: dict[str, list[np.ndarray]] = {}
            with torch.inference_mode():
                for index, segment in enumerate(segments):
                    batch = torch.from_numpy(segment[None, :]).to(self._device)
                    for key, value in self._model(batch).items():
                        outputs.setdefault(key, []).append(value.cpu().numpy())
                    if progress:
                        progress((index + 1) / len(segments))

        frames = int(np.ceil(len(audio) / SAMPLE_RATE * config.frames_per_second)) + 1
        output_dict = {key: _deframe(np.concatenate(value))[:frames] for key, value in outputs.items()}
        post_processor = RegressionPostProcessor(
            config.frames_per_second,
            classes_num=config.classes_num,
            onset_threshold=0.3,
            offset_threshold=0.3,
            frame_threshold=0.1,
            pedal_offset_threshold=0.2,
        )
        note_events, pedal_events = post_processor.output_dict_to_midi_events(output_dict)
        notes = [
            NoteEvent(float(e["onset_time"]), float(e["offset_time"]), int(e["midi_note"]), int(e["velocity"]))
            for e in note_events
            if e["offset_time"] > e["onset_time"]
        ]
        pedals = [
            PedalEvent(float(e["onset_time"]), float(e["offset_time"]))
            for e in pedal_events or []
            if e["offset_time"] > e["onset_time"]
        ]
        return Transcription(notes=notes, pedals=pedals, duration=len(audio) / SAMPLE_RATE)


def _enframe(audio: np.ndarray, segment_samples: int) -> list[np.ndarray]:
    """Cut the signal into segments overlapping by half."""
    padded_length = max(1, int(np.ceil(len(audio) / segment_samples))) * segment_samples
    padded = np.zeros(padded_length, dtype=np.float32)
    padded[: len(audio)] = audio
    hop = segment_samples // 2
    return [padded[start : start + segment_samples] for start in range(0, padded_length - segment_samples + 1, hop)]


def _deframe(x: np.ndarray) -> np.ndarray:
    """Stitch half-overlapping segment outputs back together.

    ``x`` has shape (segments, frames, classes); only the central half of each
    inner segment is kept, where the model has context on both sides.
    """
    if x.shape[0] == 1:
        return x[0]
    x = x[:, :-1]  # Extra frame produced by the centered STFT.
    frames = x.shape[1]
    quarter, three_quarters = frames // 4, frames * 3 // 4
    parts = [x[0, :three_quarters]]
    parts.extend(x[i, quarter:three_quarters] for i in range(1, x.shape[0] - 1))
    parts.append(x[-1, quarter:])
    return np.concatenate(parts, axis=0)
