"""Audio decoding (MP3, WAV, FLAC, OGG, M4A...) to mono float32 samples."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np

AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".oga", ".m4a", ".aac", ".opus", ".wma", ".aiff", ".aif"}
MIDI_EXTENSIONS = {".mid", ".midi"}


class AudioDecodeError(RuntimeError):
    """The file could not be read as audio."""


def load_audio(path: str | Path, sample_rate: int) -> np.ndarray:
    """Decode ``path`` to a mono float32 signal resampled to ``sample_rate``.

    ffmpeg is used when available since it reads virtually every format;
    otherwise librosa's own decoders are tried.
    """
    path = Path(path)
    if not path.is_file():
        raise AudioDecodeError(f"Fichier introuvable : {path}")
    if shutil.which("ffmpeg"):
        audio = _load_with_ffmpeg(path, sample_rate)
    else:
        audio = _load_with_librosa(path, sample_rate)
    if audio.size == 0:
        raise AudioDecodeError("Le fichier ne contient aucun son.")
    return audio


def _load_with_ffmpeg(path: Path, sample_rate: int) -> np.ndarray:
    command = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", str(path),
        "-vn", "-ac", "1", "-ar", str(sample_rate), "-f", "f32le", "-",
    ]
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode != 0:
        detail = result.stderr.decode(errors="replace").strip().splitlines()
        raise AudioDecodeError(
            "Impossible de décoder le fichier audio"
            + (f" ({detail[-1]})" if detail else "") + "."
        )
    return np.frombuffer(result.stdout, dtype=np.float32).copy()


def _load_with_librosa(path: Path, sample_rate: int) -> np.ndarray:
    import librosa

    try:
        audio, _ = librosa.load(str(path), sr=sample_rate, mono=True)
    except Exception as error:  # librosa surfaces many backend-specific errors
        raise AudioDecodeError(
            f"Impossible de décoder le fichier audio ({error}). Installer ffmpeg permet de lire plus de formats."
        ) from error
    return audio.astype(np.float32)
