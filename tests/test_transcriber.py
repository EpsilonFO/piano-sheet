import numpy as np
import pytest

from piano_sheet.transcriber import SAMPLE_RATE, PianoTranscriber, _deframe, _enframe, find_checkpoint


def test_enframe_deframe_shapes():
    audio = np.random.default_rng(0).standard_normal(SAMPLE_RATE * 25).astype(np.float32)
    segments = _enframe(audio, SAMPLE_RATE * 10)
    assert len(segments) == 5 and all(len(s) == SAMPLE_RATE * 10 for s in segments)
    frames = np.zeros((len(segments), 1001, 88))
    assert _deframe(frames).shape == (3000, 88)


def tone(pitch, seconds, rng):
    """A rough piano-like tone: decaying, slightly inharmonic partials."""
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    f0 = 440 * 2 ** ((pitch - 69) / 12)
    signal = np.zeros_like(t)
    for k in range(1, 9):
        if f0 * k < SAMPLE_RATE / 2:
            signal += 0.6 ** k * np.sin(2 * np.pi * f0 * k * t * (1 + 0.0004 * k * k) + rng.uniform(0, 6))
    return 0.4 * signal * np.exp(-3 * t) * np.minimum(1, t / 0.004)


@pytest.mark.model
@pytest.mark.skipif(find_checkpoint() is None, reason="model checkpoint not downloaded")
def test_model_hears_the_notes():
    rng = np.random.default_rng(0)
    expected = [(0.5 + 0.5 * i, p) for i, p in enumerate([60, 64, 67, 72, 48, 55])]
    audio = np.zeros(SAMPLE_RATE * 5, dtype=np.float32)
    for onset, pitch in expected:
        start = int(onset * SAMPLE_RATE)
        y = tone(pitch, 0.45, rng)
        audio[start:start + len(y)] += y
    transcription = PianoTranscriber(device="cpu").transcribe(audio)
    for onset, pitch in expected:
        assert any(n.pitch == pitch and abs(n.onset - onset) < 0.05 for n in transcription.notes), (onset, pitch)
