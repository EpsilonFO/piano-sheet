import time
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from piano_sheet.notes import Transcription, write_midi
from piano_sheet.server import create_app

from conftest import perform


class FakeTranscriber:
    """Stands in for the neural network: returns a fixed waltz."""

    sample_rate = 16_000

    def __init__(self, notes):
        self.notes = notes
        self.loaded = False

    def load(self):
        self.loaded = True

    def transcribe(self, audio, progress=None):
        for step in range(1, 5):
            if progress:
                progress(step / 4)
        return Transcription(self.notes, duration=len(audio) / self.sample_rate)


def write_wav(path, seconds=1.0):
    samples = (np.sin(np.linspace(0, 2 * np.pi * 440 * seconds, int(16000 * seconds))) * 8000).astype(np.int16)
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(16000)
        out.writeframes(samples.tobytes())


@pytest.fixture
def client(tmp_path, waltz):
    app = create_app(data_dir=tmp_path / "data", transcriber=FakeTranscriber(perform(waltz, bpm=120, seed=1)))
    with TestClient(app) as test_client:
        yield test_client


def wait_for(client, job_id, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["state"] in ("done", "error"):
            return job
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def test_index_and_info(client):
    assert client.get("/").status_code == 200
    assert "Piano Sheet" in client.get("/").text
    info = client.get("/api/info").json()
    assert ".mp3" in info["extensions"]
    assert "3/4" in info["time_signatures"]


def test_audio_upload_to_score(client, tmp_path):
    audio = tmp_path / "Ma valse préférée.wav"
    write_wav(audio)
    with open(audio, "rb") as f:
        response = client.post("/api/jobs", files={"file": (audio.name, f, "audio/wav")})
    assert response.status_code == 202, response.text
    job = wait_for(client, response.json()["id"])
    assert job["state"] == "done", job["error"]
    assert job["analysis"]["time_signature"] == "3/4"
    assert job["options"]["title"] == "Ma valse préférée"

    musicxml = client.get(job["files"]["musicxml"])
    assert musicxml.status_code == 200
    assert b"<score-partwise" in musicxml.content
    assert "Ma_valse_préférée.musicxml" in musicxml.headers["content-disposition"] or \
        "Ma_valse_pr" in musicxml.headers["content-disposition"]
    assert client.get(job["files"]["midi"]).content.startswith(b"MThd")
    assert client.get(job["files"]["transcription"]).content.startswith(b"MThd")
    assert client.get(job["audio_url"]).status_code == 200

    # Changing the settings rewrites the score without transcribing again.
    rescored = client.post(f"/api/jobs/{job['id']}/rescore", json={"time_signature": "4/4", "key": "G"})
    assert rescored.status_code == 200, rescored.text
    analysis = rescored.json()["analysis"]
    assert (analysis["time_signature"], analysis["key_name"]) == ("4/4", "Sol majeur")
    assert rescored.json()["options"]["title"] == "Ma valse préférée"

    assert client.delete(f"/api/jobs/{job['id']}").status_code == 204
    assert client.get(f"/api/jobs/{job['id']}").status_code == 404


def test_midi_upload_skips_transcription(client, tmp_path, waltz):
    path = tmp_path / "valse.mid"
    write_midi(Transcription(perform(waltz, bpm=120, seed=2)), path)
    with open(path, "rb") as f:
        response = client.post("/api/jobs", files={"file": ("valse.mid", f, "audio/midi")},
                               data={"time_signature": "3/4", "title": "Valse"})
    job = wait_for(client, response.json()["id"])
    assert job["state"] == "done"
    assert job["analysis"]["measures"] == 16


def test_rejects_unsupported_files(client):
    response = client.post("/api/jobs", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 415
    response = client.post("/api/jobs", files={"file": ("empty.mp3", b"", "audio/mpeg")})
    assert response.status_code == 400


def test_rejects_invalid_options(client, tmp_path):
    audio = tmp_path / "a.wav"
    write_wav(audio)
    with open(audio, "rb") as f:
        response = client.post("/api/jobs", files={"file": ("a.wav", f, "audio/wav")}, data={"time_signature": "7/8"})
    assert response.status_code == 422


def test_undecodable_audio_reports_an_error(client):
    response = client.post("/api/jobs", files={"file": ("broken.mp3", b"not really an mp3" * 100, "audio/mpeg")})
    job = wait_for(client, response.json()["id"])
    assert job["state"] == "error"
    assert job["error"]


def test_unknown_job(client):
    assert client.get("/api/jobs/" + "0" * 32).status_code == 404
    assert client.get("/api/jobs/" + "0" * 32 + "/files/musicxml").status_code == 404


def test_jobs_survive_a_restart(tmp_path, waltz):
    data = tmp_path / "data"
    transcriber = FakeTranscriber(perform(waltz, bpm=120, seed=1))
    audio = tmp_path / "a.wav"
    write_wav(audio)
    with TestClient(create_app(data_dir=data, transcriber=transcriber)) as first:
        with open(audio, "rb") as f:
            job_id = first.post("/api/jobs", files={"file": ("a.wav", f, "audio/wav")}).json()["id"]
        wait_for(first, job_id)
    with TestClient(create_app(data_dir=data, transcriber=transcriber)) as second:
        job = second.get(f"/api/jobs/{job_id}").json()
        assert job["state"] == "done"
        assert second.get(job["files"]["musicxml"]).status_code == 200
