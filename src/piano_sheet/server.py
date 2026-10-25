"""Web application: upload a recording, follow the transcription, read the score."""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .audio import AudioDecodeError
from .keys import SUPPORTED_KEYS
from .notes import read_midi
from .pipeline import (
    SUPPORTED_EXTENSIONS,
    Options,
    arrange,
    safe_basename,
    title_from_filename,
    transcribe_file,
    write_outputs,
)
from .rhythm import SUPPORTED_TIME_SIGNATURES
from .transcriber import ModelUnavailableError

logger = logging.getLogger("piano_sheet")

STATIC_DIR = Path(__file__).parent / "static"

# Share of the overall progress bar given to each stage.
_STAGES = {
    "queued": (0.0, 0.0, "En attente"),
    "decode": (0.0, 0.05, "Décodage de l'audio"),
    "model": (0.05, 0.1, "Chargement du modèle"),
    "transcribe": (0.1, 0.9, "Transcription des notes"),
    "arrange": (0.9, 1.0, "Écriture de la partition"),
    "done": (1.0, 1.0, "Terminé"),
    "error": (0.0, 0.0, "Échec"),
}

_FILES = {
    # kind: (stored name, download suffix, media type)
    "musicxml": ("score.musicxml", ".musicxml", "application/vnd.recordare.musicxml+xml"),
    "midi": ("score.mid", ".mid", "audio/midi"),
    "transcription": ("transcription.mid", ".transcription.mid", "audio/midi"),
}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass
class Job:
    id: str
    filename: str
    input_name: str
    options: dict
    created_at: float = field(default_factory=time.time)
    state: str = "queued"  # queued | running | done | error
    stage: str = "queued"
    progress: float = 0.0
    error: str | None = None
    analysis: dict | None = None

    def public(self, queue_position: int | None = None) -> dict:
        base = f"/api/jobs/{self.id}"
        data = {
            "id": self.id,
            "filename": self.filename,
            "state": self.state,
            "stage": self.stage,
            "stage_label": _STAGES.get(self.stage, (0, 0, self.stage))[2],
            "progress": round(self.progress, 4),
            "error": self.error,
            "analysis": self.analysis,
            "options": self.options,
            "created_at": self.created_at,
            "audio_url": f"{base}/audio",
        }
        if queue_position:
            data["queue_position"] = queue_position
        if self.state == "done":
            data["files"] = {kind: f"{base}/files/{kind}" for kind in _FILES}
        return data


class JobManager:
    """Runs transcriptions one at a time in a background thread."""

    def __init__(self, data_dir: Path, transcriber=None, max_seconds: float | None = None) -> None:
        self.jobs_dir = Path(data_dir) / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.max_seconds = max_seconds
        self._transcriber = transcriber
        self._jobs: dict[str, Job] = {}
        self._queue: list[str] = []
        self._lock = threading.Lock()
        # music21 keeps global state: engrave one score at a time.
        self._engrave_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="transcription")
        self._load_existing()

    @property
    def transcriber(self):
        if self._transcriber is None:
            from .transcriber import PianoTranscriber

            self._transcriber = PianoTranscriber()
        return self._transcriber

    def preload_model(self) -> None:
        """Load the model in the background so that the first job starts fast."""

        def load() -> None:
            try:
                self.transcriber.load()
                logger.info("Modèle de transcription chargé")
            except Exception:
                logger.exception("Le modèle n'a pas pu être chargé")

        self._executor.submit(load)

    def job_dir(self, job_id: str) -> Path:
        return self.jobs_dir / job_id

    def _load_existing(self) -> None:
        for path in self.jobs_dir.glob("*/job.json"):
            try:
                job = Job(**json.loads(path.read_text()))
            except (OSError, ValueError, TypeError):
                continue
            if job.state in ("queued", "running"):
                job.state, job.stage, job.error = "error", "error", "Interrompu par un redémarrage du serveur."
            self._jobs[job.id] = job

    def _save(self, job: Job) -> None:
        path = self.job_dir(job.id) / "job.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(job), ensure_ascii=False))
        tmp.replace(path)

    def get(self, job_id: str) -> Job:
        job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return job

    def queue_position(self, job_id: str) -> int | None:
        with self._lock:
            return self._queue.index(job_id) + 1 if job_id in self._queue else None

    def submit(self, upload_path: Path, filename: str, options: Options) -> Job:
        job_id = uuid.uuid4().hex
        directory = self.job_dir(job_id)
        directory.mkdir(parents=True)
        input_name = "input" + Path(filename).suffix.lower()
        shutil.move(str(upload_path), directory / input_name)
        job = Job(id=job_id, filename=filename, input_name=input_name, options=asdict(options))
        with self._lock:
            self._jobs[job_id] = job
            self._queue.append(job_id)
        self._save(job)
        self._executor.submit(self._run, job)
        return job

    def _set_stage(self, job: Job, stage: str, fraction: float) -> None:
        start, end, _ = _STAGES[stage]
        job.stage = stage
        job.progress = start + (end - start) * max(0.0, min(1.0, fraction))

    def _run(self, job: Job) -> None:
        with self._lock:
            if job.id in self._queue:
                self._queue.remove(job.id)
        job.state = "running"
        self._save(job)
        directory = self.job_dir(job.id)
        try:
            transcription = transcribe_file(
                directory / job.input_name,
                self.transcriber,
                on_stage=lambda stage, fraction: self._set_stage(job, stage, fraction),
                max_seconds=self.max_seconds,
            )
            self._set_stage(job, "arrange", 0.0)
            self._engrave(job, transcription, Options(**job.options))
            job.state, job.stage, job.progress = "done", "done", 1.0
        except (AudioDecodeError, ModelUnavailableError, ValueError) as error:
            job.state, job.stage, job.error = "error", "error", str(error)
        except Exception as error:  # Keep the server alive whatever happens.
            logger.exception("Transcription %s failed", job.id)
            job.state, job.stage, job.error = "error", "error", f"Erreur inattendue : {error}"
        self._save(job)

    def _engrave(self, job: Job, transcription, options: Options) -> None:
        directory = self.job_dir(job.id)
        with self._engrave_lock:
            score, analysis = arrange(transcription, options)
            files = write_outputs(transcription, score, directory, "work")
        files.musicxml.replace(directory / _FILES["musicxml"][0])
        files.score_midi.replace(directory / _FILES["midi"][0])
        files.transcription_midi.replace(directory / _FILES["transcription"][0])
        job.analysis = analysis.to_dict()
        job.options = asdict(options)

    def rescore(self, job_id: str, options: Options) -> Job:
        """Rewrite the score with new settings, reusing the transcribed notes."""
        job = self.get(job_id)
        if job.state != "done":
            raise RuntimeError("La transcription n'est pas terminée.")
        transcription = read_midi(self.job_dir(job_id) / _FILES["transcription"][0])
        self._engrave(job, transcription, options)
        self._save(job)
        return job

    def delete(self, job_id: str) -> None:
        job = self.get(job_id)
        if job.state in ("queued", "running"):
            raise RuntimeError("Impossible de supprimer une transcription en cours.")
        with self._lock:
            del self._jobs[job_id]
        shutil.rmtree(self.job_dir(job_id), ignore_errors=True)

    def cleanup(self, max_age_seconds: float) -> int:
        """Delete finished jobs older than ``max_age_seconds``."""
        now = time.time()
        expired = [
            job.id for job in list(self._jobs.values())
            if job.state in ("done", "error") and now - job.created_at > max_age_seconds
        ]
        for job_id in expired:
            self.delete(job_id)
        return len(expired)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


class RescoreRequest(BaseModel):
    time_signature: str | None = None
    bpm: float | None = None
    key: str | None = None
    title: str | None = None
    dynamics: bool = True
    min_velocity: int = 12


def _options(time_signature, bpm, key, title, dynamics, min_velocity) -> Options:
    options = Options(
        time_signature=time_signature or None,
        bpm=bpm or None,
        key=key or None,
        title=(title or "").strip() or None,
        dynamics=dynamics,
        min_velocity=min_velocity,
    )
    try:
        options.validate()
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return options


def create_app(
    data_dir: str | Path | None = None,
    transcriber=None,
    preload_model: bool | None = None,
) -> FastAPI:
    data_dir = Path(data_dir or os.environ.get("PIANO_SHEET_DATA_DIR", "data"))
    max_upload = int(_env_float("PIANO_SHEET_MAX_UPLOAD_MB", 50) * 1024 * 1024)
    max_minutes = _env_float("PIANO_SHEET_MAX_DURATION_MIN", 20)
    retention_hours = _env_float("PIANO_SHEET_RETENTION_HOURS", 72)
    if preload_model is None:
        preload_model = os.environ.get("PIANO_SHEET_PRELOAD_MODEL", "1") != "0" and transcriber is None

    manager = JobManager(data_dir, transcriber=transcriber, max_seconds=max_minutes * 60)
    if retention_hours > 0:
        manager.cleanup(retention_hours * 3600)
    if preload_model:
        manager.preload_model()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        manager.shutdown()

    app = FastAPI(title="Piano Sheet", description="De l'enregistrement à la partition de piano.", lifespan=lifespan)
    app.state.jobs = manager

    def find(job_id: str) -> Job:
        try:
            return manager.get(job_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Transcription introuvable.") from error

    @app.get("/api/info")
    def info() -> dict:
        return {
            "extensions": sorted(SUPPORTED_EXTENSIONS),
            "max_upload_mb": max_upload // (1024 * 1024),
            "max_duration_min": max_minutes,
            "time_signatures": list(SUPPORTED_TIME_SIGNATURES),
            "keys": SUPPORTED_KEYS,
        }

    @app.post("/api/jobs", status_code=202)
    def create_job(
        file: UploadFile = File(...),
        time_signature: str | None = Form(None),
        bpm: float | None = Form(None),
        key: str | None = Form(None),
        title: str | None = Form(None),
        dynamics: bool = Form(True),
        min_velocity: int = Form(12),
    ) -> dict:
        filename = Path(file.filename or "enregistrement").name
        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=415,
                detail=f"Format non pris en charge ({suffix or 'inconnu'}). Formats acceptés : "
                + ", ".join(sorted(SUPPORTED_EXTENSIONS)),
            )
        options = _options(time_signature, bpm, key, title or title_from_filename(filename), dynamics, min_velocity)

        manager.jobs_dir.mkdir(parents=True, exist_ok=True)
        upload_path = manager.jobs_dir / f".upload-{uuid.uuid4().hex}{suffix}"
        size = 0
        try:
            with open(upload_path, "wb") as out:
                while chunk := file.file.read(1 << 20):
                    size += len(chunk)
                    if size > max_upload:
                        raise HTTPException(
                            status_code=413,
                            detail=f"Fichier trop volumineux (maximum {max_upload // (1024 * 1024)} Mo).",
                        )
                    out.write(chunk)
            if size == 0:
                raise HTTPException(status_code=400, detail="Le fichier est vide.")
            job = manager.submit(upload_path, filename, options)
        finally:
            upload_path.unlink(missing_ok=True)
        return job.public(manager.queue_position(job.id))

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        job = find(job_id)
        return job.public(manager.queue_position(job.id))

    @app.post("/api/jobs/{job_id}/rescore")
    def rescore(job_id: str, request: RescoreRequest) -> dict:
        job = find(job_id)
        options = _options(
            request.time_signature,
            request.bpm,
            request.key,
            request.title or job.options.get("title"),
            request.dynamics,
            request.min_velocity,
        )
        try:
            job = manager.rescore(job_id, options)
        except RuntimeError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return job.public()

    @app.delete("/api/jobs/{job_id}", status_code=204)
    def delete_job(job_id: str) -> None:
        find(job_id)
        try:
            manager.delete(job_id)
        except RuntimeError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.get("/api/jobs/{job_id}/files/{kind}")
    def download(job_id: str, kind: str) -> FileResponse:
        job = find(job_id)
        if kind not in _FILES or job.state != "done":
            raise HTTPException(status_code=404, detail="Fichier introuvable.")
        stored, suffix, media_type = _FILES[kind]
        name = safe_basename(job.options.get("title") or Path(job.filename).stem) + suffix
        return FileResponse(manager.job_dir(job_id) / stored, media_type=media_type, filename=name)

    @app.get("/api/jobs/{job_id}/audio")
    def audio(job_id: str) -> FileResponse:
        job = find(job_id)
        return FileResponse(manager.job_dir(job_id) / job.input_name, filename=job.filename)

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/jobs/{job_id}", include_in_schema=False)
    def job_page(job_id: str) -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> RedirectResponse:
        return RedirectResponse("/static/favicon.svg")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app
