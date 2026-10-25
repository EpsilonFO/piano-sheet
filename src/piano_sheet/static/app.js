"use strict";

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

const VIEWS = ["upload-view", "progress-view", "error-view", "result-view"];

function showView(name) {
  for (const view of VIEWS) $("#" + view).hidden = view !== name;
  window.scrollTo({ top: 0 });
}

function formatTime(seconds) {
  const s = Math.max(0, Math.floor(seconds));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

async function errorMessage(response) {
  try {
    const data = await response.json();
    if (typeof data.detail === "string") return data.detail;
    if (Array.isArray(data.detail)) return data.detail.map((d) => d.msg).join(" ");
  } catch (_) {
    /* Not JSON. */
  }
  return `Erreur ${response.status}`;
}

/* ------------------------------------------------------------------ */
/* Key names                                                          */

const STEPS = { C: "Do", D: "Ré", E: "Mi", F: "Fa", G: "Sol", A: "La", B: "Si" };

function keyLabel(key) {
  // music21 notation: upper case = major, lower case = minor, "-" = flat.
  const minor = key[0] === key[0].toLowerCase();
  let name = STEPS[key[0].toUpperCase()];
  if (key.slice(1) === "#") name += "♯";
  if (key.slice(1) === "-") name += "♭";
  return `${name} ${minor ? "mineur" : "majeur"}`;
}

/* ------------------------------------------------------------------ */
/* Recent transcriptions (kept in this browser only)                  */

const RECENT_KEY = "piano-sheet:recent";

function loadRecent() {
  try {
    return JSON.parse(localStorage.getItem(RECENT_KEY)) || [];
  } catch (_) {
    return [];
  }
}

function saveRecent(list) {
  try {
    localStorage.setItem(RECENT_KEY, JSON.stringify(list.slice(0, 12)));
  } catch (_) {
    /* Storage unavailable (private browsing...): not essential. */
  }
}

function rememberJob(job) {
  const list = loadRecent().filter((item) => item.id !== job.id);
  list.unshift({ id: job.id, name: job.options?.title || job.filename, created: job.created_at });
  saveRecent(list);
}

function forgetJob(id) {
  saveRecent(loadRecent().filter((item) => item.id !== id));
}

function renderRecent() {
  const list = loadRecent();
  const container = $("#recent-list");
  container.replaceChildren();
  for (const item of list) {
    const li = document.createElement("li");
    const a = document.createElement("a");
    a.href = `/jobs/${item.id}`;
    const name = document.createElement("span");
    name.className = "recent-name";
    name.textContent = item.name;
    const date = document.createElement("span");
    date.className = "recent-date";
    date.textContent = new Date(item.created * 1000).toLocaleString("fr-FR", {
      day: "numeric", month: "short", hour: "2-digit", minute: "2-digit",
    });
    a.append(name, date);
    a.addEventListener("click", (event) => {
      event.preventDefault();
      navigate(a.getAttribute("href"));
    });
    li.append(a);
    container.append(li);
  }
  $("#recent").hidden = list.length === 0;
}

/* ------------------------------------------------------------------ */
/* Server info: formats, choices                                      */

let info = { extensions: [], max_upload_mb: 50, time_signatures: [], keys: [] };

async function loadInfo() {
  try {
    info = await fetch("/api/info").then((r) => r.json());
  } catch (_) {
    return;
  }
  for (const select of $$("select[data-time-signatures]")) {
    for (const ts of info.time_signatures) select.append(new Option(ts, ts));
  }
  for (const select of $$("select[data-keys]")) {
    for (const key of info.keys) select.append(new Option(keyLabel(key), key));
  }
  $("#file-input").accept = info.extensions.join(",");
  $("#dropzone-hint").textContent =
    `MP3, WAV, FLAC, OGG, M4A ou MIDI — jusqu'à ${info.max_upload_mb} Mo et ${Math.round(info.max_duration_min)} minutes`;
}

/* ------------------------------------------------------------------ */
/* Upload                                                             */

let selectedFile = null;

function extensionOf(name) {
  const dot = name.lastIndexOf(".");
  return dot >= 0 ? name.slice(dot).toLowerCase() : "";
}

function selectFile(file) {
  const error = $("#upload-error");
  error.hidden = true;
  if (!file) return;
  if (info.extensions.length && !info.extensions.includes(extensionOf(file.name))) {
    error.textContent = `Format non pris en charge (${extensionOf(file.name) || "inconnu"}).`;
    error.hidden = false;
    return;
  }
  if (file.size > info.max_upload_mb * 1024 * 1024) {
    error.textContent = `Fichier trop volumineux (maximum ${info.max_upload_mb} Mo).`;
    error.hidden = false;
    return;
  }
  selectedFile = file;
  $("#dropzone").classList.add("has-file");
  $("#dropzone-title").textContent = file.name;
  $("#dropzone-hint").textContent = `${(file.size / 1024 / 1024).toFixed(1)} Mo — cliquez pour changer de fichier`;
  $("#submit-button").disabled = false;
}

function setupUpload() {
  const dropzone = $("#dropzone");
  const input = $("#file-input");
  input.addEventListener("change", () => selectFile(input.files[0]));
  for (const type of ["dragenter", "dragover"]) {
    dropzone.addEventListener(type, (event) => {
      event.preventDefault();
      dropzone.classList.add("dragging");
    });
  }
  for (const type of ["dragleave", "drop"]) {
    dropzone.addEventListener(type, () => dropzone.classList.remove("dragging"));
  }
  dropzone.addEventListener("drop", (event) => {
    event.preventDefault();
    selectFile(event.dataTransfer.files[0]);
  });

  $("#upload-form").addEventListener("submit", (event) => {
    event.preventDefault();
    if (!selectedFile) return;
    const form = event.target;
    const data = new FormData();
    data.append("file", selectedFile);
    for (const name of ["title", "time_signature", "bpm", "key"]) {
      const value = form.elements[name].value.trim();
      if (value) data.append(name, value);
    }
    data.append("dynamics", form.elements.dynamics.checked ? "true" : "false");

    const button = $("#submit-button");
    const progress = $("#upload-progress");
    const error = $("#upload-error");
    button.disabled = true;
    progress.hidden = false;
    error.hidden = true;

    const request = new XMLHttpRequest();
    request.open("POST", "/api/jobs");
    request.responseType = "json";
    request.upload.addEventListener("progress", (e) => {
      if (e.lengthComputable) $("#upload-bar").style.width = `${(100 * e.loaded) / e.total}%`;
    });
    request.addEventListener("load", () => {
      progress.hidden = true;
      button.disabled = false;
      if (request.status === 202) {
        rememberJob(request.response);
        navigate(`/jobs/${request.response.id}`);
      } else {
        const detail = request.response?.detail;
        error.textContent = typeof detail === "string" ? detail : `Erreur ${request.status}`;
        error.hidden = false;
      }
    });
    request.addEventListener("error", () => {
      progress.hidden = true;
      button.disabled = false;
      error.textContent = "Le serveur est injoignable.";
      error.hidden = false;
    });
    request.send(data);
  });
}

/* ------------------------------------------------------------------ */
/* Routing and job polling                                            */

let pollTimer = null;
let currentJob = null;
let fileVersion = Date.now();

function navigate(path) {
  if (location.pathname !== path) history.pushState({}, "", path);
  route();
}

function route() {
  clearTimeout(pollTimer);
  player.stop();
  const match = location.pathname.match(/^\/jobs\/([0-9a-f]{32})$/);
  if (match) {
    openJob(match[1]);
  } else {
    currentJob = null;
    renderRecent();
    showView("upload-view");
  }
}

async function openJob(id) {
  let response;
  try {
    response = await fetch(`/api/jobs/${id}`);
  } catch (_) {
    pollTimer = setTimeout(() => openJob(id), 3000);
    return;
  }
  if (response.status === 404) {
    forgetJob(id);
    showError("Cette transcription n'existe plus (elle a peut-être expiré).");
    return;
  }
  if (!response.ok) {
    showError(await errorMessage(response));
    return;
  }
  const job = await response.json();
  if (job.state === "done") {
    if (currentJob?.id !== job.id) showResult(job);
  } else if (job.state === "error") {
    showError(job.error || "Erreur inconnue.");
  } else {
    showProgress(job);
    pollTimer = setTimeout(() => openJob(id), 1000);
  }
}

function showProgress(job) {
  showViewOnce("progress-view");
  $("#progress-file").textContent = job.filename;
  $("#progress-bar").style.width = `${Math.round(job.progress * 100)}%`;
  $("#progress-stage").textContent = job.queue_position > 1
    ? `En attente (${job.queue_position - 1} transcription(s) avant la vôtre)`
    : job.stage_label;
  $("#progress-percent").textContent = job.state === "running" ? `${Math.round(job.progress * 100)} %` : "";
}

function showViewOnce(name) {
  if ($("#" + name).hidden) showView(name);
}

function showError(message) {
  $("#error-message").textContent = message;
  showView("error-view");
}

/* ------------------------------------------------------------------ */
/* Result: summary, sheet music, adjustments                          */

let osmd = null;
let zoom = window.innerWidth < 640 ? 0.7 : 1;

function fileUrl(job, kind) {
  return `${job.files[kind]}?v=${fileVersion}`;
}

function fillSummary(job) {
  const analysis = job.analysis;
  $("#result-title").textContent = job.options.title || job.filename;
  document.title = `${job.options.title || job.filename} — Piano Sheet`;
  const chips = [
    `Tempo ${analysis.tempo_bpm}`,
    `Mesure ${analysis.time_signature}`,
    analysis.key_name,
    `${analysis.notes} notes`,
    `${analysis.measures} mesures`,
  ];
  $("#result-chips").replaceChildren(...chips.map((text) => {
    const li = document.createElement("li");
    li.textContent = text;
    return li;
  }));
  $("#result-warnings").replaceChildren(...analysis.warnings.map((text) => {
    const li = document.createElement("li");
    li.textContent = text;
    return li;
  }));
  $("#download-musicxml").href = fileUrl(job, "musicxml");
  $("#download-midi").href = fileUrl(job, "midi");
  $("#download-transcription").href = fileUrl(job, "transcription");

  const form = $("#rescore-form");
  form.elements.title.value = job.options.title || "";
  form.elements.time_signature.value = job.options.time_signature || "";
  form.elements.bpm.value = job.options.bpm || "";
  form.elements.key.value = job.options.key || "";
  form.elements.min_velocity.value = job.options.min_velocity ?? 12;
  $("#min-velocity-value").textContent = `(${form.elements.min_velocity.value})`;
  form.elements.dynamics.checked = job.options.dynamics !== false;
}

async function showResult(job) {
  currentJob = job;
  rememberJob(job);
  fileVersion = Date.now();
  fillSummary(job);
  const audio = $("#original-audio");
  audio.src = job.audio_url;
  showView("result-view");
  await renderSheet(job);
  await player.load(fileUrl(job, "midi"));
}

async function renderSheet(job) {
  const sheet = $("#sheet");
  const status = $("#sheet-status");
  if (typeof opensheetmusicdisplay === "undefined") {
    status.textContent = "";
    sheet.textContent =
      "L'affichage de la partition nécessite une connexion à Internet (bibliothèque OpenSheetMusicDisplay). " +
      "Vous pouvez télécharger le fichier MusicXML et l'ouvrir dans MuseScore.";
    return;
  }
  sheet.classList.add("loading");
  status.textContent = "Mise en page de la partition…";
  try {
    const xml = await fetch(fileUrl(job, "musicxml")).then((r) => r.text());
    if (!osmd) {
      osmd = new opensheetmusicdisplay.OpenSheetMusicDisplay(sheet, {
        autoResize: true,
        backend: "svg",
        drawTitle: true,
        drawSubtitle: false,
        drawComposer: true,
        drawPartNames: false,
        drawingParameters: "default",
      });
    }
    await osmd.load(xml);
    osmd.zoom = zoom;
    osmd.render();
    player.resetCursor();
    status.textContent = "";
  } catch (error) {
    console.error(error);
    status.textContent = "Impossible d'afficher la partition. Le fichier MusicXML reste téléchargeable.";
  } finally {
    sheet.classList.remove("loading");
  }
}

function setZoom(value) {
  zoom = Math.min(2, Math.max(0.4, Math.round(value * 10) / 10));
  $("#zoom-level").textContent = `${Math.round(zoom * 100)} %`;
  if (osmd && osmd.Sheet) {
    osmd.zoom = zoom;
    osmd.render();
    player.resetCursor();
  }
}

function setupResult() {
  const minVelocity = $("#rescore-form").elements.min_velocity;
  minVelocity.addEventListener("input", () => {
    $("#min-velocity-value").textContent = `(${minVelocity.value})`;
  });
  $("#zoom-in").addEventListener("click", () => setZoom(zoom + 0.1));
  $("#zoom-out").addEventListener("click", () => setZoom(zoom - 0.1));
  $("#zoom-level").textContent = `${Math.round(zoom * 100)} %`;

  $("#print-button").addEventListener("click", () => {
    if (!osmd || !osmd.Sheet) return window.print();
    player.pause();
    // Paginated A4 layout for printing, back to continuous layout afterwards.
    osmd.setOptions({ pageFormat: "A4_P" });
    osmd.zoom = 1;
    osmd.render();
    window.print();
  });
  window.addEventListener("afterprint", () => {
    if (!osmd || !osmd.Sheet) return;
    osmd.setOptions({ pageFormat: "Endless" });
    osmd.zoom = zoom;
    osmd.render();
    player.resetCursor();
  });

  $("#rescore-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!currentJob) return;
    const form = event.target;
    const button = $("#rescore-button");
    const status = $("#rescore-status");
    const body = {
      title: form.elements.title.value.trim() || null,
      time_signature: form.elements.time_signature.value || null,
      bpm: form.elements.bpm.value ? Number(form.elements.bpm.value) : null,
      key: form.elements.key.value || null,
      min_velocity: Number(form.elements.min_velocity.value),
      dynamics: form.elements.dynamics.checked,
    };
    button.disabled = true;
    status.textContent = "Mise à jour…";
    player.stop();
    try {
      const response = await fetch(`/api/jobs/${currentJob.id}/rescore`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!response.ok) throw new Error(await errorMessage(response));
      currentJob = await response.json();
      rememberJob(currentJob);
      fileVersion = Date.now();
      fillSummary(currentJob);
      await renderSheet(currentJob);
      await player.load(fileUrl(currentJob, "midi"));
      status.textContent = "Partition mise à jour.";
    } catch (error) {
      status.textContent = error.message;
    } finally {
      button.disabled = false;
    }
  });

  $("#delete-button").addEventListener("click", async () => {
    if (!currentJob || !confirm("Supprimer définitivement cette transcription ?")) return;
    const response = await fetch(`/api/jobs/${currentJob.id}`, { method: "DELETE" });
    if (response.ok || response.status === 404) {
      forgetJob(currentJob.id);
      navigate("/");
    }
  });
  $("#new-button").addEventListener("click", (event) => {
    event.preventDefault();
    navigate("/");
  });
  $("#home-link").addEventListener("click", (event) => {
    event.preventDefault();
    navigate("/");
  });
}

/* ------------------------------------------------------------------ */
/* Playback of the score with a sampled piano, cursor on the sheet    */

const PIANO_SAMPLES = "https://tonejs.github.io/audio/salamander/";
const SAMPLE_NOTES = [
  "A0", "C1", "D#1", "F#1", "A1", "C2", "D#2", "F#2", "A2", "C3", "D#3", "F#3", "A3", "C4", "D#4",
  "F#4", "A4", "C5", "D#5", "F#5", "A5", "C6", "D#6", "F#6", "A6", "C7", "D#7", "F#7", "A7", "C8",
];

class ScorePlayer {
  constructor() {
    this.midi = null;
    this.sampler = null;
    this.state = "stopped";
    this.scheduled = false;
    this.cursorSeconds = null;
    this.cursorIndex = -1;
    this.frame = null;
  }

  get available() {
    return typeof Tone !== "undefined" && typeof Midi !== "undefined";
  }

  async load(url) {
    this.stop();
    this.midi = null;
    this.scheduled = false;
    this.cursorSeconds = null;
    $("#play-button").disabled = !this.available;
    if (!this.available) return;
    try {
      const buffer = await fetch(url).then((r) => r.arrayBuffer());
      this.midi = new Midi(buffer);
    } catch (error) {
      console.error(error);
    }
    this.showTime(0);
  }

  get duration() {
    return this.midi ? this.midi.duration : 0;
  }

  async ensureSampler() {
    if (!this.sampler) {
      const urls = {};
      for (const note of SAMPLE_NOTES) urls[note] = `${note.replace("#", "s")}.mp3`;
      this.sampler = new Tone.Sampler({ urls, release: 1.2, baseUrl: PIANO_SAMPLES }).toDestination();
    }
    await Tone.loaded();
  }

  schedule() {
    const transport = Tone.Transport;
    transport.cancel();
    for (const track of this.midi.tracks) {
      for (const note of track.notes) {
        transport.schedule((time) => {
          this.sampler.triggerAttackRelease(note.name, note.duration, time, note.velocity);
        }, note.time);
      }
    }
    this.scheduled = true;
  }

  async toggle() {
    if (this.state === "playing") this.pause();
    else await this.play();
  }

  async play() {
    if (!this.midi) return;
    const button = $("#play-button");
    button.disabled = true;
    $("#play-time").textContent = "Chargement…";
    try {
      await Tone.start();
      await this.ensureSampler();
    } catch (error) {
      console.error(error);
      $("#play-time").textContent = "Son indisponible";
      button.disabled = false;
      return;
    }
    button.disabled = false;
    if (!this.scheduled) this.schedule();
    if (Tone.Transport.seconds >= this.duration) Tone.Transport.seconds = 0;
    $("#original-audio").pause();
    Tone.Transport.start();
    this.state = "playing";
    button.classList.add("playing");
    if (osmd && osmd.cursor) osmd.cursor.show();
    this.tick();
  }

  pause() {
    if (this.state !== "playing") return;
    Tone.Transport.pause();
    this.sampler?.releaseAll();
    this.state = "paused";
    $("#play-button").classList.remove("playing");
    cancelAnimationFrame(this.frame);
  }

  stop() {
    if (this.available) {
      Tone.Transport.stop();
      Tone.Transport.seconds = 0;
    }
    this.sampler?.releaseAll();
    this.state = "stopped";
    $("#play-button").classList.remove("playing");
    cancelAnimationFrame(this.frame);
    this.showTime(0);
    if (osmd && osmd.cursor) {
      osmd.cursor.reset();
      osmd.cursor.hide();
      this.cursorIndex = 0;
    }
  }

  seek(fraction) {
    if (!this.midi) return;
    this.sampler?.releaseAll();
    Tone.Transport.seconds = fraction * this.duration;
    this.showTime(Tone.Transport.seconds);
    this.moveCursor(Tone.Transport.seconds, true);
  }

  showTime(seconds) {
    $("#play-time").textContent = `${formatTime(seconds)} / ${formatTime(this.duration)}`;
    if (this.duration && document.activeElement !== $("#seek")) {
      $("#seek").value = String(Math.round((1000 * seconds) / this.duration));
    }
  }

  tick() {
    const seconds = Tone.Transport.seconds;
    this.showTime(seconds);
    this.moveCursor(seconds, false);
    if (seconds > this.duration + 0.5) {
      this.stop();
      return;
    }
    this.frame = requestAnimationFrame(() => this.tick());
  }

  resetCursor() {
    this.cursorSeconds = null;
    this.cursorIndex = -1;
  }

  /** Time (seconds) of every cursor position of the rendered score. */
  computeCursorTimes() {
    if (!osmd || !osmd.cursor || !this.midi) return null;
    const cursor = osmd.cursor;
    const header = this.midi.header;
    const times = [];
    cursor.reset();
    while (!cursor.Iterator.EndReached) {
      const wholeNotes = cursor.Iterator.CurrentSourceTimestamp.RealValue;
      times.push(header.ticksToSeconds(Math.round(wholeNotes * 4 * header.ppq)));
      cursor.next();
    }
    cursor.reset();
    this.cursorIndex = 0;
    return times;
  }

  moveCursor(seconds, force) {
    if (!osmd || !osmd.cursor) return;
    if (!this.cursorSeconds) {
      this.cursorSeconds = this.computeCursorTimes();
      if (!this.cursorSeconds) return;
    }
    const times = this.cursorSeconds;
    let target = 0;
    let low = 0;
    let high = times.length - 1;
    while (low <= high) {
      const mid = (low + high) >> 1;
      if (times[mid] <= seconds + 0.02) {
        target = mid;
        low = mid + 1;
      } else {
        high = mid - 1;
      }
    }
    if (target === this.cursorIndex && !force) return;
    const cursor = osmd.cursor;
    if (target < this.cursorIndex) {
      cursor.reset();
      this.cursorIndex = 0;
    }
    while (this.cursorIndex < target) {
      cursor.next();
      this.cursorIndex += 1;
    }
    cursor.show();
    if ($("#follow").checked && cursor.cursorElement) {
      const rect = cursor.cursorElement.getBoundingClientRect();
      if (rect.top < 80 || rect.bottom > window.innerHeight - 40) {
        cursor.cursorElement.scrollIntoView({ block: "center", behavior: "smooth" });
      }
    }
  }
}

const player = new ScorePlayer();

function setupPlayer() {
  $("#play-button").addEventListener("click", () => player.toggle());
  $("#stop-button").addEventListener("click", () => player.stop());
  $("#seek").addEventListener("input", (event) => player.seek(Number(event.target.value) / 1000));
  // Playing the original recording and the score at the same time is confusing.
  $("#original-audio").addEventListener("play", () => player.pause());
}

/* ------------------------------------------------------------------ */

document.addEventListener("DOMContentLoaded", async () => {
  setupUpload();
  setupResult();
  setupPlayer();
  window.addEventListener("popstate", route);
  await loadInfo();
  route();
});
