"""Command line interface: ``piano-sheet transcribe``, ``serve`` and ``download-model``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .audio import MIDI_EXTENSIONS
from .rhythm import SUPPORTED_TIME_SIGNATURES


def _progress_bar(label: str):
    def update(fraction: float) -> None:
        width = 30
        filled = int(width * fraction)
        sys.stderr.write(f"\r{label} [{'#' * filled}{'.' * (width - filled)}] {fraction * 100:5.1f}%")
        if fraction >= 1:
            sys.stderr.write("\n")
        sys.stderr.flush()

    return update


_STAGE_LABELS = {
    "decode": "Décodage de l'audio...",
    "model": "Chargement du modèle de transcription...",
    "arrange": "Écriture de la partition...",
}


def _transcribe(args: argparse.Namespace) -> int:
    from .pipeline import Options, run

    options = Options(
        time_signature=args.time_signature,
        bpm=args.bpm,
        key=args.key,
        title=args.title,
        dynamics=not args.no_dynamics,
        min_velocity=args.min_velocity,
    )
    try:
        options.validate()
    except ValueError as error:
        print(f"Erreur : {error}", file=sys.stderr)
        return 2

    transcribe_bar = _progress_bar("Transcription des notes")
    announced: set[str] = set()

    def on_stage(stage: str, fraction: float) -> None:
        if stage == "transcribe":
            transcribe_bar(fraction)
        elif stage not in announced:
            announced.add(stage)
            print(_STAGE_LABELS.get(stage, stage), file=sys.stderr)

    transcriber = None
    if Path(args.input).suffix.lower() not in MIDI_EXTENSIONS:
        from .transcriber import ModelUnavailableError, PianoTranscriber, ensure_checkpoint, find_checkpoint

        if find_checkpoint() is None:
            print("Premier lancement : téléchargement du modèle de transcription.", file=sys.stderr)
            try:
                ensure_checkpoint(progress=_progress_bar("Téléchargement du modèle (165 Mo)"))
            except ModelUnavailableError as error:
                print(f"Erreur : {error}", file=sys.stderr)
                return 1
        transcriber = PianoTranscriber(device=args.device)

    output_dir = args.output or Path(args.input).resolve().parent
    try:
        analysis, files = run(args.input, output_dir, options, transcriber=transcriber, on_stage=on_stage)
    except Exception as error:  # Report any failure as a readable message.
        print(f"Erreur : {error}", file=sys.stderr)
        return 1

    print()
    print(f"Tempo     : {analysis.tempo_bpm} BPM")
    print(f"Mesure    : {analysis.time_signature}")
    print(f"Tonalité  : {analysis.key_name}")
    print(f"Notes     : {analysis.notes} ({analysis.measures} mesures)")
    for warning in analysis.warnings:
        print(f"Attention : {warning}")
    print()
    print(f"Partition (MusicXML) : {files.musicxml}")
    print(f"Partition (MIDI)     : {files.score_midi}")
    print(f"Transcription brute  : {files.transcription_midi}")
    print()
    print("Ouvrez le fichier MusicXML avec MuseScore (gratuit) pour l'afficher, l'éditer ou l'imprimer en PDF.")
    return 0


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .server import create_app

    app = create_app(data_dir=args.data_dir)
    print(f"Piano Sheet est disponible sur http://{args.host}:{args.port}", file=sys.stderr)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def _download_model(args: argparse.Namespace) -> int:
    from .transcriber import ModelUnavailableError, ensure_checkpoint, find_checkpoint

    existing = find_checkpoint(args.model_dir)
    if existing:
        print(f"Le modèle est déjà présent : {existing}")
        return 0
    try:
        path = ensure_checkpoint(args.model_dir, progress=_progress_bar("Téléchargement du modèle (165 Mo)"))
    except ModelUnavailableError as error:
        print(f"Erreur : {error}", file=sys.stderr)
        return 1
    print(f"Modèle installé : {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="piano-sheet",
        description="Transforme un enregistrement de piano (MP3, WAV, FLAC...) en partition.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    transcribe = commands.add_parser(
        "transcribe",
        help="Convertir un fichier audio (ou MIDI) en partition",
        description="Convertit un enregistrement de piano en partition MusicXML et MIDI.",
    )
    transcribe.add_argument("input", help="Fichier audio (mp3, wav, flac, ogg, m4a...) ou MIDI")
    transcribe.add_argument("-o", "--output", type=Path, help="Dossier de sortie (par défaut : celui du fichier)")
    transcribe.add_argument("--title", help="Titre affiché sur la partition (par défaut : nom du fichier)")
    transcribe.add_argument(
        "--time-signature", "-t", choices=SUPPORTED_TIME_SIGNATURES, help="Mesure (détectée automatiquement sinon)"
    )
    transcribe.add_argument("--bpm", type=float, help="Tempo en battements par minute (détecté sinon)")
    transcribe.add_argument("--key", "-k", help="Tonalité, ex. 'Eb', 'f#', 'Bb minor' (détectée sinon)")
    transcribe.add_argument("--no-dynamics", action="store_true", help="Ne pas écrire de nuances (p, mf, f...)")
    transcribe.add_argument(
        "--min-velocity", type=int, default=12, help="Ignorer les notes plus faibles que cette vélocité (0-127)"
    )
    transcribe.add_argument("--device", help="Périphérique PyTorch : 'cpu' ou 'cuda' (automatique par défaut)")
    transcribe.set_defaults(handler=_transcribe)

    serve = commands.add_parser("serve", help="Lancer l'interface web")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--data-dir", type=Path, help="Dossier où stocker les fichiers envoyés et les partitions")
    serve.set_defaults(handler=_serve)

    download = commands.add_parser("download-model", help="Télécharger le modèle de transcription (~165 Mo)")
    download.add_argument("--model-dir", type=Path, help="Dossier de destination")
    download.set_defaults(handler=_download_model)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
