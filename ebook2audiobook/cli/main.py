"""
CastBook CLI — entry point for all command-line interaction.

Commands (Phase A):
  castbook --help           Show help.
  castbook doctor           Check system and report hardware tier.
  castbook ingest <file>    Extract book and write book.json.

Later milestones will add:
  castbook convert          Run the full pipeline (M2+).
  castbook models download  Download required models (M2+).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import click

from ebook2audiobook import __version__
from ebook2audiobook.utils.env import run_doctor


@click.group()
@click.version_option(__version__, prog_name="castbook")
def cli() -> None:
    """CastBook — local, offline ebook-to-audiobook conversion."""


# ---------------------------------------------------------------------------
# castbook doctor
# ---------------------------------------------------------------------------


@cli.command()
def doctor() -> None:
    """
    Check the system and report the supported hardware tier.

    Checks: Python version, RAM, GPU/VRAM, FFmpeg, free disk space.
    Verdict: narrator-only (low resource) or multi-voice (full pipeline).
    """
    report = run_doctor()

    click.echo("")
    click.secho("╔══════════════════════════════╗", fg="cyan")
    click.secho("║      castbook doctor          ║", fg="cyan")
    click.secho("╚══════════════════════════════╝", fg="cyan")
    click.echo("")

    # Python
    py = report["python"]
    click.secho("Python", bold=True)
    click.echo(f"  Version  : {sys.version.split()[0]}")
    click.echo(f"  Platform : {py['platform']}")
    click.echo("")

    # RAM
    ram = report["ram_gib"]
    click.secho("Memory", bold=True)
    if ram is not None:
        color = "green" if ram >= 8.0 else ("yellow" if ram >= 4.0 else "red")
        click.secho(f"  RAM      : {ram:.1f} GiB", fg=color)
    else:
        click.secho("  RAM      : detection failed", fg="yellow")
    click.echo("")

    # GPU
    gpu = report["gpu"]
    click.secho("GPU", bold=True)
    backend = gpu["backend"]
    color = "green" if backend == "cuda" else ("yellow" if backend == "mps" else "white")
    click.secho(f"  Backend  : {backend.upper()}", fg=color)
    click.echo(f"  Device   : {gpu['device_name']}")
    if gpu["vram_gib"] is not None:
        click.echo(f"  VRAM     : {gpu['vram_gib']:.1f} GiB")
    click.echo("")

    # FFmpeg
    ff = report["ffmpeg"]
    click.secho("FFmpeg", bold=True)
    if ff["installed"]:
        click.secho(f"  Status   : installed ({ff['path']})", fg="green")
        if ff["version"]:
            click.echo(f"  Version  : {ff['version']}")
    else:
        click.secho(
            "  Status   : NOT FOUND — install via Homebrew (macOS: brew install ffmpeg) "
            "or apt (Linux: sudo apt install ffmpeg)",
            fg="red",
        )
    click.echo("")

    # Disk
    disk = report["disk_free_gib"]
    click.secho("Disk", bold=True)
    if disk is not None:
        color = "green" if disk >= 20.0 else ("yellow" if disk >= 5.0 else "red")
        click.secho(f"  Free     : {disk:.1f} GiB", fg=color)
    else:
        click.secho("  Free     : detection failed", fg="yellow")
    click.echo("")

    # Tier verdict
    tier = report["tier"]
    click.secho("━" * 40, fg="cyan")
    if tier == "multi_voice":
        click.secho("  ✓ Tier: MULTI-VOICE", fg="green", bold=True)
        click.echo("    Full pipeline: LLM attribution + XTTS-v2 TTS.")
    elif tier == "narrator_only":
        click.secho("  ⚠ Tier: NARRATOR-ONLY", fg="yellow", bold=True)
        click.echo("    Single voice; VITS engine.  Upgrade RAM for multi-voice.")
    elif tier == "unsupported":
        click.secho("  ✗ Tier: UNSUPPORTED", fg="red", bold=True)
        click.echo("    RAM below 4 GiB floor.  Cannot run narrator-only mode reliably.")
    else:
        click.secho("  ? Tier: UNKNOWN (could not detect RAM)", fg="yellow", bold=True)
    click.secho("━" * 40, fg="cyan")
    click.echo("")


# ---------------------------------------------------------------------------
# castbook ingest
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--project",
    "-p",
    required=True,
    help="Project name.  Output goes to projects/<name>/.",
)
@click.option(
    "--override",
    is_flag=True,
    default=False,
    help="If set, overwrite an existing book.json (re-run is idempotent by default).",
)
def ingest(file: Path, project: str, override: bool) -> None:
    """
    Extract text from FILE and write projects/<project>/book.json.

    Accepts .epub, .txt, .docx, .pdf (text layer only).

    \b
    Example:
        castbook ingest mybook.epub --project dracula
    """
    from ebook2audiobook.chunker.chapter_detector import ChapterDetector
    from ebook2audiobook.ingest import ExtractionError, ScannedPdfError, default_registry

    project_dir = Path("projects") / project
    book_json_path = project_dir / "book.json"
    chapters_txt_path = project_dir / "chapters.txt"

    # Idempotency check.
    if book_json_path.exists() and not override:
        click.echo(f"book.json already exists at '{book_json_path}'. Use --override to re-run.")
        return

    project_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"  Extracting: {file}")
    try:
        registry = default_registry()
        book = registry.extract(file)
    except ScannedPdfError as exc:
        click.secho(f"\n✗ {exc}", fg="red")
        sys.exit(1)
    except ExtractionError as exc:
        click.secho(f"\n✗ Extraction failed: {exc}", fg="red")
        sys.exit(1)

    click.echo("  Running chapter detection …")
    detector = ChapterDetector(project_dir=project_dir)
    book = detector.detect(book)

    click.echo("  Writing book.json …")
    book_json_path.write_text(
        book.model_dump_json(indent=2),
        encoding="utf-8",
    )

    # Human-readable chapter summary.
    lines: list[str] = [f"Book: {book.title}", f"Format: {book.source_format}", ""]
    for ch in book.chapters:
        conf_str = f"{ch.confidence:.2f}"
        title_str = ch.title or "(untitled)"
        para_count = len(ch.paragraphs)
        lines.append(
            f"Chapter {ch.index + 1}: {title_str} (confidence {conf_str}, {para_count} paragraphs)"
        )
        # Show first 2 paragraphs as a preview.
        for p in ch.paragraphs[:2]:
            preview = p.text[:120].replace("\n", " ")
            lines.append(f"  {preview}…")
        lines.append("")

    chapters_txt_path.write_text("\n".join(lines), encoding="utf-8")

    click.secho(
        f"\n✓ Extracted {book.chapter_count} chapter(s), ~{book.word_count} words.",
        fg="green",
    )
    click.echo(f"  book.json   → {book_json_path}")
    click.echo(f"  chapters.txt → {chapters_txt_path}")


# ---------------------------------------------------------------------------
# castbook convert
# ---------------------------------------------------------------------------


@cli.command()
@click.option(
    "--project",
    "-p",
    required=True,
    help="Project name in projects/<project>/ to convert.",
)
@click.option(
    "--engine",
    "-e",
    default="fake",
    type=click.Choice(["fake", "vits", "xtts"], case_sensitive=False),
    help="TTS Engine to use for synthesis (defaults to fake for offline testing).",
)
@click.option(
    "--device",
    default="cpu",
    type=click.Choice(["cpu", "cuda", "mps"], case_sensitive=False),
    help="Device to run TTS on (cpu, cuda, mps). Ignored for fake engine.",
)
def convert(project: str, engine: str, device: str) -> None:
    """
    Run narrator-only conversion from book.json to chapter MP3s and M4B.

    Supports pause and resume safely via projects/<project>/state.sqlite.

    \b
    Example:
        castbook convert --project dracula
    """
    from ebook2audiobook.models.book import Book
    from ebook2audiobook.models_manager.manager import ModelManager
    from ebook2audiobook.orchestrator.pipeline import PipelineOrchestrator
    from ebook2audiobook.tts import get_engine

    project_dir = Path("projects") / project
    book_json_path = project_dir / "book.json"

    if not book_json_path.exists():
        click.secho(
            f"Error: {book_json_path} not found.\n"
            f"Run 'castbook ingest <file> --project {project}' first.",
            fg="red",
        )
        sys.exit(1)

    book = Book.model_validate_json(book_json_path.read_text(encoding="utf-8"))

    model_manager = ModelManager.instance()

    # Initialise selected TTS engine via factory
    try:
        tts_engine = get_engine(
            engine.lower(),
            device=device.lower(),
            model_manager=model_manager,
        )
    except Exception as exc:
        click.secho(f"Error loading TTS engine: {exc}", fg="red")
        sys.exit(1)

    click.echo(f"Starting conversion for '{book.title}' in projects/{project}/")
    click.echo(f"  Chapters : {book.chapter_count}")
    click.echo(f"  Engine   : {tts_engine.engine_name}")
    click.echo(f"  Device   : {device}")
    click.echo("")

    orchestrator = PipelineOrchestrator(
        project_dir=project_dir,
        tts_engine=tts_engine,
        model_manager=model_manager,
    )

    last_reported = 0

    def on_progress(done: int, total: int, eta_sec: float) -> None:
        nonlocal last_reported
        pct = (done / total * 100) if total else 100.0
        # Print progress every segment or on completion
        mins = int(eta_sec // 60)
        secs = int(eta_sec % 60)
        eta_str = f"{mins}m {secs:02d}s" if mins > 0 else f"{secs}s"
        click.echo(f"\r  Progress: [{done}/{total}] {pct:5.1f}% | ETA: {eta_str}   ", nl=False)
        last_reported = done

    job = orchestrator.run_narrator_pipeline(book=book, progress_callback=on_progress)
    click.echo("\n")
    click.secho(f"✓ Conversion complete! Stage status: {job.stage_status.value}", fg="green")
    click.echo(f"  Audio files: {project_dir / 'output'}")


# ---------------------------------------------------------------------------
# castbook attribute
# ---------------------------------------------------------------------------


@cli.command()
@click.option(
    "--project",
    "-p",
    required=True,
    help="Project name in projects/<project>/ to attribute.",
)
@click.option(
    "--model",
    "-m",
    default="llama3.2:3b",
    show_default=True,
    help="Ollama model tag to use for attribution (e.g. llama3.2:3b, mistral:7b).",
)
@click.option(
    "--ollama-url",
    default="http://localhost:11434",
    show_default=True,
    help="Base URL for the local Ollama daemon.",
)
@click.option(
    "--override",
    is_flag=True,
    default=False,
    help="Overwrite an existing attribution.json.",
)
def attribute(project: str, model: str, ollama_url: str, override: bool) -> None:
    """
    Run speaker attribution on projects/<project>/book.json using a local Ollama LLM.

    Reads book.json, processes each paragraph that contains dialogue using AttributionStage,
    and writes projects/<project>/attribution.json with per-span speaker labels.

    Ollama must be running before you call this command:
        ollama serve

    \\b
    Example:
        castbook attribute --project dracula --model llama3.2:3b
    """
    import json

    from ebook2audiobook.attribution import AttributionError, CharacterRegistry, OllamaAttributor
    from ebook2audiobook.attribution.stage import AttributionStage
    from ebook2audiobook.models.book import Book
    from ebook2audiobook.models.job import Job
    from ebook2audiobook.store.db import JobDatabase

    project_dir = Path("projects") / project
    book_json_path = project_dir / "book.json"
    attribution_path = project_dir / "attribution.json"
    db_path = project_dir / "state.sqlite"

    if not book_json_path.exists():
        click.secho(
            f"Error: {book_json_path} not found.\n"
            f"Run 'castbook ingest <file> --project {project}' first.",
            fg="red",
        )
        sys.exit(1)

    if attribution_path.exists() and not override:
        click.echo(
            f"attribution.json already exists at '{attribution_path}'. Use --override to re-run."
        )
        return

    book = Book.model_validate_json(book_json_path.read_text(encoding="utf-8"))
    registry = CharacterRegistry()

    click.echo(f"  Book     : {book.title}")
    click.echo(f"  Chapters : {book.chapter_count}")
    click.echo(f"  Model    : {model}")
    click.echo(f"  Ollama   : {ollama_url}")
    click.echo("")

    try:
        attributor = OllamaAttributor(model=model, base_url=ollama_url)
    except AttributionError as exc:
        click.secho(f"\n✗ {exc}", fg="red")
        sys.exit(1)

    # Use AttributionStage instead of direct loop (unifies CLI and stage paths)
    db = JobDatabase(db_path)
    db.init_schema()
    job = Job(
        id=f"{project}-attribution",
        book_id=book.id,
        project_name=project,
        current_stage="attribution",
    )

    def on_progress(done: int, total: int) -> None:
        pct = (done / total * 100) if total else 100.0
        click.echo(f"\r  Progress: [{done}/{total}] {pct:5.1f}%   ", nl=False)

    stage = AttributionStage(db=db, attributor=attributor, registry=registry)

    try:
        segments = stage.run(book=book, job=job, progress_callback=on_progress)
        click.echo("")
    except Exception as exc:
        click.secho(f"\n✗ Attribution failed: {exc}", fg="red")
        sys.exit(1)

    # Export to attribution.json (legacy format for compatibility)
    # Group segments by chapter and paragraph
    chapters_data = []
    total_spans = 0
    unknown_spans = 0

    for ch_idx, chapter in enumerate(book.chapters):
        paragraphs_data = []
        for para in chapter.paragraphs:
            para_segments = [s for s in segments if s.paragraph_id == para.id]
            spans = [
                {
                    "text": s.text,
                    "speaker": s.speaker,
                    "kind": s.kind.value,
                }
                for s in para_segments
            ]
            paragraphs_data.append({"paragraph_id": para.id, "spans": spans})
            total_spans += len(spans)
            unknown_spans += sum(1 for s in spans if s["speaker"] == "unknown")

        chapters_data.append(
            {
                "chapter_index": ch_idx,
                "title": chapter.title,
                "paragraphs": paragraphs_data,
            }
        )

    result = {
        "book_title": book.title,
        "model": model,
        "characters": registry.summary(),
        "chapters": chapters_data,
        "stats": {
            "total_spans": total_spans,
            "unknown_spans": unknown_spans,
            "unknown_pct": round(unknown_spans / total_spans * 100, 1) if total_spans else 0.0,
        },
    }

    attribution_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    click.echo("")
    click.secho("✓ Attribution complete!", fg="green")
    click.echo(f"  Characters found : {len(registry.character_names())}")
    click.echo(f"  Total spans      : {total_spans}")
    if unknown_spans:
        click.secho(
            f"  Unknown spans    : {unknown_spans} — review attribution.json to assign voices.",
            fg="yellow",
        )
    click.echo(f"  attribution.json → {attribution_path}")


# ---------------------------------------------------------------------------
# castbook cast
# ---------------------------------------------------------------------------


@cli.group()
def cast() -> None:
    """Manage character casting and voice assignment."""
    pass


@cast.command()
@click.option(
    "--project",
    "-p",
    required=True,
    help="Project name in projects/<project>/.",
)
def build(project: str) -> None:
    """
    Build a cast seed list using NER before attribution.

    Runs spaCy NER over the book text to find characters with >=3 mentions.
    Writes projects/<project>/cast_seed.json for manual editing.
    """
    from ebook2audiobook.attribution.cast_builder import build_cast_seed
    from ebook2audiobook.models.book import Book

    project_dir = Path("projects") / project
    book_json_path = project_dir / "book.json"
    cast_seed_path = project_dir / "cast_seed.json"

    if not book_json_path.exists():
        click.secho(
            f"Error: {book_json_path} not found.\n"
            f"Run 'castbook ingest <file> --project {project}' first.",
            fg="red",
        )
        sys.exit(1)

    book = Book.model_validate_json(book_json_path.read_text(encoding="utf-8"))

    click.echo(f"  Building cast seed for: {book.title}")
    click.echo(f"  Chapters: {book.chapter_count}")
    click.echo("")

    try:
        seeds = build_cast_seed(book, min_mentions=3)
    except Exception as exc:
        click.secho(f"Error building cast seed: {exc}", fg="red")
        sys.exit(1)

    # Convert to the format expected by cast_seed.json
    from ebook2audiobook.attribution.cast_builder import CastSeedFile

    seed_data = CastSeedFile(
        characters=[
            {"name": seed.name, "aliases": seed.aliases} for seed in seeds
        ]
    )

    cast_seed_path.write_text(seed_data.model_dump_json(indent=2), encoding="utf-8")

    click.echo("")
    click.secho("✓ Cast seed built!", fg="green")
    click.echo(f"  Characters found: {len(seeds)}")
    for seed in seeds[:10]:  # Show first 10
        click.echo(f"    - {seed.name} ({seed.mention_count} mentions)")
    if len(seeds) > 10:
        click.echo(f"    ... and {len(seeds) - 10} more")
    click.echo(f"  cast_seed.json → {cast_seed_path}")
    click.echo("")
    click.echo("You can now edit cast_seed.json to add nicknames or remove characters.")
    click.echo("Then run: castbook attribute --project <project>")


# ---------------------------------------------------------------------------
# castbook models
# ---------------------------------------------------------------------------


@cli.group()
def models() -> None:
    """Manage TTS and LLM models."""
    pass


@models.command()
def list() -> None:
    """
    List downloaded models from the manifest.

    Verifies files exist on disk before showing them.
    """
    import json
    from pathlib import Path

    from ebook2audiobook.config import get_manifest_path, get_models_dir

    manifest_path = get_manifest_path()
    models_dir = get_models_dir()

    if not manifest_path.exists():
        click.echo("No models manifest found.")
        click.echo("Run 'castbook models download <model>' to download models.")
        return

    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except Exception as exc:
        click.secho(f"Error reading manifest: {exc}", fg="red")
        return

    if not manifest:
        click.echo("No models recorded in manifest.")
        return

    click.echo("")
    click.secho("Downloaded Models:", bold=True)
    click.echo("")

    for model_id, info in manifest.items():
        # Verify file exists on disk
        model_path = Path(info.get("path", ""))
        present = model_path.exists() if model_path else False

        click.echo(f"  {model_id}")
        click.echo(f"    Path     : {model_path}")
        click.echo(f"    Size     : {info.get('size', 'unknown')}")
        click.echo(f"    Licence  : {info.get('licence', 'unknown')}")
        click.echo(f"    Present  : {'yes' if present else 'no'}")
        click.echo(f"    Downloaded: {info.get('downloaded_at', 'unknown')}")
        click.echo("")

    click.echo(f"Manifest: {manifest_path}")
    click.echo(f"Models directory: {models_dir}")


@models.command()
@click.argument("model")
@click.option("--yes", is_flag=True, help="Skip confirmation prompt.")
def download(model: str, yes: bool) -> None:
    """
    Download a TTS or LLM model.

    Available models: vits, xtts, or an Ollama model tag (e.g., llama3.2:3b).

    Prints size and licence before downloading.
    """
    from ebook2audiobook.config import get_manifest_path, get_models_dir

    models_dir = get_models_dir()
    manifest_path = get_manifest_path()

    if model == "vits":
        _download_vits(models_dir, manifest_path, yes)
    elif model == "xtts":
        _download_xtts(models_dir, manifest_path, yes)
    elif model.startswith("llama") or ":" in model:
        _download_ollama(model, yes)
    else:
        click.secho(f"Unknown model: {model}", fg="red")
        click.echo("Available: vits, xtts, or an Ollama model tag (e.g., llama3.2:3b)")
        sys.exit(1)


def _download_vits(models_dir: Path, manifest_path: Path, yes: bool) -> None:
    """Download VITS-VCTK model."""
    # VITS-VCTK approximate size: ~350 MB
    size_str = "~350 MB"
    licence = "MIT/Apache (weights), ODC-By 1.0 (VCTK corpus - attribution required)"

    click.echo("")
    click.secho("VITS-VCTK Model", bold=True)
    click.echo(f"  Size    : {size_str}")
    click.echo(f"  Licence : {licence}")
    click.echo("")

    if not yes:
        if not click.confirm("Download this model?"):
            click.echo("Download cancelled.")
            return

    click.echo("Downloading VITS-VCTK...")
    try:
        from TTS.api import TTS

        # Load model (Coqui will download if not present)
        TTS("tts_models/en/vctk/vits").to("cpu")
        click.secho("✓ VITS-VCTK downloaded successfully.", fg="green")
    except ImportError:
        click.secho("Error: coqui-tts not installed. Run: uv sync --extra tts", fg="red")
        sys.exit(1)
    except Exception as exc:
        click.secho(f"Download failed: {exc}", fg="red")
        sys.exit(1)

    # Update manifest
    _update_manifest("vits", models_dir, manifest_path, size_str, licence)


def _download_xtts(models_dir: Path, manifest_path: Path, yes: bool) -> None:
    """Download XTTS-v2 model."""
    # XTTS-v2 approximate size: >2 GB
    size_str = ">2 GB"
    licence = "Coqui Public Model License - non-commercial"

    click.echo("")
    click.secho("XTTS-v2 Model", bold=True)
    click.echo(f"  Size    : {size_str}")
    click.echo(f"  Licence : {licence}")
    click.echo("")

    if not yes:
        if not click.confirm("Download this model?"):
            click.echo("Download cancelled.")
            return

    click.echo("Downloading XTTS-v2 (this may take a while)...")
    try:
        from TTS.api import TTS

        # Load model (Coqui will download if not present)
        TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cpu")
        click.secho("✓ XTTS-v2 downloaded successfully.", fg="green")
    except ImportError:
        click.secho("Error: coqui-tts not installed. Run: uv sync --extra tts", fg="red")
        sys.exit(1)
    except Exception as exc:
        click.secho(f"Download failed: {exc}", fg="red")
        sys.exit(1)

    # Update manifest
    _update_manifest("xtts", models_dir, manifest_path, size_str, licence)


def _download_ollama(model_tag: str, yes: bool) -> None:
    """Download an Ollama model."""
    import shutil

    if not shutil.which("ollama"):
        click.secho("Ollama not found on PATH.", fg="yellow")
        click.echo("Install Ollama from https://ollama.com or run:")
        click.echo(f"  ollama pull {model_tag}")
        return

    click.echo("")
    click.secho(f"Ollama Model: {model_tag}", bold=True)
    click.echo("")

    if not yes:
        if not click.confirm(f"Pull {model_tag} from Ollama?"):
            click.echo("Pull cancelled.")
            return

    click.echo(f"Pulling {model_tag} from Ollama...")
    try:
        subprocess.run(["ollama", "pull", model_tag], check=True)
        click.secho(f"✓ {model_tag} pulled successfully.", fg="green")
    except subprocess.CalledProcessError as exc:
        click.secho(f"Pull failed: {exc}", fg="red")
        sys.exit(1)


def _update_manifest(
    model_id: str, models_dir: Path, manifest_path: Path, size: str, licence: str
) -> None:
    """Update the models manifest with downloaded model info."""
    import json
    from datetime import UTC, datetime

    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = {}
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)

    manifest[model_id] = {
        "path": str(models_dir),
        "size": size,
        "licence": licence,
        "downloaded_at": datetime.now(UTC).isoformat(),
    }

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    click.echo(f"Manifest updated: {manifest_path}")


if __name__ == "__main__":
    cli()
