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
    type=click.Choice(["fake", "vits"], case_sensitive=False),
    help="TTS Engine to use for synthesis (defaults to fake for offline testing).",
)
def convert(project: str, engine: str) -> None:
    """
    Run narrator-only conversion from book.json to chapter MP3s and M4B.

    Supports pause and resume safely via projects/<project>/state.sqlite.

    \b
    Example:
        castbook convert --project dracula
    """
    from ebook2audiobook.models.book import Book
    from ebook2audiobook.orchestrator.pipeline import PipelineOrchestrator
    from ebook2audiobook.tts.fake_tts import FakeTTS

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

    # Initialise selected TTS engine
    if engine.lower() == "fake":
        tts_engine = FakeTTS()
    else:
        click.secho(
            "VITS engine integration benchmark scheduled for Week 6. Using FakeTTS for testing.",
            fg="yellow",
        )
        tts_engine = FakeTTS()

    click.echo(f"Starting conversion for '{book.title}' in projects/{project}/")
    click.echo(f"  Chapters : {book.chapter_count}")
    click.echo(f"  Engine   : {tts_engine.engine_name}")
    click.echo("")

    orchestrator = PipelineOrchestrator(project_dir=project_dir, tts_engine=tts_engine)

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


if __name__ == "__main__":
    cli()
