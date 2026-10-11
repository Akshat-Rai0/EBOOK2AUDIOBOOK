"""Tests for Cast CLI review commands and service functions."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from ebook2audiobook.casting.service import (
    approve_cast,
    load_cast,
    merge_characters,
    rename_character,
    set_character_voice,
    suggest_cast,
    undo_last_operation,
    verify_cast_approval,
)
from ebook2audiobook.cli.main import cli
from ebook2audiobook.models.segment import Segment, SegmentKind, SegmentStatus
from ebook2audiobook.store.db import JobDatabase


def _setup_project(project_dir: Path) -> None:
    project_dir.mkdir(parents=True, exist_ok=True)
    db = JobDatabase(project_dir / "state.sqlite")
    db.init_schema()

    segments = [
        Segment(
            id="c00-p000-s00",
            chapter=0,
            paragraph_id="c00-p000",
            speaker_id="Mira",
            kind=SegmentKind.DIALOGUE,
            text="Hello there.",
            confidence=0.95,
            status=SegmentStatus.PENDING,
        ),
        Segment(
            id="c00-p000-s01",
            chapter=0,
            paragraph_id="c00-p000",
            speaker_id="Tomas",
            kind=SegmentKind.DIALOGUE,
            text="General Kenobi.",
            confidence=0.70,  # below 0.80
            status=SegmentStatus.PENDING,
        ),
        Segment(
            id="c00-p000-s02",
            chapter=0,
            paragraph_id="c00-p000",
            speaker_id="unknown",
            kind=SegmentKind.DIALOGUE,
            text="Who speaks?",
            confidence=0.50,
            status=SegmentStatus.PENDING,
        ),
    ]
    db.register_segments(segments)


class TestCastService:
    def test_suggest_and_approve_cycle(self, tmp_path: Path):
        project_dir = tmp_path / "proj"
        _setup_project(project_dir)

        cast = suggest_cast(project_dir, engine="fake")
        assert len(cast.characters) >= 2
        assert (project_dir / "cast.json").exists()

        assert not verify_cast_approval(project_dir)
        cast_hash = approve_cast(project_dir)
        assert len(cast_hash) == 64
        assert verify_cast_approval(project_dir)

        # Mutating cast.json invalidates approval
        char = cast.characters[0]
        char.line_count += 5
        (project_dir / "cast.json").write_text(cast.model_dump_json(indent=2), encoding="utf-8")
        assert not verify_cast_approval(project_dir)

    def test_approve_auto_routes_unknown_to_narrator(self, tmp_path: Path):
        project_dir = tmp_path / "proj"
        _setup_project(project_dir)

        approve_cast(project_dir, auto=True)
        db = JobDatabase(project_dir / "state.sqlite")
        segs = db.get_all_segments()
        unknowns = [s for s in segs if s.speaker_id == "unknown"]
        assert len(unknowns) == 0

    def test_set_character_voice_locks_character(self, tmp_path: Path):
        project_dir = tmp_path / "proj"
        _setup_project(project_dir)
        suggest_cast(project_dir, engine="fake")

        char = set_character_voice(project_dir, "Mira", "fake-char-2", engine="fake")
        assert char.user_locked is True
        assert char.voice is not None
        assert char.voice.voice_id == "fake-char-2"

    def test_merge_characters_transactional(self, tmp_path: Path):
        project_dir = tmp_path / "proj"
        _setup_project(project_dir)
        suggest_cast(project_dir, engine="fake")

        winner, _ = merge_characters(project_dir, "Tomas", "Mira")
        assert winner.name == "Mira"

        cast = load_cast(project_dir)
        assert cast is not None
        # Tomas is now merged as an alias of Mira
        mira = cast.get_character("Mira")
        assert mira is not None
        assert "Tomas" in mira.aliases
        # The list of canonical characters has only Mira, not a separate Tomas
        assert [c.name for c in cast.characters] == ["Mira"]

        # Verify SQLite updated
        db = JobDatabase(project_dir / "state.sqlite")
        segs = db.get_all_segments()
        assert not any(s.speaker_id == "Tomas" for s in segs)

        # Undo merge
        rec = undo_last_operation(project_dir)
        assert rec is not None
        cast_after_undo = load_cast(project_dir)
        assert cast_after_undo is not None
        assert cast_after_undo.get_character("Tomas") is not None

    def test_merge_both_user_locked_aborts(self, tmp_path: Path):
        import pytest

        project_dir = tmp_path / "proj"
        _setup_project(project_dir)
        suggest_cast(project_dir, engine="fake")

        set_character_voice(project_dir, "Mira", "fake-char-1")
        set_character_voice(project_dir, "Tomas", "fake-char-2")

        with pytest.raises(ValueError, match="Both .* are user_locked"):
            merge_characters(project_dir, "Tomas", "Mira")

    def test_rename_character(self, tmp_path: Path):
        project_dir = tmp_path / "proj"
        _setup_project(project_dir)
        suggest_cast(project_dir, engine="fake")

        char = rename_character(project_dir, "Mira", "Mira Vane")
        assert char.display_name == "Mira Vane"


class TestCastCLI:
    def test_cast_cli_workflow(self, tmp_path: Path, monkeypatch):
        # Point projects/ to tmp_path
        monkeypatch.chdir(tmp_path)
        project_dir = tmp_path / "projects" / "test_proj"
        _setup_project(project_dir)

        runner = CliRunner()

        # 1. suggest
        res = runner.invoke(cli, ["cast", "suggest", "-p", "test_proj"])
        assert res.exit_code == 0
        assert "Suggested voices assigned" in res.output

        # 2. show
        res = runner.invoke(cli, ["cast", "show", "-p", "test_proj"])
        assert res.exit_code == 0
        assert "Mira" in res.output

        # 3. voice
        res = runner.invoke(cli, ["cast", "voice", "test_proj", "Mira", "fake-char-2"])
        assert res.exit_code == 0
        assert "[user_locked]" in res.output

        # 4. sample
        res = runner.invoke(cli, ["cast", "sample", "test_proj", "Mira"])
        assert res.exit_code == 0
        assert "Sample generated" in res.output
        assert (project_dir / "samples" / "mira.wav").exists()

        # 5. review (interactive input)
        # We have Tomas (conf 0.70 < 0.80) and unknown (conf 0.50 < 0.80)
        # Input 'Tomas' for first prompt, and 'Old Man' for second prompt
        res = runner.invoke(cli, ["cast", "review", "-p", "test_proj"], input="Tomas\nOld Man\n")
        assert res.exit_code == 0
        assert "Updated → Old Man" in res.output

        # 6. approve
        res = runner.invoke(cli, ["cast", "approve", "-p", "test_proj"])
        assert res.exit_code == 0
        assert "Cast approved!" in res.output
