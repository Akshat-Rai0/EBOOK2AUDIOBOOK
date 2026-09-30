"""
Tests for castbook doctor CLI command.

We mock the environment detection functions so the test never touches real
hardware, never downloads anything, and runs in < 1 ms.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from click.testing import CliRunner

from ebook2audiobook.cli.main import cli


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


def _mock_report(tier: str = "multi_voice", ram: float = 16.0) -> dict:
    return {
        "python": {
            "version": "3.11.0",
            "platform": "macOS-15.0",
            "executable": "/usr/bin/python3",
        },
        "ram_gib": ram,
        "gpu": {"backend": "mps", "vram_gib": None, "device_name": "Apple Silicon (MPS)"},
        "disk_free_gib": 150.0,
        "ffmpeg": {
            "installed": True,
            "version": "ffmpeg version 7.0",
            "path": "/usr/local/bin/ffmpeg",
        },
        "tier": tier,
    }


class TestDoctorCommand:
    def test_doctor_runs(self, runner):
        """doctor command exits without error when all checks pass."""
        with patch("ebook2audiobook.cli.main.run_doctor", return_value=_mock_report()):
            result = runner.invoke(cli, ["doctor"])
        assert result.exit_code == 0, result.output

    def test_doctor_shows_tier_multi_voice(self, runner):
        with patch("ebook2audiobook.cli.main.run_doctor", return_value=_mock_report("multi_voice")):
            result = runner.invoke(cli, ["doctor"])
        assert "MULTI-VOICE" in result.output

    def test_doctor_shows_tier_narrator_only(self, runner):
        report = _mock_report("narrator_only", ram=5.0)
        with patch("ebook2audiobook.cli.main.run_doctor", return_value=report):
            result = runner.invoke(cli, ["doctor"])
        assert "NARRATOR-ONLY" in result.output

    def test_doctor_shows_tier_unsupported(self, runner):
        report = _mock_report("unsupported", ram=2.0)
        with patch("ebook2audiobook.cli.main.run_doctor", return_value=report):
            result = runner.invoke(cli, ["doctor"])
        assert "UNSUPPORTED" in result.output

    def test_doctor_shows_ffmpeg_missing(self, runner):
        report = _mock_report()
        report["ffmpeg"] = {"installed": False, "version": None, "path": None}
        with patch("ebook2audiobook.cli.main.run_doctor", return_value=report):
            result = runner.invoke(cli, ["doctor"])
        assert "NOT FOUND" in result.output


class TestIngestCommand:
    def test_ingest_txt(self, runner, txt_file, tmp_path, monkeypatch):
        """ingest command with a TXT file should produce book.json."""
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(
            cli,
            ["ingest", str(txt_file), "--project", "test_project"],
        )
        assert result.exit_code == 0, result.output
        assert "book.json" in result.output or "✓" in result.output

    def test_ingest_idempotent(self, runner, txt_file, tmp_path, monkeypatch):
        """Running ingest twice without --override should short-circuit on second run."""
        monkeypatch.chdir(tmp_path)
        runner.invoke(cli, ["ingest", str(txt_file), "--project", "mybook"])
        result2 = runner.invoke(cli, ["ingest", str(txt_file), "--project", "mybook"])
        assert "already exists" in result2.output

    def test_ingest_scanned_pdf_exits_nonzero(
        self, runner, scanned_pdf_file, tmp_path, monkeypatch
    ):
        """Scanned PDF must cause a non-zero exit with a helpful message."""
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(
            cli,
            ["ingest", str(scanned_pdf_file), "--project", "scan_test"],
        )
        assert result.exit_code != 0
        assert "scanned" in result.output.lower() or "text layer" in result.output.lower()
