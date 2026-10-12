"""Tests for cast schema migration, export, and hardware tier refusal."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from ebook2audiobook.models.cast import Cast
from ebook2audiobook.models.job import ConversionMode
from ebook2audiobook.orchestrator.pipeline import PipelineOrchestrator
from ebook2audiobook.tts.fake_tts import FakeTTS


class TestCastMigrationAndSchema:
    def test_cast_schema_export_matches_model(self):
        schema_file = Cast.schema_path()
        assert schema_file.exists(), f"{schema_file} does not exist"

        saved_schema = json.loads(schema_file.read_text(encoding="utf-8"))
        current_schema = Cast.model_json_schema()

        assert saved_schema["title"] == current_schema["title"]
        assert "properties" in saved_schema
        assert "characters" in saved_schema["properties"]

    def test_legacy_cast_v1_0_migration(self):
        """A legacy cast JSON missing voice_assignments or schema_version migrates cleanly."""
        legacy_data = {
            "characters": [
                {
                    "name": "Old Mira",
                    "aliases": ["Mira"],
                    "line_count": 5,
                }
            ]
        }
        migrated = Cast.model_validate(legacy_data)
        assert len(migrated.characters) == 1
        char = migrated.characters[0]
        assert char.name == "Old Mira"
        assert char.voice_assignments == {}
        assert char.user_locked is False
        assert migrated.schema_version == "1.1"

    def test_low_ram_tier_refuses_multivoice_mode(self, tmp_path: Path):
        project_dir = tmp_path / "proj"
        tts = FakeTTS()

        # Mock ram_gib to 4.0 GB -> tier is 'narrator_only'
        with patch("ebook2audiobook.utils.env.get_ram_gib", return_value=4.0):
            # MULTI_VOICE mode must raise RuntimeError
            with pytest.raises(RuntimeError, match="insufficient for multi-voice mode"):
                PipelineOrchestrator(
                    project_dir=project_dir,
                    tts_engine=tts,
                    mode=ConversionMode.MULTI_VOICE,
                )

            # NARRATOR_ONLY mode must succeed cleanly
            orch = PipelineOrchestrator(
                project_dir=project_dir,
                tts_engine=tts,
                mode=ConversionMode.NARRATOR_ONLY,
            )
            assert orch.mode == ConversionMode.NARRATOR_ONLY
