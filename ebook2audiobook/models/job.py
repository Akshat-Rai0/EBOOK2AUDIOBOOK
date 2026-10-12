"""
Pydantic models for Job and StageStatus — the orchestration layer's state.

Each conversion is one Job.  The job state is persisted in state.sqlite so a
crashed or stopped run can resume without redoing finished stages.

``StageStatus`` values follow a simple state machine:
  pending → running → done
                   ↘ failed → (retry) → done | failed permanently
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field


class StageStatus(StrEnum):
    """Status of a single pipeline stage within a job."""

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class ConversionMode(StrEnum):
    """
    Hardware-tier selector chosen by the user at upload time.

    narrator_only — single voice, no LLM; runs on the low-resource floor
                    (4 GB RAM / 2 GB VRAM, VITS engine).
    multi_voice   — full pipeline with local LLM attribution; needs ~8 GB RAM.
    """

    NARRATOR_ONLY = "narrator_only"
    MULTI_VOICE = "multi_voice"


def _utc_now() -> datetime:
    """Return timezone-aware current UTC time."""
    return datetime.now(UTC)


class Job(BaseModel):
    """
    One conversion job: from file upload to audiobook output.

    ``id`` is a UUID generated when the job is created.
    ``project_name`` maps to the folder ``projects/<project_name>/``.
    ``current_stage`` is updated as each pipeline stage starts and finishes.
    """

    id: str = Field(..., description="UUID.")
    book_id: str = Field(..., description="UUID of the Book being converted.")
    project_name: str = Field(..., description="Maps to projects/<project_name>/.")
    mode: ConversionMode = ConversionMode.MULTI_VOICE
    current_stage: str = Field(
        default="ingest",
        description="Name of the stage currently executing.",
    )
    stage_status: StageStatus = StageStatus.PENDING
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)
    error: str | None = Field(
        default=None,
        description="Last error message, if any.",
    )
    schema_version: str = Field(default="1.0")

    @classmethod
    def schema_path(cls) -> Path:
        return Path("docs/schemas/job.schema.json")
