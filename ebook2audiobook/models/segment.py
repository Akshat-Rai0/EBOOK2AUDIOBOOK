"""
Pydantic models for Segment — the atomic unit of the TTS synthesis pipeline.

A Segment is one span of text spoken by exactly one speaker.  It is created
from a Paragraph by the attribution agent and consumed by the TTS agent.

Data contract (version 1.2 - M4):

  id                  — ``c<CC>-p<PPP>-s<SS>`` (chapter / paragraph / segment indices).
  speaker_id          — canonical character ID, ``"narrator"``, or ``"unknown"``.
  speaker             — alias for speaker_id (retained for backward compatibility).
  kind                — dialogue | narration | thought  (drives voice-acting style).
  confidence          — 0.0 to 1.0 (LLM attribution certainty, 1.0 for rule/user).
  source              — llm | rule | user  (user edits are authoritative and locked).
  evidence            — optional short text snippet explaining attribution.
  voice_hash          — SHA-256 hash of (text + voice_id) for selective cache invalidation.
  continues_previous  — True if this dialogue segment continues a multi-paragraph quote.
  status              — pending → running → done | failed | skipped.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Self

from pydantic import BaseModel, Field, model_validator


class SegmentKind(StrEnum):
    """Semantic kind of a spoken segment."""

    DIALOGUE = "dialogue"
    NARRATION = "narration"
    THOUGHT = "thought"


class SegmentStatus(StrEnum):
    """
    Processing status stored in state.sqlite and mirrored in segments.json.

    Think of it like a to-do list: pending → in progress (running) → done or
    failed.  A failed segment is retried up to two times, then marked skipped
    so the rest of the book can still be produced.
    """

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


class SegmentSource(StrEnum):
    """Origin of the speaker attribution assignment."""

    LLM = "llm"
    RULE = "rule"
    USER = "user"  # Authoritative: a re-run NEVER overwrites source=user


class Segment(BaseModel):
    """
    One unit of speech: text + speaker + kind + processing state.

    The TTS agent reads ``text``, looks up the voice for ``speaker_id`` in the
    Cast, and writes a WAV file to ``audio_path``.
    """

    id: str = Field(..., description="Stable identifier: c<CC>-p<PPP>-s<SS>")
    chapter: int = Field(..., description="0-based chapter index.")
    paragraph_id: str = Field(..., description="Parent paragraph id: c<CC>-p<PPP>")
    speaker_id: str = Field(
        default="narrator",
        description="Canonical character identifier or 'narrator' / 'unknown'.",
    )
    speaker: str = Field(
        default="narrator",
        description="Legacy alias for speaker_id.",
    )
    kind: SegmentKind = SegmentKind.NARRATION
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Attribution confidence (0.0 to 1.0).",
    )
    source: SegmentSource = Field(
        default=SegmentSource.RULE,
        description="Origin: llm, rule, or user.",
    )
    evidence: str | None = Field(
        default=None,
        description="Optional short rationale or snippet from attribution.",
    )
    voice_hash: str | None = Field(
        default=None,
        description="SHA-256 hash of text + voice_id for cache validation.",
    )
    continues_previous: bool = Field(
        default=False,
        description="True if this dialogue segment continues a multi-paragraph quote.",
    )
    text: str = Field(..., description="Text to synthesise, after TTS-safe cleaning.")
    status: SegmentStatus = SegmentStatus.PENDING
    audio_path: str | None = Field(
        default=None,
        description="Absolute path to the generated WAV file, set when status=done.",
    )
    retry_count: int = Field(default=0, description="Number of synthesis retries so far.")
    schema_version: str = Field(default="1.2")

    @model_validator(mode="after")
    def _sync_speaker_fields(self) -> Self:
        """Keep speaker and speaker_id synchronized."""
        if self.speaker != "narrator" and self.speaker_id == "narrator":
            self.speaker_id = self.speaker
        elif self.speaker_id != "narrator" and self.speaker == "narrator":
            self.speaker = self.speaker_id
        return self

    @property
    def is_user_locked(self) -> bool:
        """True if this segment was manually set by the user and must not be overwritten."""
        return self.source == SegmentSource.USER

    @classmethod
    def schema_path(cls) -> Path:
        return Path("docs/schemas/segment.schema.json")
