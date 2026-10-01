"""
Pydantic models for the Book ingestion layer.

Data contract (version 1.0):
  Paragraph  — one block of text with one semantic purpose.
  Chapter    — ordered list of paragraphs, detected by format-specific heuristics.
  Book       — the complete parsed representation of an ebook file.

The ``original_text`` field on Paragraph preserves the raw text before any
cleaning step, so the pipeline is always reversible and auditable.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class Paragraph(BaseModel):
    """
    One block of text within a chapter.

    ``id`` follows the pattern ``c<chapter_index>-p<paragraph_index>``, e.g.
    ``c01-p003`` for the third paragraph of chapter one.  Indices are
    zero-padded to three digits so lexicographic order matches reading order.
    """

    id: str = Field(..., description="Stable identifier: c<CC>-p<PPP>")
    text: str = Field(..., description="Cleaned text, ready for NLP.")
    original_text: str = Field(..., description="Raw text before any cleaning.")
    is_dialogue: bool = Field(
        default=False,
        description="True when the paragraph contains at least one quoted speech span.",
    )

    model_config = {"frozen": False}


class Chapter(BaseModel):
    """
    One chapter extracted from the ebook.

    ``confidence`` (0–1) reflects how certain the detector is that this is a
    real chapter boundary.  A score below 0.5 means the boundary was found by
    weak heuristics; the user can override with ``chapters_override.json``.
    """

    id: str = Field(..., description="Stable identifier: c<CC>")
    index: int = Field(..., description="0-based reading order.")
    title: str = Field(..., description="Chapter heading as extracted from the file.")
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Chapter-detection confidence (1.0 = structural, <0.5 = heuristic).",
    )
    paragraphs: list[Paragraph] = Field(default_factory=list)


class Book(BaseModel):
    """
    The complete parsed representation of one ebook.

    Written to ``projects/<name>/book.json`` after the ingest stage.
    Downstream agents treat this as their authoritative source of truth.
    """

    id: str = Field(..., description="UUID generated at extraction time.")
    title: str = Field(default="", description="Book title from metadata or filename.")
    author: str = Field(default="", description="Author from metadata, empty if unknown.")
    language: str = Field(
        default="en",
        description="BCP-47 language tag detected or defaulted.",
    )
    source_path: str = Field(..., description="Absolute path to the original file.")
    source_format: str = Field(
        ...,
        description="One of: epub, txt, docx, pdf.",
    )
    chapters: list[Chapter] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    schema_version: str = Field(default="1.0")

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def chapter_count(self) -> int:
        return len(self.chapters)

    @property
    def word_count(self) -> int:
        return sum(len(p.text.split()) for ch in self.chapters for p in ch.paragraphs)

    def get_chapter(self, index: int) -> Chapter | None:
        """Return chapter by 0-based index, or None."""
        for ch in self.chapters:
            if ch.index == index:
                return ch
        return None

    @classmethod
    def schema_path(cls) -> Path:
        """Canonical path for the exported JSON schema."""
        return Path("docs/schemas/book.schema.json")
