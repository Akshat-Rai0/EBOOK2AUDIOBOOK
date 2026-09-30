"""
TXT extractor — extracts Book from plain-text files.

Chapter detection for TXT relies on heuristics (see ``chapter_detector.py``).
The extractor delegates cleaning and chapter detection to shared modules.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from ebook2audiobook.ingest.cleaning import clean, detect_dialogue
from ebook2audiobook.ingest.extractor import Extractor
from ebook2audiobook.models.book import Book, Chapter, Paragraph


class TxtExtractor(Extractor):
    """Extracts a Book from a UTF-8 plain-text file."""

    @property
    def supported_extensions(self) -> list[str]:
        return [".txt"]

    def can_handle(self, path: Path) -> bool:
        return path.suffix.lower() == ".txt"

    def extract(self, path: Path) -> Book:
        """
        Read *path*, split into paragraphs, and return a Book.

        Chapter splitting is handled by ``ChapterDetector`` in the chunker
        layer; here we produce a single chapter with all paragraphs so that
        the ingest stage is format-agnostic and the chunker can always be
        tested independently.

        (The CLI pipeline calls chunker after ingest, so tests that call
        ``TxtExtractor.extract`` directly will see one chapter.)
        """
        raw = path.read_text(encoding="utf-8", errors="replace")
        paragraphs = self._split_paragraphs(raw)

        chapter = Chapter(
            id="c00",
            index=0,
            title="",  # ChapterDetector will fill this
            confidence=0.0,  # no structure info; detector will re-score
            paragraphs=paragraphs,
        )

        return Book(
            id=str(uuid.uuid4()),
            title=path.stem,
            source_path=str(path),
            source_format="txt",
            chapters=[chapter],
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split_paragraphs(raw: str) -> list[Paragraph]:
        """
        Split raw text into paragraphs (separated by one or more blank lines).
        Returns a list of cleaned Paragraph objects with stable ids.
        """
        blocks = [b.strip() for b in raw.split("\n\n") if b.strip()]
        paragraphs: list[Paragraph] = []
        for i, block in enumerate(blocks):
            original = block
            cleaned = clean(block, strip_headers=True)
            if not cleaned:
                continue
            paragraphs.append(
                Paragraph(
                    id=f"c00-p{i:03d}",
                    text=cleaned,
                    original_text=original,
                    is_dialogue=detect_dialogue(cleaned),
                )
            )
        return paragraphs
