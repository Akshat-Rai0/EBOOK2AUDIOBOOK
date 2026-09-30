"""
ChapterDetector — post-processes a Book to identify chapter boundaries.

The four extractors (TXT, DOCX, EPUB, PDF) each return a Book with their best
guess at chapters.  This module refines those guesses using:

1. ``chapters_override.json`` — user-supplied ground truth (always wins).
2. Format-specific heuristics (already applied by the extractor).
3. TXT-specific paragraph-level patterns when the extractor produced one big
   chapter (common for plain-text files with no blank-line-separated headings).

**Confidence thresholds:**
  ≥ 0.85 → structural (EPUB spine, DOCX Heading 1)  — trust fully.
  ≥ 0.60 → heuristic (PDF font change, ALL-CAPS)     — show to user for review.
  < 0.60 → weak (no patterns found)                  — always show for review.

**chapters_override.json format:**
```json
[
  {"index": 0, "title": "Prologue", "start_paragraph_id": "c00-p000"},
  {"index": 1, "title": "Chapter One", "start_paragraph_id": "c00-p014"}
]
```
When this file is present in the project folder, it fully replaces the detected
chapter structure.  This is the escape hatch for books with unusual formatting.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from ebook2audiobook.models.book import Book, Chapter, Paragraph

logger = logging.getLogger(__name__)

# Pattern: "Chapter 1", "Chapter One", "Chapter I", "CHAPTER 12", "Chapter 1: Title", etc.
_RE_CHAPTER_LINE = re.compile(
    r"^\s*(chapter\s+(\d+|[ivxlcdm]+|one|two|three|four|five|six|seven|eight|nine|ten"
    r"|eleven|twelve|[a-z\-]+)\b.*|prologue\b.*|epilogue\b.*|part\s+\d+\b.*)\s*$",
    re.IGNORECASE,
)

# ALL-CAPS line, 4–60 chars, no lowercase letters.
_RE_ALL_CAPS = re.compile(r"^[A-Z][A-Z\s\d\-:']{3,59}$")


class ChapterDetector:
    """
    Refines chapter detection in a Book and applies the override file if present.

    Usage::

        detector = ChapterDetector(project_dir=Path("projects/mybook"))
        book = detector.detect(book)
    """

    def __init__(self, project_dir: Path | None = None) -> None:
        """
        Parameters
        ----------
        project_dir:
            The project folder.  If provided and a ``chapters_override.json``
            exists inside it, the override is applied automatically.
        """
        self.project_dir = project_dir

    def detect(self, book: Book) -> Book:
        """
        Run chapter detection and return the updated Book (mutates in place).

        Steps:
        1. If override file exists, apply it and return early.
        2. If the book already has high-confidence chapters (≥ 0.85), return
           as-is (EPUB / DOCX structural chapters need no further processing).
        3. Otherwise run heuristic re-splitting on TXT/PDF books.
        """
        override = self._load_override()
        if override:
            logger.info("Applying chapters_override.json (%d entries)", len(override))
            return self._apply_override(book, override)

        # Check if already well-structured.
        if book.chapters and all(ch.confidence >= 0.85 for ch in book.chapters):
            logger.debug("All chapters have confidence ≥ 0.85; skipping re-detection.")
            return book

        # For single-chapter books (TXT fallback), try heuristic split.
        if len(book.chapters) == 1 and book.chapters[0].confidence < 0.85:
            logger.info("Single low-confidence chapter detected; running heuristic split.")
            book.chapters = self._heuristic_split(book.chapters[0].paragraphs)

        return book

    # ------------------------------------------------------------------
    # Override file handling
    # ------------------------------------------------------------------

    def _load_override(self) -> list[dict] | None:
        """Return parsed override data, or None if the file does not exist."""
        if not self.project_dir:
            return None
        override_path = self.project_dir / "chapters_override.json"
        if not override_path.exists():
            return None
        try:
            data = json.loads(override_path.read_text())
            if not isinstance(data, list):
                logger.warning("chapters_override.json must be a JSON array; ignoring.")
                return None
            return data
        except json.JSONDecodeError as exc:
            logger.warning("chapters_override.json is invalid JSON: %s; ignoring.", exc)
            return None

    @staticmethod
    def _apply_override(book: Book, override: list[dict]) -> Book:
        """
        Rebuild chapter structure from the override list.

        Each override entry must have ``index``, ``title``, and
        ``start_paragraph_id``.  Paragraphs are re-assigned based on the
        start_paragraph_id boundaries.
        """
        # Collect all paragraphs in reading order.
        all_paragraphs: list[Paragraph] = []
        for ch in sorted(book.chapters, key=lambda c: c.index):
            all_paragraphs.extend(ch.paragraphs)

        # Build a lookup: paragraph_id → flat index.
        id_to_idx = {p.id: i for i, p in enumerate(all_paragraphs)}

        new_chapters: list[Chapter] = []
        override_sorted = sorted(override, key=lambda e: e.get("index", 0))

        for i, entry in enumerate(override_sorted):
            title = entry.get("title", "")
            start_id = entry.get("start_paragraph_id", "")
            start_idx = id_to_idx.get(start_id, 0)

            # End is the start of the next override entry or the end of the book.
            if i + 1 < len(override_sorted):
                next_id = override_sorted[i + 1].get("start_paragraph_id", "")
                end_idx = id_to_idx.get(next_id, len(all_paragraphs))
            else:
                end_idx = len(all_paragraphs)

            paragraphs = all_paragraphs[start_idx:end_idx]
            # Re-index paragraph ids for this chapter.
            for j, p in enumerate(paragraphs):
                p.id = f"c{i:02d}-p{j:03d}"

            new_chapters.append(
                Chapter(
                    id=f"c{i:02d}",
                    index=i,
                    title=title,
                    confidence=1.0,  # user-supplied = full trust
                    paragraphs=paragraphs,
                )
            )

        book.chapters = new_chapters
        return book

    # ------------------------------------------------------------------
    # Heuristic split (TXT / weak PDF)
    # ------------------------------------------------------------------

    def _heuristic_split(self, paragraphs: list[Paragraph]) -> list[Chapter]:
        """
        Split a flat list of paragraphs into chapters based on heading patterns.

        A paragraph is treated as a chapter heading if:
        - It matches the "Chapter N" regex, OR
        - It is ≤ 60 characters, ALL-CAPS, and appears on its own short line.
        """
        chapters: list[Chapter] = []
        chapter_index = 0
        current_title = ""
        current_paragraphs: list[Paragraph] = []
        confidence = 0.40

        def flush() -> None:
            nonlocal chapter_index
            if current_paragraphs:
                for j, p in enumerate(current_paragraphs):
                    p.id = f"c{chapter_index:02d}-p{j:03d}"
                chapters.append(
                    Chapter(
                        id=f"c{chapter_index:02d}",
                        index=chapter_index,
                        title=current_title,
                        confidence=confidence,
                        paragraphs=list(current_paragraphs),
                    )
                )
                chapter_index += 1

        for para in paragraphs:
            text = para.text.strip()
            first_line = text.split("\n")[0].strip()

            if _RE_CHAPTER_LINE.match(first_line):
                flush()
                current_title = first_line
                current_paragraphs = []
                confidence = 0.75
            elif _RE_ALL_CAPS.match(first_line) and len(first_line) <= 60:
                flush()
                current_title = first_line
                current_paragraphs = []
                confidence = 0.60
            else:
                current_paragraphs.append(para)

        flush()

        # No headings found at all → one chapter with original paragraphs.
        if not chapters:
            for j, p in enumerate(paragraphs):
                p.id = f"c00-p{j:03d}"
            chapters.append(
                Chapter(
                    id="c00",
                    index=0,
                    title="",
                    confidence=0.40,
                    paragraphs=paragraphs,
                )
            )

        return chapters
