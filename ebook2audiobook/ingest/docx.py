"""
DOCX extractor — extracts Book from Microsoft Word .docx files.

**Chapter detection strategy for DOCX:**
DOCX files use named paragraph styles.  A paragraph styled ``Heading 1`` or
``Heading 2`` is treated as a chapter boundary with high confidence (0.90).
This is much more reliable than heuristic text matching.

**Why python-docx and not docx2txt?**
``python-docx`` exposes paragraph styles, which is necessary to identify
chapter headings.  ``docx2txt`` strips all structure and would force us back
to TXT-style heuristics.

Confidence scores:
  Heading 1 found     → 0.90
  Heading 2 found     → 0.80
  No heading styles   → 0.40 (fall back to TXT heuristics in ChapterDetector)
"""

from __future__ import annotations

import uuid
from pathlib import Path

from ebook2audiobook.ingest.cleaning import clean, detect_dialogue
from ebook2audiobook.ingest.extractor import ExtractionError, Extractor
from ebook2audiobook.models.book import Book, Chapter, Paragraph

# Guard against missing optional dependency (python-docx) at import time.
try:
    from docx import Document as DocxDocument
    from docx.oxml.ns import qn  # noqa: F401  (used for future table detection)

    _DOCX_AVAILABLE = True
except ImportError:  # pragma: no cover
    _DOCX_AVAILABLE = False


class DocxExtractor(Extractor):
    """Extracts a Book from a .docx file using paragraph heading styles."""

    @property
    def supported_extensions(self) -> list[str]:
        return [".docx"]

    def can_handle(self, path: Path) -> bool:
        return path.suffix.lower() == ".docx"

    def extract(self, path: Path) -> Book:
        if not _DOCX_AVAILABLE:
            raise ExtractionError("python-docx is not installed. Run: uv add python-docx")

        try:
            doc = DocxDocument(str(path))
        except Exception as exc:
            raise ExtractionError(f"Could not open DOCX file: {exc}") from exc

        chapters = self._build_chapters(doc)
        if not chapters:
            raise ExtractionError(f"No content found in DOCX file: {path}")

        # Determine overall detection confidence from the first chapter.
        # (All chapters share the same detection method for this file.)
        confidence = chapters[0].confidence

        return Book(
            id=str(uuid.uuid4()),
            title=self._extract_title(doc, path),
            source_path=str(path),
            source_format="docx",
            chapters=chapters,
            metadata={"docx_confidence": confidence},
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    _HEADING_STYLES = {
        "heading 1": 0.90,
        "heading 2": 0.80,
        "heading 3": 0.70,
    }

    def _build_chapters(self, doc: DocxDocument) -> list[Chapter]:  # type: ignore[name-defined]
        """
        Walk all paragraphs in the document.  Each Heading-styled paragraph
        starts a new chapter; body paragraphs accumulate in the current chapter.
        """
        chapters: list[Chapter] = []
        current_paragraphs: list[Paragraph] = []
        current_title = ""
        chapter_index = 0
        confidence = 0.40  # default: no headings found

        for para in doc.paragraphs:
            style_name = (para.style.name or "").lower()
            text = para.text.strip()

            # Is this a chapter heading?
            heading_conf = self._heading_styles_lower(style_name)
            if heading_conf is not None and text:
                # Flush the current chapter before starting a new one.
                if chapter_index > 0 or current_paragraphs:
                    chapters.append(
                        self._make_chapter(
                            chapter_index, current_title, current_paragraphs, heading_conf
                        )
                    )
                    chapter_index += 1
                current_title = text
                current_paragraphs = []
                confidence = heading_conf
                continue

            # Regular body paragraph.
            if not text:
                continue
            original = text
            cleaned = clean(text)
            current_paragraphs.append(
                Paragraph(
                    id=f"c{chapter_index:02d}-p{len(current_paragraphs):03d}",
                    text=cleaned,
                    original_text=original,
                    is_dialogue=detect_dialogue(cleaned),
                )
            )

        # Flush the last chapter.
        if current_title or current_paragraphs:
            chapters.append(
                self._make_chapter(chapter_index, current_title, current_paragraphs, confidence)
            )

        # If no headings were found at all, wrap everything into chapter 0.
        if not chapters:
            all_paras: list[Paragraph] = []
            for para in doc.paragraphs:
                text = para.text.strip()
                if text:
                    original = text
                    cleaned = clean(text)
                    all_paras.append(
                        Paragraph(
                            id=f"c00-p{len(all_paras):03d}",
                            text=cleaned,
                            original_text=original,
                            is_dialogue=detect_dialogue(cleaned),
                        )
                    )
            chapters = [Chapter(id="c00", index=0, title="", confidence=0.40, paragraphs=all_paras)]

        return chapters

    def _heading_styles_lower(self, style_name: str) -> float | None:
        """Return heading confidence for *style_name*, or None if not a heading."""
        for key, conf in self._HEADING_STYLES.items():
            if style_name.startswith(key):
                return conf
        return None

    @staticmethod
    def _make_chapter(
        index: int,
        title: str,
        paragraphs: list[Paragraph],
        confidence: float,
    ) -> Chapter:
        # Re-assign paragraph ids now that we know the chapter index.
        for i, p in enumerate(paragraphs):
            p.id = f"c{index:02d}-p{i:03d}"
        return Chapter(
            id=f"c{index:02d}",
            index=index,
            title=title,
            confidence=confidence,
            paragraphs=paragraphs,
        )

    @staticmethod
    def _extract_title(doc: DocxDocument, path: Path) -> str:  # type: ignore[name-defined]
        """Return core-property title if available, otherwise the filename stem."""
        try:
            title = doc.core_properties.title
            return title if title else path.stem
        except Exception:
            return path.stem
