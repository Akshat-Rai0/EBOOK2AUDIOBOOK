"""
PDF extractor — extracts Book from PDF files with a text layer.

**Text extraction library:** ``pdfplumber`` (MIT licence).
This was chosen over ``PyMuPDF`` (AGPL-3.0) to keep the project permissively
licensed.  Decision recorded 2026-10-01 in docs/DECISIONS.md.

**Scanned PDF detection:**
A PDF page with no extractable text (character count < threshold) is almost
certainly a scanned image.  When *all* pages look scanned, we raise
``ScannedPdfError`` with a clear message.  OCR support is planned for Phase B.

**Chapter detection strategy for PDF:**
PDFs have no semantic heading structure (unlike DOCX or EPUB), so we fall back
to text heuristics:
  - Lines matching ``Chapter <N>`` or ``Chapter <Roman>`` → confidence 0.80
  - ALL-CAPS short lines (≤ 60 chars) that look like headings  → confidence 0.60
  - Font-size changes (via pdfplumber char metadata)           → confidence 0.70

The ``ChapterDetector`` in the chunker layer refines these after extraction.

**Header/footer removal:**
We use ``remove_running_headers`` from the cleaning module (repeated-line
heuristic) and ``remove_page_numbers`` for bare digit lines.
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path

from ebook2audiobook.ingest.cleaning import clean, detect_dialogue
from ebook2audiobook.ingest.extractor import ExtractionError, Extractor, ScannedPdfError
from ebook2audiobook.models.book import Book, Chapter, Paragraph

try:
    import pdfplumber

    _PDF_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PDF_AVAILABLE = False

# Pages with fewer characters than this are treated as scanned/image pages.
_MIN_TEXT_CHARS_PER_PAGE = 50

# Heuristic chapter-heading patterns for PDF text.
_RE_CHAPTER_HEADER = re.compile(
    r"^\s*(chapter\s+\d+|chapter\s+[ivxlcdm]+\.?|prologue|epilogue|part\s+\d+)\b",
    re.IGNORECASE,
)
_RE_ALL_CAPS_HEADING = re.compile(r"^[A-Z\s\d\-]{4,60}$")


class PdfExtractor(Extractor):
    """Extracts a Book from a PDF with a searchable text layer."""

    @property
    def supported_extensions(self) -> list[str]:
        return [".pdf"]

    def can_handle(self, path: Path) -> bool:
        return path.suffix.lower() == ".pdf"

    def extract(self, path: Path) -> Book:
        if not _PDF_AVAILABLE:
            raise ExtractionError("pdfplumber is not installed. Run: uv add pdfplumber")

        try:
            pdf = pdfplumber.open(str(path))
        except Exception as exc:
            raise ExtractionError(f"Could not open PDF file: {exc}") from exc

        with pdf:
            self._check_for_scan(pdf, path)
            raw_pages = self._extract_pages(pdf)

        full_text = "\n\n".join(raw_pages)

        # Clean the whole text (headers, page numbers, etc.) then split chapters.
        cleaned_text = clean(full_text, strip_headers=True, is_pdf=True)
        chapters = self._split_into_chapters(cleaned_text)

        if not chapters:
            raise ExtractionError(f"No content could be extracted from PDF: {path}")

        title = path.stem
        return Book(
            id=str(uuid.uuid4()),
            title=title,
            source_path=str(path),
            source_format="pdf",
            chapters=chapters,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_for_scan(pdf: pdfplumber.PDF, path: Path) -> None:  # type: ignore[name-defined]
        """
        Raise ScannedPdfError if no page has enough extractable text.

        We check all pages and raise only if *every* page is below the threshold.
        A mixed PDF (some text, some images) is accepted.
        """
        total_pages = len(pdf.pages)
        if total_pages == 0:
            raise ExtractionError(f"PDF has no pages: {path}")

        text_pages = 0
        for page in pdf.pages:
            text = page.extract_text() or ""
            if len(text.strip()) >= _MIN_TEXT_CHARS_PER_PAGE:
                text_pages += 1

        if text_pages == 0:
            raise ScannedPdfError(
                f"'{path.name}' appears to be a scanned PDF with no text layer. "
                "OCR support is planned for Phase B. "
                "Tip: use a version with a searchable text layer."
            )

    @staticmethod
    def _extract_pages(pdf: pdfplumber.PDF) -> list[str]:  # type: ignore[name-defined]
        """Extract text from each page; skip pages below the text threshold."""
        pages: list[str] = []
        for page in pdf.pages:
            text = page.extract_text() or ""
            if len(text.strip()) >= _MIN_TEXT_CHARS_PER_PAGE:
                pages.append(text)
        return pages

    def _split_into_chapters(self, text: str) -> list[Chapter]:
        """
        Split cleaned PDF text into chapters using heuristic line matching.

        Returns at least one chapter even if no headings are found.
        """
        lines = text.split("\n")
        chapters: list[Chapter] = []
        current_title = ""
        current_lines: list[str] = []
        chapter_index = 0
        confidence = 0.50  # default for PDF

        def flush_chapter() -> None:
            nonlocal chapter_index
            if not current_lines:
                return
            block_text = "\n\n".join(
                " ".join(group) for group in self._group_into_paragraphs(current_lines) if group
            )
            paragraphs = self._make_paragraphs(chapter_index, block_text)
            if paragraphs:
                chapters.append(
                    Chapter(
                        id=f"c{chapter_index:02d}",
                        index=chapter_index,
                        title=current_title,
                        confidence=confidence,
                        paragraphs=paragraphs,
                    )
                )
                chapter_index += 1

        for line in lines:
            stripped = line.strip()
            if not stripped:
                current_lines.append("")
                continue

            # Check for chapter heading patterns.
            if _RE_CHAPTER_HEADER.match(stripped):
                flush_chapter()
                current_title = stripped
                current_lines = []
                confidence = 0.80
            elif (
                _RE_ALL_CAPS_HEADING.match(stripped) and len(stripped) >= 4 and len(stripped) <= 60
            ):
                flush_chapter()
                current_title = stripped
                current_lines = []
                confidence = 0.60
            else:
                current_lines.append(stripped)

        flush_chapter()

        # If no headings found, produce a single chapter.
        if not chapters:
            paragraphs = self._make_paragraphs(0, text)
            if paragraphs:
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

    @staticmethod
    def _group_into_paragraphs(lines: list[str]) -> list[list[str]]:
        """Group lines into paragraphs separated by blank lines."""
        groups: list[list[str]] = []
        current: list[str] = []
        for line in lines:
            if line.strip() == "":
                if current:
                    groups.append(current)
                    current = []
            else:
                current.append(line)
        if current:
            groups.append(current)
        return groups

    @staticmethod
    def _make_paragraphs(chapter_index: int, block_text: str) -> list[Paragraph]:
        """Split block_text into Paragraph objects."""
        paragraphs: list[Paragraph] = []
        for i, block in enumerate(block_text.split("\n\n")):
            block = block.strip()
            if not block:
                continue
            paragraphs.append(
                Paragraph(
                    id=f"c{chapter_index:02d}-p{i:03d}",
                    text=block,
                    original_text=block,
                    is_dialogue=detect_dialogue(block),
                )
            )
        return paragraphs
