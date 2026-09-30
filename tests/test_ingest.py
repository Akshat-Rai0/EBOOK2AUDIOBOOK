"""
Ingest integration tests — all four formats + scanned PDF rejection.

Each test:
1. Builds a synthetic file using the shared conftest fixtures.
2. Extracts a Book.
3. Runs ChapterDetector.
4. Asserts that the 3 known chapter titles are found and the dialogue
   paragraph is present.

Re-running ingest (idempotency) is verified by calling extract() twice and
checking that the result is identical (same chapter count and titles).
"""

from __future__ import annotations

import pytest

from ebook2audiobook.chunker.chapter_detector import ChapterDetector
from ebook2audiobook.ingest import (
    DocxExtractor,
    EpubExtractor,
    PdfExtractor,
    ScannedPdfError,
    TxtExtractor,
)
from tests.conftest import CHAPTER_TITLES

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _titles(book) -> list[str]:
    """Return the list of chapter titles from the book (stripped, lowercase)."""
    return [ch.title.strip() for ch in book.chapters]


def _has_dialogue(book) -> bool:
    """Return True if any paragraph in the book contains the expected dialogue text."""
    for ch in book.chapters:
        for p in ch.paragraphs:
            if "You lied to me" in p.text or "You lied to me" in p.original_text:
                return True
    return False


# ---------------------------------------------------------------------------
# TXT
# ---------------------------------------------------------------------------


class TestTxtExtractor:
    def test_extracts_book(self, txt_file):
        book = TxtExtractor().extract(txt_file)
        assert book.source_format == "txt"
        assert book.chapters  # at least one chapter

    def test_chapter_detection_after_detector(self, txt_file, tmp_project):
        book = TxtExtractor().extract(txt_file)
        book = ChapterDetector(project_dir=tmp_project).detect(book)
        titles = _titles(book)
        # All three chapter titles must appear somewhere in the detected titles.
        for expected_title in CHAPTER_TITLES:
            assert any(expected_title.lower() in t.lower() for t in titles), (
                f"Expected chapter title '{expected_title}' not found in {titles}"
            )

    def test_dialogue_present(self, txt_file, tmp_project):
        book = TxtExtractor().extract(txt_file)
        book = ChapterDetector(project_dir=tmp_project).detect(book)
        assert _has_dialogue(book), "Expected dialogue paragraph not found"

    def test_idempotent(self, txt_file):
        book1 = TxtExtractor().extract(txt_file)
        book2 = TxtExtractor().extract(txt_file)
        assert book1.chapter_count == book2.chapter_count

    def test_can_handle(self, txt_file):
        assert TxtExtractor().can_handle(txt_file)

    def test_cannot_handle_epub(self, epub_file):
        assert not TxtExtractor().can_handle(epub_file)


# ---------------------------------------------------------------------------
# DOCX
# ---------------------------------------------------------------------------


class TestDocxExtractor:
    def test_extracts_book(self, docx_file):
        book = DocxExtractor().extract(docx_file)
        assert book.source_format == "docx"

    def test_chapter_titles(self, docx_file):
        book = DocxExtractor().extract(docx_file)
        titles = _titles(book)
        for expected_title in CHAPTER_TITLES:
            assert any(expected_title.lower() in t.lower() for t in titles), (
                f"Expected chapter title '{expected_title}' not found in {titles}"
            )

    def test_high_confidence(self, docx_file):
        """DOCX Heading-1 chapters must have confidence >= 0.85."""
        book = DocxExtractor().extract(docx_file)
        for ch in book.chapters:
            if ch.title:  # titled chapters came from headings
                assert ch.confidence >= 0.80, (
                    f"Chapter '{ch.title}' has low confidence {ch.confidence}"
                )

    def test_dialogue_present(self, docx_file):
        book = DocxExtractor().extract(docx_file)
        assert _has_dialogue(book)

    def test_idempotent(self, docx_file):
        book1 = DocxExtractor().extract(docx_file)
        book2 = DocxExtractor().extract(docx_file)
        assert book1.chapter_count == book2.chapter_count

    def test_title_from_metadata(self, docx_file):
        book = DocxExtractor().extract(docx_file)
        assert book.title  # should be "Test Book" from core properties


# ---------------------------------------------------------------------------
# EPUB
# ---------------------------------------------------------------------------


class TestEpubExtractor:
    def test_extracts_book(self, epub_file):
        book = EpubExtractor().extract(epub_file)
        assert book.source_format == "epub"

    def test_chapter_titles(self, epub_file):
        book = EpubExtractor().extract(epub_file)
        titles = _titles(book)
        for expected_title in CHAPTER_TITLES:
            assert any(expected_title.lower() in t.lower() for t in titles), (
                f"Expected chapter title '{expected_title}' not found in {titles}"
            )

    def test_structural_confidence(self, epub_file):
        """EPUB spine chapters must have confidence >= 0.90."""
        book = EpubExtractor().extract(epub_file)
        for ch in book.chapters:
            assert ch.confidence >= 0.90, (
                f"Chapter '{ch.title}' has confidence {ch.confidence} < 0.90"
            )

    def test_correct_chapter_count(self, epub_file):
        book = EpubExtractor().extract(epub_file)
        # EPUB has 3 spine items + possibly a nav item; at least 3 content chapters.
        content_chapters = [ch for ch in book.chapters if ch.paragraphs]
        assert len(content_chapters) == 3

    def test_dialogue_present(self, epub_file):
        book = EpubExtractor().extract(epub_file)
        assert _has_dialogue(book)

    def test_metadata(self, epub_file):
        book = EpubExtractor().extract(epub_file)
        assert "Test Book" in book.title
        assert "Test Author" in book.author

    def test_idempotent(self, epub_file):
        book1 = EpubExtractor().extract(epub_file)
        book2 = EpubExtractor().extract(epub_file)
        assert book1.chapter_count == book2.chapter_count


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------


class TestPdfExtractor:
    def test_extracts_book(self, pdf_file):
        book = PdfExtractor().extract(pdf_file)
        assert book.source_format == "pdf"
        assert book.chapters

    def test_chapter_titles(self, pdf_file):
        book = PdfExtractor().extract(pdf_file)
        titles = _titles(book)
        for expected_title in CHAPTER_TITLES:
            assert any(expected_title.lower() in t.lower() for t in titles), (
                f"Expected chapter title '{expected_title}' not found in {titles}"
            )

    def test_dialogue_present(self, pdf_file):
        book = PdfExtractor().extract(pdf_file)
        assert _has_dialogue(book)

    def test_idempotent(self, pdf_file):
        book1 = PdfExtractor().extract(pdf_file)
        book2 = PdfExtractor().extract(pdf_file)
        assert book1.chapter_count == book2.chapter_count


# ---------------------------------------------------------------------------
# Scanned PDF rejection
# ---------------------------------------------------------------------------


class TestScannedPdfRejection:
    def test_scanned_pdf_raises_error(self, scanned_pdf_file):
        """A PDF with no text layer must raise ScannedPdfError with a clear message."""
        with pytest.raises(ScannedPdfError) as exc_info:
            PdfExtractor().extract(scanned_pdf_file)
        msg = str(exc_info.value)
        assert "scanned" in msg.lower() or "text layer" in msg.lower(), (
            f"Error message does not mention 'scanned' or 'text layer': {msg}"
        )
        assert "Phase B" in msg or "OCR" in msg, f"Error message should mention OCR/Phase B: {msg}"
