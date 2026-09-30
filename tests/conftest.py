"""
Shared pytest fixtures and synthetic file builders.

These utilities create small, self-contained ebook files in all four formats
using only installed dependencies — no network, no real book content.

The synthetic book has this structure:
  Title: "Test Book"
  Author: "Test Author"
  Chapters:
    Chapter 1: "The Beginning"   — 3 paragraphs, one with dialogue
    Chapter 2: "The Middle"      — 2 paragraphs
    Chapter 3: "The End"         — 2 paragraphs
  Expected chapter titles: ["The Beginning", "The Middle", "The End"]
  Expected dialogue paragraph: contains 'You lied to me'
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Synthetic book content (NO real book content in the repo)
# ---------------------------------------------------------------------------

CHAPTER_TITLES = ["The Beginning", "The Middle", "The End"]

CHAPTER_CONTENT = [
    # Chapter 1 — includes a dialogue paragraph
    [
        "Mira set the lamp down on the table and looked across the room.",
        '"You lied to me," she said quietly. "I did not," Tomas answered.',
        "The silence stretched between them like a fog over still water.",
    ],
    # Chapter 2
    [
        "The storm rolled in from the west, grey and cold.",
        "Neither of them spoke for a long time after that.",
    ],
    # Chapter 3
    [
        "By morning the rain had stopped and the roads were clear.",
        "Mira left without saying goodbye.",
    ],
]

DIALOGUE_TEXT = '"You lied to me," she said quietly. "I did not," Tomas answered.'


# ---------------------------------------------------------------------------
# TXT synthetic file builder
# ---------------------------------------------------------------------------


def make_txt(path: Path) -> None:
    """Write a synthetic TXT ebook with 3 chapters detectable by heuristics."""
    lines: list[str] = []
    for i, (title, paragraphs) in enumerate(zip(CHAPTER_TITLES, CHAPTER_CONTENT, strict=True)):
        lines.append(f"Chapter {i + 1}: {title}")
        lines.append("")
        for para in paragraphs:
            lines.append(para)
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# DOCX synthetic file builder
# ---------------------------------------------------------------------------


def make_docx(path: Path) -> None:
    """Write a synthetic DOCX ebook with Heading-1 chapter boundaries."""
    from docx import Document

    doc = Document()
    doc.core_properties.title = "Test Book"
    doc.core_properties.author = "Test Author"

    for title, paragraphs in zip(CHAPTER_TITLES, CHAPTER_CONTENT, strict=True):
        doc.add_heading(title, level=1)
        for para_text in paragraphs:
            doc.add_paragraph(para_text)

    doc.save(str(path))


# ---------------------------------------------------------------------------
# EPUB synthetic file builder
# ---------------------------------------------------------------------------


def make_epub(path: Path) -> None:
    """Write a synthetic EPUB 3 ebook with 3 spine items (one per chapter)."""
    from ebooklib import epub

    book = epub.EpubBook()
    book.set_identifier("test-book-001")
    book.set_title("Test Book")
    book.set_language("en")
    book.add_author("Test Author")

    chapters_epub = []
    for i, (title, paragraphs) in enumerate(zip(CHAPTER_TITLES, CHAPTER_CONTENT, strict=True)):
        ch = epub.EpubHtml(
            title=title,
            file_name=f"chapter{i + 1:02d}.xhtml",
            lang="en",
        )
        para_html = "".join(f"<p>{p}</p>" for p in paragraphs)
        ch.content = f"<h1>{title}</h1>{para_html}"
        book.add_item(ch)
        chapters_epub.append(ch)

    book.toc = tuple(epub.Link(ch.file_name, ch.title, ch.id) for ch in chapters_epub)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", *chapters_epub]

    epub.write_epub(str(path), book, {})


# ---------------------------------------------------------------------------
# PDF synthetic file builder (text layer)
# ---------------------------------------------------------------------------


def make_pdf(path: Path) -> None:
    """Write a synthetic PDF with a searchable text layer using reportlab."""
    from reportlab.lib.pagesizes import LETTER
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=LETTER)
    y_start = 720

    for i, (title, paragraphs) in enumerate(zip(CHAPTER_TITLES, CHAPTER_CONTENT, strict=True)):
        if i > 0:
            c.showPage()
        y = y_start
        # Chapter heading — ALL-CAPS to trigger the heuristic detector.
        c.setFont("Helvetica-Bold", 14)
        c.drawString(72, y, f"CHAPTER {i + 1}: {title.upper()}")
        y -= 30
        c.setFont("Helvetica", 12)
        for para_text in paragraphs:
            for line in textwrap.wrap(para_text, width=80):
                c.drawString(72, y, line)
                y -= 16
            y -= 8  # blank line between paragraphs

    c.save()


# ---------------------------------------------------------------------------
# Scanned PDF builder (no text layer)
# ---------------------------------------------------------------------------


def make_scanned_pdf(path: Path) -> None:
    """
    Write a minimal PDF whose pages contain only a rectangle (no text).
    pdfplumber extracts zero characters from this, triggering ScannedPdfError.
    """
    from reportlab.lib.pagesizes import LETTER
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=LETTER)
    # Draw only a rectangle — no text stream.
    c.rect(72, 72, 400, 600)
    c.save()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_project(tmp_path: Path) -> Path:
    """Return a temporary project directory."""
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    return project_dir


@pytest.fixture()
def txt_file(tmp_path: Path) -> Path:
    p = tmp_path / "test_book.txt"
    make_txt(p)
    return p


@pytest.fixture()
def docx_file(tmp_path: Path) -> Path:
    p = tmp_path / "test_book.docx"
    make_docx(p)
    return p


@pytest.fixture()
def epub_file(tmp_path: Path) -> Path:
    p = tmp_path / "test_book.epub"
    make_epub(p)
    return p


@pytest.fixture()
def pdf_file(tmp_path: Path) -> Path:
    p = tmp_path / "test_book.pdf"
    make_pdf(p)
    return p


@pytest.fixture()
def scanned_pdf_file(tmp_path: Path) -> Path:
    p = tmp_path / "scanned.pdf"
    make_scanned_pdf(p)
    return p
