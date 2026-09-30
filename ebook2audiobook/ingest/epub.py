"""
EPUB extractor — extracts Book from EPUB 2 and EPUB 3 files.

**Chapter detection strategy for EPUB:**
EPUB files are ZIP archives containing XHTML chapters linked from a navigation
document (NCX for EPUB 2, ``nav.xhtml`` for EPUB 3).  The spine defines reading
order.  Each spine item is one chapter — this is structural, not heuristic,
so confidence is set to 0.95.

**Licence note (AGPL-3.0):**
``ebooklib`` is AGPL-3.0.  This is acceptable for open academic use where the
source code is publicly available.  See docs/LICENSES.md for details.
Decision recorded 2026-10-01 in docs/DECISIONS.md.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from ebook2audiobook.ingest.cleaning import clean, detect_dialogue
from ebook2audiobook.ingest.extractor import ExtractionError, Extractor
from ebook2audiobook.models.book import Book, Chapter, Paragraph

try:
    import ebooklib
    from bs4 import BeautifulSoup
    from ebooklib import epub

    _EPUB_AVAILABLE = True
except ImportError:  # pragma: no cover
    _EPUB_AVAILABLE = False


class EpubExtractor(Extractor):
    """Extracts a Book from EPUB 2 and EPUB 3 files via ebooklib."""

    @property
    def supported_extensions(self) -> list[str]:
        return [".epub"]

    def can_handle(self, path: Path) -> bool:
        return path.suffix.lower() == ".epub"

    def extract(self, path: Path) -> Book:
        if not _EPUB_AVAILABLE:
            raise ExtractionError(
                "ebooklib and beautifulsoup4 are not installed. Run: uv add ebooklib beautifulsoup4"
            )

        try:
            book_epub = epub.read_epub(str(path), options={"ignore_ncx": False})
        except Exception as exc:
            raise ExtractionError(f"Could not read EPUB file: {exc}") from exc

        title = book_epub.get_metadata("DC", "title")
        title_str = title[0][0] if title else path.stem

        author = book_epub.get_metadata("DC", "creator")
        author_str = author[0][0] if author else ""

        chapters = self._build_chapters(book_epub)
        if not chapters:
            raise ExtractionError(f"No readable content found in EPUB: {path}")

        return Book(
            id=str(uuid.uuid4()),
            title=title_str,
            author=author_str,
            source_path=str(path),
            source_format="epub",
            chapters=chapters,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_chapters(self, book_epub: epub.EpubBook) -> list[Chapter]:  # type: ignore[name-defined]
        """
        Walk the EPUB spine and convert each readable document to a Chapter.

        EPUB spine items that are not HTML (images, CSS) are skipped silently.
        """
        chapters: list[Chapter] = []
        chapter_index = 0

        for item in book_epub.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            # Skip navigation documents (TOC) and NCX files — they are not story chapters.
            if isinstance(item, (epub.EpubNav, epub.EpubNcx)):
                continue
            item_name = (item.get_name() or "").lower()
            if "nav." in item_name or "toc." in item_name:
                continue

            html = item.get_content()
            if not html:
                continue

            soup = BeautifulSoup(html, "html.parser")

            # Extract chapter title from the first heading element.
            heading = soup.find(["h1", "h2", "h3"])
            title = heading.get_text(strip=True) if heading else ""

            # Remove heading from body text to avoid duplication.
            if heading:
                heading.decompose()

            # Extract paragraphs from <p> tags; fall back to body text.
            raw_paragraphs = soup.find_all("p")
            if raw_paragraphs:
                texts = [p.get_text(separator=" ", strip=True) for p in raw_paragraphs]
            else:
                body_text = soup.get_text(separator="\n")
                texts = [t.strip() for t in body_text.split("\n\n") if t.strip()]

            if not any(texts):
                continue

            paragraphs: list[Paragraph] = []
            for i, raw_text in enumerate(texts):
                if not raw_text.strip():
                    continue
                original = raw_text
                cleaned = clean(raw_text)
                if not cleaned:
                    continue
                paragraphs.append(
                    Paragraph(
                        id=f"c{chapter_index:02d}-p{i:03d}",
                        text=cleaned,
                        original_text=original,
                        is_dialogue=detect_dialogue(cleaned),
                    )
                )

            if not paragraphs:
                continue

            chapters.append(
                Chapter(
                    id=f"c{chapter_index:02d}",
                    index=chapter_index,
                    title=title,
                    confidence=0.95,  # structural — from EPUB spine
                    paragraphs=paragraphs,
                )
            )
            chapter_index += 1

        return chapters
