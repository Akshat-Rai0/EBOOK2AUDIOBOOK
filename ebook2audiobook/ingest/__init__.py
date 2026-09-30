"""Ingest package — ebook extraction and text cleaning."""

from ebook2audiobook.ingest.docx import DocxExtractor
from ebook2audiobook.ingest.epub import EpubExtractor
from ebook2audiobook.ingest.extractor import (
    ExtractionError,
    Extractor,
    ExtractorRegistry,
    ScannedPdfError,
)
from ebook2audiobook.ingest.pdf import PdfExtractor
from ebook2audiobook.ingest.txt import TxtExtractor


def default_registry() -> ExtractorRegistry:
    """Return a registry with all Phase-A extractors registered."""
    return ExtractorRegistry(
        [
            TxtExtractor(),
            DocxExtractor(),
            EpubExtractor(),
            PdfExtractor(),
        ]
    )


__all__ = [
    "Extractor",
    "ExtractorRegistry",
    "ExtractionError",
    "ScannedPdfError",
    "TxtExtractor",
    "DocxExtractor",
    "EpubExtractor",
    "PdfExtractor",
    "default_registry",
]
