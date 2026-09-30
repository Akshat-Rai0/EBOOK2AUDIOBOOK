"""
Abstract base class for all ebook format extractors.

**What this interface guarantees:**
Every extractor accepts a filesystem path and returns a ``Book`` object.
No extractor may perform network I/O or write to disk; those concerns belong
to the CLI and orchestrator layers.

**Adding a new format** means creating a new class that inherits from
``Extractor`` and implementing the three abstract methods.  The ``ExtractorRegistry``
in this module automatically discovers registered extractors by extension.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ebook2audiobook.models.book import Book


class Extractor(ABC):
    """
    Abstract base for ebook format extractors.

    Analogy: think of each extractor as a specialist translator who reads one
    dialect (EPUB, DOCX, …) and produces a single, unified transcript (Book).
    """

    @abstractmethod
    def can_handle(self, path: Path) -> bool:
        """
        Return True if this extractor can process the file at *path*.

        Implementations should check file extension and, optionally, magic bytes.
        Do NOT open large files to make this decision.
        """

    @abstractmethod
    def extract(self, path: Path) -> Book:
        """
        Parse the file at *path* and return a structured ``Book``.

        Raises
        ------
        ExtractionError
            If the file cannot be parsed (e.g., DRM-locked, empty, wrong format).
        ScannedPdfError
            If the file is a PDF with no extractable text layer.
        """

    @property
    @abstractmethod
    def supported_extensions(self) -> list[str]:
        """Return the list of file extensions handled, e.g. ``['.epub']``."""


class ExtractionError(RuntimeError):
    """Raised when a file cannot be extracted by any registered extractor."""


class ScannedPdfError(ExtractionError):
    """
    Raised when a PDF has no text layer (i.e. it is a scanned image PDF).

    OCR support is planned for Phase B; until then this is an expected,
    user-facing failure with a clear message.
    """


class ExtractorRegistry:
    """
    Finds the right extractor for a given file path.

    Usage::

        registry = ExtractorRegistry([TxtExtractor(), DocxExtractor(), ...])
        book = registry.extract(Path("mybook.epub"))
    """

    def __init__(self, extractors: list[Extractor]) -> None:
        self._extractors = extractors

    def extract(self, path: Path) -> Book:
        """Run the first matching extractor and return the Book."""
        path = Path(path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        for extractor in self._extractors:
            if extractor.can_handle(path):
                return extractor.extract(path)
        raise ExtractionError(
            f"No extractor found for '{path.suffix}' files. "
            f"Supported formats: {self._supported_extensions()}"
        )

    def _supported_extensions(self) -> str:
        exts: list[str] = []
        for e in self._extractors:
            exts.extend(e.supported_extensions)
        return ", ".join(sorted(set(exts)))
