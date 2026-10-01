"""Models package — all pydantic data contracts for ebook2audiobook."""

from ebook2audiobook.models.book import Book, Chapter, Paragraph
from ebook2audiobook.models.cast import Cast, Character, CharacterProfile, VoiceRef
from ebook2audiobook.models.job import ConversionMode, Job, StageStatus
from ebook2audiobook.models.segment import Segment, SegmentKind, SegmentSource, SegmentStatus

__all__ = [
    "Book",
    "Chapter",
    "Paragraph",
    "Segment",
    "SegmentKind",
    "SegmentSource",
    "SegmentStatus",
    "Character",
    "CharacterProfile",
    "Cast",
    "VoiceRef",
    "Job",
    "StageStatus",
    "ConversionMode",
]
