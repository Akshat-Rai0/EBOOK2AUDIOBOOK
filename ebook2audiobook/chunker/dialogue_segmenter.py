"""
Dialogue segmenter — splits paragraphs into alternating dialogue and narration spans.

**What this module does (everyday analogy):**
Think of a play script. A novel writes spoken dialogue and scene description
together in one block of text. This segmenter acts like a script formatter:
it extracts every quoted line as an actor's speech, and sends the connecting
words ("she said", "he muttered", room descriptions) to the narrator.

**Rule contract:**
- Quoted spans ("...") are marked ``kind=SegmentKind.DIALOGUE``, ``speaker_id="unknown"``
  (ready for the attribution LLM to assign).
- Everything outside quotes is marked ``kind=SegmentKind.NARRATION``, ``speaker_id="narrator"``,
  ``confidence=1.0``, ``source=SegmentSource.RULE``.
- Identifiers follow the M2 stable scheme: ``c<CC>-p<PPP>-s<SS>``.
"""

from __future__ import annotations

import logging
import re

from ebook2audiobook.chunker.sentence_splitter import split_sentences
from ebook2audiobook.ingest.cleaning import split_to_chunks
from ebook2audiobook.models.book import Book
from ebook2audiobook.models.segment import (
    Segment,
    SegmentKind,
    SegmentSource,
    SegmentStatus,
)

logger = logging.getLogger(__name__)

# Matches text inside standard double quotes, including curly and straight quotes
_RE_QUOTED_SPAN = re.compile(r'("[^"]+")')


class DialogueSegmenter:
    """
    Splits book paragraphs into sequential, alternating dialogue and narration segments.
    """

    def __init__(self, max_chars: int = 400) -> None:
        """
        Parameters
        ----------
        max_chars:
            Maximum character length per segment (enforced for TTS engine safety).
        """
        self.max_chars = max_chars

    def split_paragraph(
        self,
        paragraph_id: str,
        chapter_index: int,
        text: str,
    ) -> list[Segment]:
        """
        Split a single paragraph into alternating spoken dialogue and narration segments.

        Parameters
        ----------
        paragraph_id:
            Stable paragraph identifier, e.g. ``"c01-p002"``.
        chapter_index:
            0-based chapter index.
        text:
            Cleaned paragraph text.

        Returns
        -------
        list[Segment]
            List of ordered Segment objects with IDs ``c<CC>-p<PPP>-s<SS>``.
        """
        clean_text = text.strip()
        if not clean_text:
            return []

        # Split text into alternating pieces by quote boundaries.
        # re.split with a capturing group keeps the delimiters in the resulting list.
        parts = _RE_QUOTED_SPAN.split(clean_text)

        raw_spans: list[tuple[str, SegmentKind]] = []
        for part in parts:
            part_str = part.strip()
            if not part_str:
                continue

            if part_str.startswith('"') and part_str.endswith('"') and len(part_str) >= 2:
                # Quoted speech span
                raw_spans.append((part_str, SegmentKind.DIALOGUE))
            else:
                # Narration / dialogue tag
                raw_spans.append((part_str, SegmentKind.NARRATION))

        # If no quotes found, whole paragraph is a single narration
        if not raw_spans:
            raw_spans = [(clean_text, SegmentKind.NARRATION)]

        segments: list[Segment] = []
        seg_idx = 0

        for span_text, kind in raw_spans:
            # Further break oversized spans at sentence boundaries if exceeding max_chars
            sub_chunks = self._chunk_span(span_text, kind)

            for chunk in sub_chunks:
                seg_id = f"{paragraph_id}-s{seg_idx:02d}"
                is_dialogue = kind == SegmentKind.DIALOGUE

                segments.append(
                    Segment(
                        id=seg_id,
                        chapter=chapter_index,
                        paragraph_id=paragraph_id,
                        speaker_id="unknown" if is_dialogue else "narrator",
                        speaker="unknown" if is_dialogue else "narrator",
                        kind=kind,
                        confidence=0.0 if is_dialogue else 1.0,
                        source=SegmentSource.RULE,
                        evidence=None,
                        text=chunk,
                        status=SegmentStatus.PENDING,
                    )
                )
                seg_idx += 1

        return segments

    def segment_book(self, book: Book) -> list[Segment]:
        """
        Segment all paragraphs across all chapters in *book*.
        """
        all_segments: list[Segment] = []
        for chapter in book.chapters:
            for paragraph in chapter.paragraphs:
                segs = self.split_paragraph(
                    paragraph_id=paragraph.id,
                    chapter_index=chapter.index,
                    text=paragraph.text,
                )
                all_segments.extend(segs)
        return all_segments

    def _chunk_span(self, text: str, kind: SegmentKind) -> list[str]:
        """
        Ensure span text does not exceed max_chars, splitting at sentence boundaries.
        """
        if len(text) <= self.max_chars:
            return [text]

        # Break at sentences first
        sentences = split_sentences(text)
        chunks: list[str] = []
        for s in sentences:
            if len(s) <= self.max_chars:
                chunks.append(s)
            else:
                chunks.extend(split_to_chunks(s, max_chars=self.max_chars))

        return chunks or [text]
