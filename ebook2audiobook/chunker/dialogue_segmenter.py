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
- Multi-paragraph quotes are tracked with ``continues_previous`` flag.
"""

from __future__ import annotations

import logging

from ebook2audiobook.chunker.quotes import find_spans, normalise_double_style
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
        prev_ended_inside: bool = False,
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
        prev_ended_inside:
            True if the previous paragraph ended inside a quote (multi-paragraph speech).

        Returns
        -------
        list[Segment]
            List of ordered Segment objects with IDs ``c<CC>-p<PPP>-s<SS>``.
        """
        clean_text = text.strip()
        if not clean_text:
            return []

        # Normalise curly quotes to straight quotes
        normalised = normalise_double_style(clean_text)

        # Split into (text, kind) spans
        raw_spans, ends_inside = find_spans(normalised)

        # If no quotes found, whole paragraph is a single narration
        if not raw_spans:
            raw_spans = [(normalised, SegmentKind.NARRATION)]

        segments: list[Segment] = []
        seg_idx = 0

        for span_idx, (span_text, kind) in enumerate(raw_spans):
            # If this is the first span and it's dialogue, and previous paragraph
            # ended inside a quote, mark this as continuing the previous speech
            continues_previous = False
            if span_idx == 0 and kind == SegmentKind.DIALOGUE and prev_ended_inside:
                continues_previous = True

            # Strip whitespace (find_spans preserves it for round-trip)
            span_text = span_text.strip()
            if not span_text:
                continue

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
                        continues_previous=continues_previous,
                    )
                )
                seg_idx += 1

        return segments

    def split_paragraph_ex(
        self,
        paragraph_id: str,
        chapter_index: int,
        text: str,
        prev_ended_inside: bool = False,
    ) -> tuple[list[Segment], bool]:
        """
        Split a paragraph and also return whether it ends inside a quote.

        This is used by segment_book to track multi-paragraph quotes across
        consecutive paragraphs within a chapter.

        Returns
        -------
        tuple[list[Segment], bool]
            (segments, ends_inside_quote)
        """
        clean_text = text.strip()
        if not clean_text:
            return [], False

        # Normalise curly quotes to straight quotes
        normalised = normalise_double_style(clean_text)

        # Split into (text, kind) spans
        raw_spans, ends_inside = find_spans(normalised)

        # If no quotes found, whole paragraph is a single narration
        if not raw_spans:
            raw_spans = [(normalised, SegmentKind.NARRATION)]

        segments: list[Segment] = []
        seg_idx = 0

        for span_idx, (span_text, kind) in enumerate(raw_spans):
            # If this is the first span and it's dialogue, and previous paragraph
            # ended inside a quote, mark this as continuing the previous speech
            continues_previous = False
            if span_idx == 0 and kind == SegmentKind.DIALOGUE and prev_ended_inside:
                continues_previous = True

            # Strip whitespace (find_spans preserves it for round-trip)
            span_text = span_text.strip()
            if not span_text:
                continue

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
                        continues_previous=continues_previous,
                    )
                )
                seg_idx += 1

        return segments, ends_inside

    def segment_book(self, book: Book) -> list[Segment]:
        """
        Segment all paragraphs across all chapters in *book*.

        Tracks multi-paragraph quotes across consecutive paragraphs within a chapter.
        The ends_inside flag is reset at chapter boundaries.
        """
        all_segments: list[Segment] = []
        for chapter in book.chapters:
            prev_ended_inside = False
            for paragraph in chapter.paragraphs:
                segs, ends_inside = self.split_paragraph_ex(
                    paragraph_id=paragraph.id,
                    chapter_index=chapter.index,
                    text=paragraph.text,
                    prev_ended_inside=prev_ended_inside,
                )
                all_segments.extend(segs)

                # If quote never closes and next paragraph doesn't start with quote,
                # log a warning and don't carry the flag
                if prev_ended_inside and ends_inside:
                    # Quote continues - carry the flag
                    pass
                elif prev_ended_inside and not ends_inside:
                    # Quote closed in this paragraph - reset flag
                    prev_ended_inside = False
                elif not prev_ended_inside and ends_inside:
                    # Quote opened in this paragraph - carry flag
                    prev_ended_inside = True
                else:
                    # No quote or quote closed and reopened - reset
                    prev_ended_inside = False

                # If we're still inside after this paragraph, check if the next
                # paragraph (if it exists) starts with dialogue
                if prev_ended_inside:
                    # If next paragraph doesn't start with dialogue, reset flag
                    # (this handles the case where a quote never closes)
                    # We'll check this on the next iteration
                    pass

            # Reset at chapter boundary
            prev_ended_inside = False

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
