"""
Tests for DialogueSegmenter (M4 Step 2).

Verifies paragraph decomposition into alternating dialogue and narration spans,
stable ID generation, and character limit chunking.
"""

from __future__ import annotations

from ebook2audiobook.chunker.dialogue_segmenter import DialogueSegmenter
from ebook2audiobook.models.book import Book, Chapter, Paragraph
from ebook2audiobook.models.segment import SegmentKind, SegmentSource, SegmentStatus


class TestDialogueSegmenter:
    def test_alternating_dialogue_and_narration(self):
        """
        'You lied,' she said, 'and you know it.' splits into 3 alternating segments.
        Dialogue tags belong to narrator.
        """
        segmenter = DialogueSegmenter(max_chars=400)
        text = '"You lied," she said, "and you know it."'
        segments = segmenter.split_paragraph(paragraph_id="c01-p002", chapter_index=1, text=text)

        assert len(segments) == 3

        # Span 1: dialogue
        assert segments[0].id == "c01-p002-s00"
        assert segments[0].kind == SegmentKind.DIALOGUE
        assert segments[0].speaker_id == "unknown"
        assert segments[0].text == '"You lied,"'

        # Span 2: narration tag
        assert segments[1].id == "c01-p002-s01"
        assert segments[1].kind == SegmentKind.NARRATION
        assert segments[1].speaker_id == "narrator"
        assert segments[1].confidence == 1.0
        assert segments[1].source == SegmentSource.RULE
        assert segments[1].text == "she said,"

        # Span 3: dialogue continuation
        assert segments[2].id == "c01-p002-s02"
        assert segments[2].kind == SegmentKind.DIALOGUE
        assert segments[2].speaker_id == "unknown"
        assert segments[2].text == '"and you know it."'

    def test_pure_narration_paragraph(self):
        """Paragraph with no quotes is a single narrator segment."""
        segmenter = DialogueSegmenter()
        text = "The room was cold and the curtains were drawn tight against the storm."
        segments = segmenter.split_paragraph(paragraph_id="c00-p000", chapter_index=0, text=text)

        assert len(segments) == 1
        assert segments[0].id == "c00-p000-s00"
        assert segments[0].kind == SegmentKind.NARRATION
        assert segments[0].speaker_id == "narrator"
        assert segments[0].text == text

    def test_pure_dialogue_paragraph(self):
        """Paragraph with only dialogue is a single dialogue segment."""
        segmenter = DialogueSegmenter()
        text = '"Is anyone there?"'
        segments = segmenter.split_paragraph(paragraph_id="c00-p001", chapter_index=0, text=text)

        assert len(segments) == 1
        assert segments[0].id == "c00-p001-s00"
        assert segments[0].kind == SegmentKind.DIALOGUE
        assert segments[0].speaker_id == "unknown"
        assert segments[0].text == text

    def test_narration_preceding_dialogue(self):
        """Narration followed by dialogue."""
        segmenter = DialogueSegmenter()
        text = 'She hesitated at the door. "Can I come in?"'
        segments = segmenter.split_paragraph(paragraph_id="c02-p005", chapter_index=2, text=text)

        assert len(segments) == 2
        assert segments[0].kind == SegmentKind.NARRATION
        assert segments[0].text == "She hesitated at the door."
        assert segments[1].kind == SegmentKind.DIALOGUE
        assert segments[1].text == '"Can I come in?"'

    def test_empty_paragraph(self):
        """Empty or whitespace-only paragraph produces no segments."""
        segmenter = DialogueSegmenter()
        assert segmenter.split_paragraph("c00-p000", 0, "   ") == []

    def test_segment_book_end_to_end(self):
        """segment_book processes chapters and paragraphs in reading order."""
        book = Book(
            id="test-book",
            title="A Story",
            author="Author",
            source_path="/path/test.txt",
            source_format="txt",
            chapters=[
                Chapter(
                    id="c00",
                    index=0,
                    title="Intro",
                    paragraphs=[
                        Paragraph(
                            id="c00-p000",
                            text="The clock struck ten.",
                            original_text="The clock struck ten.",
                        ),
                        Paragraph(
                            id="c00-p001",
                            text='"Hurry," he whispered.',
                            original_text='"Hurry," he whispered.',
                        ),
                    ],
                )
            ],
        )
        segmenter = DialogueSegmenter()
        segs = segmenter.segment_book(book)
        assert len(segs) == 3
        assert segs[0].text == "The clock struck ten."
        assert segs[1].text == '"Hurry,"'
        assert segs[2].text == "he whispered."
        assert all(s.status == SegmentStatus.PENDING for s in segs)
