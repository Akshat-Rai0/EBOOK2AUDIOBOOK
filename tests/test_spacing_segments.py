"""
Regression tests for spacing and text-mismatch edge cases in segmentation.

Covers two original debug scripts:
- test_spacing_issue.py  : extra spaces between quote and narration
- test_text_mismatch.py : leading/trailing spaces on the whole paragraph

In both cases the key property is that:
  (a) the segmenter produces non-empty dialogue and narration segments, and
  (b) every segment's text is a substring of the original paragraph text so
      that str.replace in the attribution stage can find and swap it.
"""

from __future__ import annotations

from ebook2audiobook.chunker.dialogue_segmenter import DialogueSegmenter


class TestExtraSpacesInParagraph:
    """Extra spaces between quote and narration tag must not break segmentation."""

    # Two spaces between quote and tag; two spaces between sentences
    _PARA = '"Hello,"  said Harry.  "How are you?"  asked Ron.'

    def test_produces_two_dialogue_segments(self) -> None:
        """Two quoted phrases → two dialogue segments."""
        segmenter = DialogueSegmenter()
        segments = segmenter.split_paragraph(
            paragraph_id="c00-p000",
            chapter_index=0,
            text=self._PARA,
        )
        dialogue_segs = [s for s in segments if s.kind == "dialogue"]
        assert len(dialogue_segs) == 2, (
            f"Expected 2 dialogue segments; got {len(dialogue_segs)}: "
            f"{[s.text for s in dialogue_segs]}"
        )

    def test_all_segments_non_empty(self) -> None:
        """No segment may be empty or whitespace-only."""
        segmenter = DialogueSegmenter()
        segments = segmenter.split_paragraph(
            paragraph_id="c00-p000",
            chapter_index=0,
            text=self._PARA,
        )
        for seg in segments:
            assert seg.text.strip(), (
                f"Segment {seg.id} has empty/whitespace text: {repr(seg.text)}"
            )

    def test_segment_texts_are_substrings_of_paragraph(self) -> None:
        """Every segment text must be a substring of the original paragraph text.

        This is the invariant that allows the attribution stage to use
        str.replace(seg.text, marker, 1) reliably.
        """
        segmenter = DialogueSegmenter()
        segments = segmenter.split_paragraph(
            paragraph_id="c00-p000",
            chapter_index=0,
            text=self._PARA,
        )
        for seg in segments:
            assert seg.text in self._PARA, (
                f"Segment {seg.id} text {repr(seg.text)} is not a substring "
                f"of the original paragraph {repr(self._PARA)}"
            )

    def test_markers_replace_correctly_despite_extra_spaces(self) -> None:
        """Attribution-stage marker replacement must succeed for both quotes."""
        segmenter = DialogueSegmenter()
        segments = segmenter.split_paragraph(
            paragraph_id="c00-p000",
            chapter_index=0,
            text=self._PARA,
        )
        marked = self._PARA
        q_count = 0
        for seg in segments:
            if seg.kind == "dialogue" and seg.speaker_id == "unknown":
                q_count += 1
                marked = marked.replace(seg.text, f"[Q{q_count}]", 1)

        assert "[Q1]" in marked, f"[Q1] missing from: {marked!r}"
        assert "[Q2]" in marked, f"[Q2] missing from: {marked!r}"


class TestLeadingTrailingSpacesInParagraph:
    """A paragraph with leading/trailing whitespace must still segment correctly."""

    # Leading two spaces, trailing two spaces
    _PARA = '  "Hello," said Harry.  "How are you?" asked Ron.  '

    def test_produces_two_dialogue_segments(self) -> None:
        """Two quoted phrases → two dialogue segments even with surrounding whitespace."""
        segmenter = DialogueSegmenter()
        segments = segmenter.split_paragraph(
            paragraph_id="c00-p000",
            chapter_index=0,
            text=self._PARA,
        )
        dialogue_segs = [s for s in segments if s.kind == "dialogue"]
        assert len(dialogue_segs) == 2, (
            f"Expected 2 dialogue segments; got {len(dialogue_segs)}: "
            f"{[s.text for s in dialogue_segs]}"
        )

    def test_all_segments_non_empty(self) -> None:
        """No segment may have empty or whitespace-only text."""
        segmenter = DialogueSegmenter()
        segments = segmenter.split_paragraph(
            paragraph_id="c00-p000",
            chapter_index=0,
            text=self._PARA,
        )
        for seg in segments:
            assert seg.text.strip(), (
                f"Segment {seg.id} has empty/whitespace text: {repr(seg.text)}"
            )

    def test_segment_texts_are_substrings_of_paragraph(self) -> None:
        """Every segment text must be findable inside the original paragraph string."""
        segmenter = DialogueSegmenter()
        segments = segmenter.split_paragraph(
            paragraph_id="c00-p000",
            chapter_index=0,
            text=self._PARA,
        )
        for seg in segments:
            assert seg.text in self._PARA, (
                f"Segment {seg.id} text {repr(seg.text)} is not a substring "
                f"of the paragraph {repr(self._PARA)}"
            )
