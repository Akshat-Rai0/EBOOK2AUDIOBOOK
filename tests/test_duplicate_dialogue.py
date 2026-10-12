"""
Regression tests for duplicate dialogue handling.

When a paragraph contains the same quoted text twice (e.g. two characters
both say "Hello,"), the attribution stage must use str.replace(..., 1) so
that each occurrence is replaced with its own distinct Q-marker in order.
"""

from __future__ import annotations

from ebook2audiobook.chunker.dialogue_segmenter import DialogueSegmenter


class TestDuplicateDialogueMarking:
    """Duplicate dialogue text segments are marked with distinct, sequential Q-IDs."""

    _PARA = '"Hello," said Harry. "Hello," said Ron.'

    def test_duplicate_paragraph_produces_two_dialogue_segments(self) -> None:
        """Segmenter must produce exactly 2 dialogue segments for the test paragraph."""
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

    def test_duplicate_segments_both_have_same_text(self) -> None:
        """Both dialogue segments should carry the same quoted text '\"Hello,\"'."""
        segmenter = DialogueSegmenter()
        segments = segmenter.split_paragraph(
            paragraph_id="c00-p000",
            chapter_index=0,
            text=self._PARA,
        )
        dialogue_segs = [s for s in segments if s.kind == "dialogue"]
        assert dialogue_segs[0].text.strip() == '"Hello,"', (
            f"First seg text: {dialogue_segs[0].text!r}"
        )
        assert dialogue_segs[1].text.strip() == '"Hello,"', (
            f"Second seg text: {dialogue_segs[1].text!r}"
        )

    def test_sequential_replace_gives_distinct_markers(self) -> None:
        """
        Simulating the attribution-stage marker logic: str.replace(..., 1) in segment
        order must produce Q1 at the first occurrence and Q2 at the second.
        """
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
                marker = f"[Q{q_count}]"
                marked = marked.replace(seg.text, marker, 1)

        # Both quotes replaced
        assert "[Q1]" in marked, f"[Q1] not in marked paragraph: {marked!r}"
        assert "[Q2]" in marked, f"[Q2] not in marked paragraph: {marked!r}"
        # Original text completely replaced
        assert '"Hello,"' not in marked, (
            f"Original quote still present in marked paragraph: {marked!r}"
        )
        # Q1 appears before Q2 (left-to-right order preserved)
        assert marked.index("[Q1]") < marked.index("[Q2]"), (
            "Q1 marker should appear before Q2 in the paragraph"
        )

    def test_all_segments_non_empty(self) -> None:
        """No segment produced for this paragraph may have empty text."""
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
