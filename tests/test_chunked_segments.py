"""
Regression tests for dialogue chunking edge cases.

Covers:
- Long dialogues that exceed max_chars and are split into multiple segments
- The known issue where a single chunked quote gets N Q-markers (one per chunk)
- Segment text always non-empty after the whitespace-filter fix (M4 step 3b)
"""

from __future__ import annotations

from ebook2audiobook.chunker.dialogue_segmenter import DialogueSegmenter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LONG_DIALOGUE = (
    '"This is a very long piece of dialogue that will definitely exceed the '
    "400 character limit when we include all the text in this quote. "
    "It should be split into multiple segments by the chunker. "
    'We need to see if the replacement logic handles this correctly." said Harry.'
)


def _make_segmenter(max_chars: int = 100) -> DialogueSegmenter:
    """Return a segmenter with a small max_chars to force chunking."""
    return DialogueSegmenter(max_chars=max_chars)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLongDialogueChunking:
    """A dialogue that exceeds max_chars is split into multiple segments."""

    def test_long_dialogue_produces_multiple_dialogue_segments(self) -> None:
        """split_paragraph with max_chars=100 should create >1 dialogue segments."""
        segmenter = _make_segmenter(max_chars=100)
        segments = segmenter.split_paragraph(
            paragraph_id="c00-p000",
            chapter_index=0,
            text=_LONG_DIALOGUE,
        )
        dialogue_segs = [s for s in segments if s.kind == "dialogue"]
        assert len(dialogue_segs) > 1, (
            "Expected >1 dialogue segment for a long quote; got "
            f"{len(dialogue_segs)}: {[s.text for s in dialogue_segs]}"
        )

    def test_all_segments_have_non_empty_text(self) -> None:
        """No segment should have empty or whitespace-only text (M4 fix 3b)."""
        segmenter = _make_segmenter(max_chars=100)
        segments = segmenter.split_paragraph(
            paragraph_id="c00-p000",
            chapter_index=0,
            text=_LONG_DIALOGUE,
        )
        for seg in segments:
            assert seg.text.strip(), (
                f"Segment {seg.id} has empty text: {repr(seg.text)}"
            )

    def test_all_dialogue_segments_have_unknown_speaker(self) -> None:
        """Chunked dialogue segments must all start as speaker_id='unknown'."""
        segmenter = _make_segmenter(max_chars=100)
        segments = segmenter.split_paragraph(
            paragraph_id="c00-p000",
            chapter_index=0,
            text=_LONG_DIALOGUE,
        )
        for seg in segments:
            if seg.kind == "dialogue":
                assert seg.speaker_id == "unknown", (
                    f"Segment {seg.id} has speaker_id={seg.speaker_id!r}; expected 'unknown'"
                )

    def test_chunked_quote_gets_multiple_q_markers(self) -> None:
        """
        Known issue (regression guard): a single long quote that is chunked into
        N dialogue segments produces N separate Q-markers, not one.

        This test documents the *current* behaviour so any future change that
        collapses multiple chunks into one marker is caught immediately.
        """
        segmenter = _make_segmenter(max_chars=100)
        segments = segmenter.split_paragraph(
            paragraph_id="c00-p000",
            chapter_index=0,
            text=_LONG_DIALOGUE,
        )

        # Simulate the attribution-stage marker logic (same as stage.py)
        marked = _LONG_DIALOGUE
        q_count = 0
        for seg in segments:
            if seg.kind == "dialogue" and seg.speaker_id == "unknown":
                q_count += 1
                marker = f"[Q{q_count}]"
                marked = marked.replace(seg.text, marker, 1)

        dialogue_segs = [s for s in segments if s.kind == "dialogue"]
        # Each chunk currently produces its own marker → q_count == len(dialogue_segs)
        assert q_count == len(dialogue_segs), (
            f"Expected {len(dialogue_segs)} Q-markers (one per chunk) but got {q_count}. "
            "If this fails, the chunking / marker logic has changed — verify intentional."
        )
        # All markers must appear in the final string
        for i in range(1, q_count + 1):
            assert f"[Q{i}]" in marked, f"[Q{i}] missing from marked paragraph"


class TestShortDialogueSegmentation:
    """A short paragraph that fits in one segment should produce exactly one dialogue seg."""

    def test_short_dialogue_produces_one_segment(self) -> None:
        """'"Hello," said Harry.' → one dialogue segment + one narration segment."""
        segmenter = DialogueSegmenter()
        segments = segmenter.split_paragraph(
            paragraph_id="c00-p000",
            chapter_index=0,
            text='"Hello," said Harry.',
        )
        dialogue_segs = [s for s in segments if s.kind == "dialogue"]
        assert len(dialogue_segs) == 1, (
            f"Expected 1 dialogue segment; got {len(dialogue_segs)}"
        )
        assert dialogue_segs[0].text.strip() == '"Hello,"'
