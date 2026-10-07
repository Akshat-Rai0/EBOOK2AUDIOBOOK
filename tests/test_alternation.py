"""Tests for speaker alternation heuristic."""

from ebook2audiobook.attribution.alternation import apply_alternation_correction
from ebook2audiobook.models.segment import Segment, SegmentKind


def test_alternation_corrects_middle_unknown():
    """A/?/A pattern should correct middle segment to match outer speakers."""
    segments = [
        Segment(
            id="seg1",
            chapter=1,
            paragraph_id="p001",
            speaker="Harry",
            speaker_id="harry",
            kind=SegmentKind.DIALOGUE,
            text="Hello.",
            confidence=0.9,
            source="llm",
            evidence="",
        ),
        Segment(
            id="seg2",
            chapter=1,
            paragraph_id="p001",
            speaker="unknown",
            speaker_id="unknown",
            kind=SegmentKind.DIALOGUE,
            text="Hi there.",
            confidence=0.5,
            source="llm",
            evidence="",
        ),
        Segment(
            id="seg3",
            chapter=1,
            paragraph_id="p001",
            speaker="Harry",
            speaker_id="harry",
            kind=SegmentKind.DIALOGUE,
            text="How are you?",
            confidence=0.9,
            source="llm",
            evidence="",
        ),
    ]

    corrections = apply_alternation_correction(segments)

    assert len(corrections) == 1
    assert "seg2" in corrections
    speaker, confidence, evidence = corrections["seg2"]
    assert speaker == "harry"
    assert confidence == 0.75
    assert "alternation" in evidence


def test_alternation_does_not_touch_user_locked():
    """User-locked segments should never be corrected."""
    segments = [
        Segment(
            id="seg1",
            chapter=1,
            paragraph_id="p001",
            speaker="Harry",
            speaker_id="harry",
            kind=SegmentKind.DIALOGUE,
            text="Hello.",
            confidence=0.9,
            source="llm",
            evidence="",
        ),
        Segment(
            id="seg2",
            chapter=1,
            paragraph_id="p001",
            speaker="Ron",
            speaker_id="ron",
            kind=SegmentKind.DIALOGUE,
            text="Hi there.",
            confidence=0.5,
            source="user",  # User-locked
            evidence="",
        ),
        Segment(
            id="seg3",
            chapter=1,
            paragraph_id="p001",
            speaker="Harry",
            speaker_id="harry",
            kind=SegmentKind.DIALOGUE,
            text="How are you?",
            confidence=0.9,
            source="llm",
            evidence="",
        ),
    ]

    corrections = apply_alternation_correction(segments)

    assert len(corrections) == 0


def test_alternation_skips_high_confidence_middle():
    """High-confidence middle segments should not be corrected."""
    segments = [
        Segment(
            id="seg1",
            chapter=1,
            paragraph_id="p001",
            speaker="Harry",
            speaker_id="harry",
            kind=SegmentKind.DIALOGUE,
            text="Hello.",
            confidence=0.9,
            source="llm",
            evidence="",
        ),
        Segment(
            id="seg2",
            chapter=1,
            paragraph_id="p001",
            speaker="Ron",
            speaker_id="ron",
            kind=SegmentKind.DIALOGUE,
            text="Hi there.",
            confidence=0.8,  # Above max_confidence (0.75)
            source="llm",
            evidence="",
        ),
        Segment(
            id="seg3",
            chapter=1,
            paragraph_id="p001",
            speaker="Harry",
            speaker_id="harry",
            kind=SegmentKind.DIALOGUE,
            text="How are you?",
            confidence=0.9,
            source="llm",
            evidence="",
        ),
    ]

    corrections = apply_alternation_correction(segments)

    assert len(corrections) == 0


def test_alternation_requires_both_outer_segments_known():
    """If either outer segment is unknown, no correction should be applied."""
    segments = [
        Segment(
            id="seg1",
            chapter=1,
            paragraph_id="p001",
            speaker="unknown",
            speaker_id="unknown",
            kind=SegmentKind.DIALOGUE,
            text="Hello.",
            confidence=0.5,
            source="llm",
            evidence="",
        ),
        Segment(
            id="seg2",
            chapter=1,
            paragraph_id="p001",
            speaker="unknown",
            speaker_id="unknown",
            kind=SegmentKind.DIALOGUE,
            text="Hi there.",
            confidence=0.5,
            source="llm",
            evidence="",
        ),
        Segment(
            id="seg3",
            chapter=1,
            paragraph_id="p001",
            speaker="Harry",
            speaker_id="harry",
            kind=SegmentKind.DIALOGUE,
            text="How are you?",
            confidence=0.9,
            source="llm",
            evidence="",
        ),
    ]

    corrections = apply_alternation_correction(segments)

    assert len(corrections) == 0


def test_alternation_ignores_narration():
    """Narration segments should be ignored in alternation detection."""
    segments = [
        Segment(
            id="seg1",
            chapter=1,
            paragraph_id="p001",
            speaker="Harry",
            speaker_id="harry",
            kind=SegmentKind.DIALOGUE,
            text="Hello.",
            confidence=0.9,
            source="llm",
            evidence="",
        ),
        Segment(
            id="seg2",
            chapter=1,
            paragraph_id="p001",
            speaker="narrator",
            speaker_id="narrator",
            kind=SegmentKind.NARRATION,
            text="He walked away.",
            confidence=1.0,
            source="rule",
            evidence="",
        ),
        Segment(
            id="seg3",
            chapter=1,
            paragraph_id="p001",
            speaker="Harry",
            speaker_id="harry",
            kind=SegmentKind.DIALOGUE,
            text="Goodbye.",
            confidence=0.9,
            source="llm",
            evidence="",
        ),
    ]

    corrections = apply_alternation_correction(segments)

    assert len(corrections) == 0
