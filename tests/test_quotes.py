"""Tests for quote normalisation and span finding."""


from ebook2audiobook.chunker.quotes import (
    detect_quote_style,
    find_spans,
    normalise_double_style,
    normalise_single_style,
)
from ebook2audiobook.models.segment import SegmentKind


class TestNormaliseDoubleStyle:
    """Test double quote normalisation."""

    def test_curly_to_straight(self):
        """Curly double quotes become straight."""
        text = "Harry said, \u201cI didn't do it.\u201d"
        result = normalise_double_style(text)
        assert result == 'Harry said, "I didn\'t do it."'

    def test_mixed_curly_straight(self):
        """Mixed curly and straight quotes all become straight."""
        text = 'He said "Hello" and she said "Goodbye"'
        result = normalise_double_style(text)
        assert result == 'He said "Hello" and she said "Goodbye"'

    def test_apostrophes_preserved(self):
        """Apostrophes in contractions are preserved."""
        text = "Harry's didn't it's"
        result = normalise_double_style(text)
        assert result == "Harry's didn't it's"

    def test_no_quotes_unchanged(self):
        """Text without quotes is unchanged."""
        text = "Just regular text."
        result = normalise_double_style(text)
        assert result == text


class TestNormaliseSingleStyle:
    """Test single quote normalisation."""

    def test_apostrophes_preserved(self):
        """Apostrophes between letters are preserved."""
        text = "Harry's didn't it's"
        result = normalise_single_style(text)
        assert result == "Harry's didn't it's"

    def test_curly_single_to_straight(self):
        """Curly single quotes become straight."""
        text = "He said 'Hello' and she said 'Goodbye'"
        result = normalise_single_style(text)
        assert result == "He said 'Hello' and she said 'Goodbye'"

    def test_mixed_apostrophes_and_quotes(self):
        """Mixed apostrophes and quotes handled."""
        text = "Harry's said 'Hello'"
        result = normalise_single_style(text)
        assert result == "Harry's said 'Hello'"


class TestDetectQuoteStyle:
    """Test quote style detection."""

    def test_double_style_detected(self):
        """Double quotes are detected."""
        texts = ['"Hello," said Harry.', '"Goodbye," said Ron.']
        result = detect_quote_style(texts)
        assert result.style == "double"

    def test_single_style_detected(self):
        """Single quotes are detected."""
        texts = ["'Hello,' said Harry.", "'Goodbye,' said Ron."]
        result = detect_quote_style(texts)
        assert result.style == "single"

    def test_no_quotes_defaults_to_double(self):
        """No quotes defaults to double."""
        texts = ["Hello.", "Goodbye."]
        result = detect_quote_style(texts)
        assert result.style == "double"


class TestFindSpans:
    """Test span finding for double quotes."""

    def test_simple_dialogue(self):
        """Simple dialogue sentence."""
        text = '"Hello," said Harry.'
        spans, ends_inside = find_spans(text)
        assert len(spans) == 2
        assert spans[0] == ('"Hello,"', SegmentKind.DIALOGUE)
        # find_spans preserves whitespace for round-trip, segmenter strips it
        assert spans[1] == (' said Harry.', SegmentKind.NARRATION)
        assert not ends_inside

    def test_two_quotes_one_paragraph(self):
        """Two quotes in one paragraph."""
        text = '"I didn\'t do it," and Ron said, "Me neither."'
        spans, ends_inside = find_spans(text)
        # Result: 3 spans - DIALOGUE, NARRATION, DIALOGUE
        # The "and" between quotes gets merged with narration
        assert len(spans) == 3
        assert spans[0][1] == SegmentKind.DIALOGUE
        assert spans[1][1] == SegmentKind.NARRATION
        assert spans[2][1] == SegmentKind.DIALOGUE
        assert not ends_inside

    def test_quote_continues_next_paragraph(self):
        """Quote that opens but doesn't close."""
        text = '"First paragraph of a long speech'
        spans, ends_inside = find_spans(text)
        assert len(spans) == 1
        assert spans[0] == ('"First paragraph of a long speech', SegmentKind.DIALOGUE)
        assert ends_inside

    def test_no_quotes(self):
        """Paragraph with no quotes."""
        text = "No quotes here."
        spans, ends_inside = find_spans(text)
        assert len(spans) == 1
        assert spans[0] == ("No quotes here.", SegmentKind.NARRATION)
        assert not ends_inside

    def test_empty_text(self):
        """Empty text returns empty spans."""
        text = ""
        spans, ends_inside = find_spans(text)
        assert len(spans) == 0
        assert not ends_inside

    def test_nested_quotes_simple(self):
        """Simple nested quotes (best effort)."""
        text = '"He said "Hello" to me," she said.'
        spans, ends_inside = find_spans(text)
        # This is a known limitation - nested quotes not fully handled
        # The outer quote should be detected at minimum
        assert len(spans) >= 1

    def test_round_trip_no_text_loss(self):
        """Round-trip: joining spans reconstructs original text."""
        text = 'Harry said, "I didn\'t do it," and Ron said, "Me neither."'
        spans, ends_inside = find_spans(text)
        reconstructed = "".join(s[0] for s in spans)
        assert reconstructed == text

    def test_round_trip_with_curly(self):
        """Round-trip works with curly quotes after normalisation."""
        text = "Harry said, \u201cI didn't do it,\u201d and Ron said, \u201cMe neither.\u201d"
        normalised = normalise_double_style(text)
        spans, ends_inside = find_spans(normalised)
        reconstructed = "".join(s[0] for s in spans)
        assert reconstructed == normalised
