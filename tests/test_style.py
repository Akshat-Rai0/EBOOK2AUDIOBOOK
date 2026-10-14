"""Unit tests for style tagging module."""

from ebook2audiobook.evaluation.style import (
    QuoteStyle,
    StyleAnalysis,
    analyze_chapter_styles,
    apply_manual_overrides,
    detect_style,
)


def test_detect_style_tagged():
    """Test detection of tagged quotes."""
    quote = "Hello"
    before = ""
    after = '", said Alice.'
    speakers = {"alice", "bob"}

    style = detect_style(quote, before, after, speakers)

    assert style == QuoteStyle.TAGGED


def test_detect_style_pronoun_tagged():
    """Test detection of pronoun-tagged quotes."""
    quote = "Hello"
    before = "He said, "
    after = "."
    speakers = {"narrator"}

    style = detect_style(quote, before, after, speakers)

    assert style == QuoteStyle.PRONOUN_TAGGED


def test_detect_style_untagged():
    """Test detection of untagged quotes."""
    quote = "Hello"
    before = "She looked at him. "
    after = " He smiled."
    speakers = {"alice", "bob"}

    style = detect_style(quote, before, after, speakers)

    assert style == QuoteStyle.UNTAGGED


def test_detect_style_first_person():
    """Test detection of first-person quotes."""
    quote = "I'll be there"
    before = ""
    after = "."
    speakers = {"I"}

    style = detect_style(quote, before, after, speakers)

    assert style == QuoteStyle.FIRST_PERSON


def test_detect_style_three_or_more():
    """Test detection of 3+ speaker scenes."""
    quote = "We agree"
    before = ""
    after = "."
    speakers = {"alice", "bob", "charlie"}

    style = detect_style(quote, before, after, speakers)

    assert style == QuoteStyle.THREE_OR_MORE


def test_analyze_chapter_styles():
    """Test chapter-level style analysis."""
    quotes = [
        {"quote_id": "Q1", "text": "Hello", "paragraph_id": "p1"},
        {"quote_id": "Q2", "text": "Hi there", "paragraph_id": "p1"},
    ]
    context_map = {"p1": ("", '", said Alice.')}
    speaker_map = {"s1": "alice", "s2": "bob"}

    analysis, _ = analyze_chapter_styles(quotes, context_map, speaker_map)

    assert analysis.total_quotes == 2
    assert analysis.style_counts["tagged"] == 2


def test_apply_manual_overrides():
    """Test manual style overrides."""
    from ebook2audiobook.evaluation.style import QuoteStyle

    analysis = StyleAnalysis(
        total_quotes=2,
        style_counts={"tagged": 2, "untagged": 0},
        manual_overrides={},
    )
    quote_styles = {"Q1": QuoteStyle.TAGGED, "Q2": QuoteStyle.TAGGED}
    overrides = {"Q1": "untagged"}

    updated = apply_manual_overrides(analysis, overrides, quote_styles)

    assert updated.style_counts["tagged"] == 1
    assert updated.style_counts["untagged"] == 1
    assert updated.manual_overrides == overrides
