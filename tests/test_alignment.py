"""Unit tests for quote alignment module."""

from ebook2audiobook.evaluation.alignment import (
    AlignmentResult,
    extract_quotes_from_text,
)


def test_extract_quotes_simple():
    """Test quote extraction from simple dialogue."""
    text = '"Hello," said Alice. "Hi there," Bob replied.'

    quotes = extract_quotes_from_text(text)

    assert len(quotes) == 2
    assert quotes[0].quote_id == "Q1"
    assert quotes[0].text == "Hello,"
    assert quotes[1].quote_id == "Q2"
    assert quotes[1].text == "Hi there,"


def test_extract_quotes_no_dialogue():
    """Test extraction from text with no quotes."""
    text = "The room was empty. No one spoke."

    quotes = extract_quotes_from_text(text)

    assert len(quotes) == 0


def test_extract_quotes_multi_paragraph():
    """Test extraction handles quotes spanning paragraphs."""
    text = '"Hello," Alice began. Then she continued, "I wonder if he will come."'

    quotes = extract_quotes_from_text(text)

    # Two separate quotes in one paragraph
    assert len(quotes) == 2


def test_extract_quotes_curly_quotes():
    """Test that curly quotes are normalized."""
    text = "\u201cHello,\u201d she said."  # Curly quotes

    quotes = extract_quotes_from_text(text)

    assert len(quotes) == 1
    assert quotes[0].text == "Hello,"


def test_alignment_result_metrics():
    """Test AlignmentResult metric calculations."""
    result = AlignmentResult(
        total_gold_quotes=10,
        total_extractor_quotes=10,
        matched_quotes=8,
        missed_quotes=["Q5"],
        invented_quotes=["Q11"],
        precision=0.8,
        recall=0.8,
        f1=0.8,
    )

    assert result.precision == 0.8
    assert result.recall == 0.8
    assert result.f1 == 0.8


def test_alignment_result_zero_division():
    """Test AlignmentResult handles zero division."""
    result = AlignmentResult(
        total_gold_quotes=0,
        total_extractor_quotes=0,
        matched_quotes=0,
        missed_quotes=[],
        invented_quotes=[],
        precision=0.0,
        recall=0.0,
        f1=0.0,
    )

    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.f1 == 0.0
