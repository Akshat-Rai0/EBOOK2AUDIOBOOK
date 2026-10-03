"""Tests for the text cleaning pipeline."""

from __future__ import annotations

from ebook2audiobook.ingest.cleaning import (
    clean,
    dehyphenate,
    detect_dialogue,
    normalize_quotes,
    normalize_unicode,
    normalize_whitespace,
    remove_page_numbers,
    remove_running_headers,
    remove_soft_hyphens,
    split_to_chunks,
)


class TestNormalizeUnicode:
    def test_nfc_composition(self):
        # é as decomposed (e + combining accent) → composed
        decomposed = "e\u0301"
        result = normalize_unicode(decomposed)
        assert result == "\u00e9"

    def test_already_nfc_unchanged(self):
        text = "Hello, world!"
        assert normalize_unicode(text) == text


class TestRemoveSoftHyphens:
    def test_removes_soft_hyphen(self):
        text = "hyphen\u00adated"
        assert remove_soft_hyphens(text) == "hyphenated"

    def test_no_soft_hyphen_unchanged(self):
        text = "no hyphens here"
        assert remove_soft_hyphens(text) == text


class TestNormalizeQuotes:
    def test_curly_double_quotes(self):
        text = "\u201cHello\u201d"
        assert normalize_quotes(text) == '"Hello"'

    def test_curly_single_quotes(self):
        text = "\u2018Hello\u2019"
        assert normalize_quotes(text) == "'Hello'"

    def test_guillemets(self):
        text = "\u00abHello\u00bb"
        assert normalize_quotes(text) == '"Hello"'

    def test_mixed_quotes(self):
        text = "\u201cShe said, \u2018yes\u2019\u201d"
        assert normalize_quotes(text) == "\"She said, 'yes'\""


class TestDehyphenate:
    def test_joins_hyphenated_line_break(self):
        text = "hyphen-\nated"
        assert dehyphenate(text) == "hyphenated"

    def test_preserves_mid_word_hyphen(self):
        text = "well-known"
        assert dehyphenate(text) == "well-known"

    def test_preserves_hyphen_before_space(self):
        text = "end-of-chapter\n  Next sentence."
        # The newline is followed by a space, so should not be joined.
        assert dehyphenate(text) == text


class TestNormalizeWhitespace:
    def test_collapses_spaces(self):
        assert normalize_whitespace("hello   world") == "hello world"

    def test_collapses_blank_lines(self):
        text = "para one\n\n\n\npara two"
        assert normalize_whitespace(text) == "para one\n\npara two"

    def test_strips_ends(self):
        assert normalize_whitespace("  hello  ") == "hello"


class TestRemovePageNumbers:
    def test_removes_bare_digit_line(self):
        text = "Paragraph text.\n42\nMore text."
        result = remove_page_numbers(text)
        assert "42" not in result.split("\n")
        # But multi-digit numbers embedded in text should not be removed.
        text2 = "There were 42 apples."
        assert "42" in remove_page_numbers(text2)

    def test_removes_padded_number(self):
        text = "Para.\n   123   \nPara."
        result = remove_page_numbers(text)
        assert "123" not in result


class TestRemoveRunningHeaders:
    def test_removes_repeated_line(self):
        line = "THE GREAT NOVEL"
        text = f"{line}\nParagraph.\n{line}\nAnother.\n{line}\nEnd."
        result = remove_running_headers(text, min_repeats=3)
        assert line not in result

    def test_keeps_non_repeated_line(self):
        text = "Once.\nTwice.\nThrice.\nFour times."
        result = remove_running_headers(text, min_repeats=5)
        assert result == text


class TestClean:
    def test_full_pipeline(self):
        raw = "\u201cHello\u201d \u2018world\u2019!\nThis is hyphen-\nated."
        result = clean(raw)
        assert '"' in result  # curly quotes normalised
        assert "hyphenated" in result

    def test_is_pdf_strips_page_numbers(self):
        text = "Chapter text.\n7\nMore text."
        result = clean(text, is_pdf=True)
        assert "\n7\n" not in result

    def test_idempotent(self):
        raw = "Hello world. \u201cShe said.\u201d"
        once = clean(raw)
        twice = clean(once)
        assert once == twice


class TestDetectDialogue:
    def test_detects_double_quotes(self):
        assert detect_dialogue('"Hello," she said.')

    def test_detects_em_dash(self):
        assert detect_dialogue("— He answered.")

    def test_no_dialogue(self):
        assert not detect_dialogue("He walked across the room.")


class TestSplitToChunks:
    def test_short_text_single_chunk(self):
        text = "Hello world. How are you?"
        chunks = split_to_chunks(text, max_chars=200)
        assert len(chunks) == 1
        assert chunks[0] == text.strip()

    def test_splits_at_sentence_boundary(self):
        sentence_a = "A" * 50 + "."
        sentence_b = "B" * 50 + "."
        text = sentence_a + " " + sentence_b
        chunks = split_to_chunks(text, max_chars=60)
        # Each chunk should be one sentence.
        assert len(chunks) == 2
        assert chunks[0] == sentence_a
        assert chunks[1] == sentence_b

    def test_oversized_sentence_split(self):
        """A sentence longer than max_chars is now split at punctuation or spaces."""
        # Split at comma
        long_sent = "A" * 50 + ", " + "B" * 50 + ", " + "C" * 50 + "."
        chunks = split_to_chunks(long_sent, max_chars=100)
        assert len(chunks) > 1
        # Each chunk should be ≤ max_chars
        for chunk in chunks:
            assert len(chunk) <= 100

    def test_split_at_comma_nearest_middle(self):
        """Split at punctuation mark nearest the middle."""
        text = "A" * 40 + ", " + "B" * 40 + ", " + "C" * 40 + "."
        chunks = split_to_chunks(text, max_chars=100)
        # Should split at one of the commas
        assert len(chunks) == 2
        for chunk in chunks:
            assert len(chunk) <= 100

    def test_split_at_em_dash(self):
        """Split at em-dash."""
        text = "A" * 60 + "—" + "B" * 60 + "."
        chunks = split_to_chunks(text, max_chars=100)
        assert len(chunks) == 2
        for chunk in chunks:
            assert len(chunk) <= 100

    def test_split_at_brackets(self):
        """Split at brackets."""
        text = "A" * 40 + " ( " + "B" * 40 + " ) " + "C" * 40 + "."
        chunks = split_to_chunks(text, max_chars=100)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 100

    def test_split_at_spaces_fallback(self):
        """Split at spaces when no punctuation exists."""
        text = "A" * 50 + " " + "B" * 50 + " " + "C" * 50
        chunks = split_to_chunks(text, max_chars=100)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 100

    def test_never_mid_word_split(self):
        """Never split in the middle of a word."""
        # Long word without spaces should hard-split at max_chars and recurse
        text = "A" * 300
        chunks = split_to_chunks(text, max_chars=100)
        # Hard-split produces 3 chunks (100, 100, 100)
        assert len(chunks) == 3
        # Each chunk should be exactly max_chars or less
        for chunk in chunks:
            assert len(chunk) <= 100

    def test_empty_text(self):
        assert split_to_chunks("", max_chars=100) == []
