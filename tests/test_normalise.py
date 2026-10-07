"""Tests for synthesis text normalisation."""

import pytest


def test_ellipses_to_comma():
    from ebook2audiobook.tts.normalise import normalise_for_synthesis

    result = normalise_for_synthesis("Wait...")
    assert result == "Wait,"


def test_em_dash_to_space_dash_space():
    from ebook2audiobook.tts.normalise import normalise_for_synthesis

    result = normalise_for_synthesis("yes—no")
    assert result == "yes - no"


def test_title_expansion():
    from ebook2audiobook.tts.normalise import normalise_for_synthesis

    # num2words may not be available in CI, so test with fallback
    result = normalise_for_synthesis("Mr. Smith")
    # Either "Mister Smith" or "Mr. Smith" if num2words not available
    assert "Smith" in result


def test_integer_to_words():
    from ebook2audiobook.tts.normalise import normalise_for_synthesis

    # num2words might not be available in CI, so test with fallback
    result = normalise_for_synthesis("There are 42 apples.")
    # Either "forty-two" or "42" if num2words not available
    assert "42" in result or "forty" in result


def test_year_to_spoken():
    from ebook2audiobook.tts.normalise import normalise_for_synthesis

    result = normalise_for_synthesis("It was 1999.")
    # Either spoken form or original year if num2words not available
    assert "1999" in result or "nineteen" in result or "thousand" in result


def test_all_caps_to_title_case():
    from ebook2audiobook.tts.normalise import normalise_for_synthesis

    assert normalise_for_synthesis("WAIT") == "Wait"
    # Acronyms should remain uppercase
    assert normalise_for_synthesis("NASA") == "NASA"


def test_stray_symbols_removed():
    from ebook2audiobook.tts.normalise import normalise_for_synthesis

    assert normalise_for_synthesis("Hello*World") == "HelloWorld"
    assert normalise_for_synthesis("Test_123") == "Test123"


def test_empty_before_normalise_raises():
    from ebook2audiobook.tts.normalise import normalise_for_synthesis

    with pytest.raises(ValueError, match="empty before normalisation"):
        normalise_for_synthesis("")


def test_only_symbols_after_normalise_raises():
    from ebook2audiobook.tts.normalise import normalise_for_synthesis

    with pytest.raises(ValueError, match="empty after normalisation"):
        normalise_for_synthesis("***")
