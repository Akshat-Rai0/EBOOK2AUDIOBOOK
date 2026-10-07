"""Tests for OllamaAttributor V2 (ID-based attribution)."""

from ebook2audiobook.attribution.attributor import QuoteAttribution


def test_quote_attribution_dataclass():
    """Test QuoteAttribution dataclass structure."""
    attr = QuoteAttribution(quote_id="Q1", speaker_id="harry", confidence=0.95)

    assert attr.quote_id == "Q1"
    assert attr.speaker_id == "harry"
    assert attr.confidence == 0.95


def test_quote_attribution_fields_required():
    """Test that QuoteAttribution requires all fields."""
    try:
        QuoteAttribution(quote_id="Q1", speaker_id="harry")  # Missing confidence
        assert False, "Should have raised TypeError"
    except TypeError:
        pass  # Expected
