"""Automatic style tagging for quotes.

Analogy: like categorizing test questions by type (multiple choice, essay,
short answer) to see which question formats are hardest.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import StrEnum

logger = logging.getLogger(__name__)


class QuoteStyle(StrEnum):
    """Style category for a quote."""

    TAGGED = "tagged"  # Has explicit dialogue tag (e.g., "said NAME")
    PRONOUN_TAGGED = "pronoun_tagged"  # Has pronoun tag (e.g., "he said")
    UNTAGGED = "untagged"  # No dialogue tag, speaker inferred from context
    ALTERNATING = "alternating"  # Two-person rapid dialogue with few tags
    THREE_OR_MORE = "three_or_more"  # Scene with 3+ speakers
    FIRST_PERSON = "first_person"  # First-person narration (epistolary)


@dataclass
class StyleAnalysis:
    """Result of style analysis for a chapter."""

    total_quotes: int
    style_counts: dict[str, int]
    manual_overrides: dict[str, str]  # quote_id -> style if manually set


def detect_style(
    quote_text: str,
    context_before: str,
    context_after: str,
    speakers_in_window: set[str],
) -> QuoteStyle:
    """
    Detect the style category for a single quote.

    Parameters
    ----------
    quote_text:
        The quoted text itself.
    context_before:
        Text immediately before the quote.
    context_after:
        Text immediately after the quote.
    speakers_in_window:
        Set of speaker IDs seen in the surrounding 20 segments.

    Returns
    -------
    QuoteStyle
        The detected style category.
    """
    # Check for explicit dialogue tag with name
    # Pattern: quote + ", said NAME" or "NAME said," + quote
    name_pattern = r"\b[A-Z][a-zA-Z]+\b"
    tag_verbs = (
        "said|asked|replied|answered|whispered|shouted|yelled|cried|"
        "muttered|murmured|exclaimed|continued|added|remarked|noted|"
        "observed|commented|declared|stated|explained|argued|insisted|"
        "agreed|disagreed|promised|warned|threatened|begged|pleaded|"
        "laughed|sighed|groaned|snapped|growled"
    )
    tag_pattern_with_name = rf"[,;:]\s+({tag_verbs})\s+{name_pattern}"
    tag_pattern_name_first = rf"{name_pattern}\s+({tag_verbs})[,;:]\s+\""

    if re.search(tag_pattern_with_name, context_after, re.IGNORECASE):
        return QuoteStyle.TAGGED
    if re.search(tag_pattern_name_first, context_before, re.IGNORECASE):
        return QuoteStyle.TAGGED

    # Check for pronoun tag (he said, she replied, etc.)
    pronoun_pattern = rf"\b(he|she|they|it)\s+({tag_verbs})[,;:]\s+\""
    pronoun_pattern_first = rf"\b(he|she|they|it)\s+({tag_verbs})[,;:]"

    if re.search(pronoun_pattern, context_after, re.IGNORECASE):
        return QuoteStyle.PRONOUN_TAGGED
    if re.search(pronoun_pattern_first, context_before, re.IGNORECASE):
        return QuoteStyle.PRONOUN_TAGGED

    # Check for first-person (I as speaker)
    if "I" in speakers_in_window:
        return QuoteStyle.FIRST_PERSON

    # Check for three-or-more speakers
    if len(speakers_in_window) >= 3:
        return QuoteStyle.THREE_OR_MORE

    # Check for alternating pattern (heuristic: rapid quote alternation)
    # This is harder to detect from a single quote; needs sequence context
    # For now, classify as untagged if no other category matches
    return QuoteStyle.UNTAGGED


def analyze_chapter_styles(
    quotes: list[dict[str, str]],
    context_map: dict[str, tuple[str, str]],
    speaker_map: dict[str, str],
    window_size: int = 20,
) -> tuple[StyleAnalysis, dict[str, QuoteStyle]]:
    """
    Analyze style categories for all quotes in a chapter.

    Parameters
    ----------
    quotes:
        List of quote dicts with keys: quote_id, text, paragraph_id.
    context_map:
        Mapping from paragraph_id to (text_before, text_after).
    speaker_map:
        Mapping from segment_id to speaker_id.
    window_size:
        Number of surrounding segments to consider for speaker count.

    Returns
    -------
    tuple
        (StyleAnalysis, dict of quote_id -> QuoteStyle)
    """
    style_counts: dict[str, int] = {}
    quote_styles: dict[str, QuoteStyle] = {}

    # Build speaker-in-window map
    speakers_by_quote: dict[str, set[str]] = {}
    # For simplicity, use all speakers from the chapter
    all_speakers = set(speaker_map.values())
    for quote in quotes:
        speakers_by_quote[quote["quote_id"]] = all_speakers

    for quote in quotes:
        quote_id = quote["quote_id"]
        quote_text = quote["text"]
        paragraph_id = quote.get("paragraph_id", "")

        # Get context
        before, after = context_map.get(paragraph_id, ("", ""))
        speakers_in_window = speakers_by_quote.get(quote_id, set())

        # Detect style
        style = detect_style(quote_text, before, after, speakers_in_window)
        quote_styles[quote_id] = style
        style_counts[style.value] = style_counts.get(style.value, 0) + 1

    analysis = StyleAnalysis(
        total_quotes=len(quotes),
        style_counts=style_counts,
        manual_overrides={},  # To be populated if user wants to override
    )
    return analysis, quote_styles


def apply_manual_overrides(
    analysis: StyleAnalysis,
    overrides: dict[str, str],
    quote_styles: dict[str, QuoteStyle],
) -> StyleAnalysis:
    """
    Apply manual style overrides to an analysis.

    Parameters
    ----------
    analysis:
        Original style analysis.
    overrides:
        Mapping from quote_id to manual style string.
    quote_styles:
        Mapping from quote_id to detected QuoteStyle.

    Returns
    -------
    StyleAnalysis
        Updated analysis with overrides applied.
    """
    # Update style counts after overrides
    new_style_counts = analysis.style_counts.copy()

    for quote_id, old_style in {q: s.value for q, s in quote_styles.items()}.items():
        if quote_id in overrides:
            new_style = overrides[quote_id]
            # Decrement old count
            new_style_counts[old_style] = new_style_counts.get(old_style, 0) - 1
            # Increment new count
            new_style_counts[new_style] = new_style_counts.get(new_style, 0) + 1

    return StyleAnalysis(
        total_quotes=analysis.total_quotes,
        style_counts=new_style_counts,
        manual_overrides=overrides,
    )
