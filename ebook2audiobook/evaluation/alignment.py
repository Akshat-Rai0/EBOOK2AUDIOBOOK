"""Quote alignment between gold labels and production DialogueSegmenter.

Analogy: like checking that a proofreader's markup matches the original text,
ensuring every highlighted quote corresponds to an actual quote in the source.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

from ebook2audiobook.chunker.quotes import find_spans, normalise_double_style
from ebook2audiobook.models.segment import SegmentKind

logger = logging.getLogger(__name__)


@dataclass
class QuoteSpan:
    """A single quote span extracted from text."""

    quote_id: str  # e.g., "Q1", "Q2"
    text: str
    start_char: int  # Character index in normalized text
    end_char: int


@dataclass
class AlignmentResult:
    """Result of aligning gold labels with extractor output."""

    total_gold_quotes: int
    total_extractor_quotes: int
    matched_quotes: int
    missed_quotes: list[str]  # Gold quote IDs not found by extractor
    invented_quotes: list[str]  # Extractor quote IDs not in gold
    precision: float  # matched / total_extractor_quotes
    recall: float  # matched / total_gold_quotes
    f1: float  # Harmonic mean of precision and recall


def extract_quotes_from_text(text: str) -> list[QuoteSpan]:
    """
    Extract quote spans from text using production DialogueSegmenter logic.

    Returns list of QuoteSpan objects with IDs [Q1], [Q2], etc.
    """
    # Normalize quotes (curly to straight)
    normalized = normalise_double_style(text)

    # Use find_spans to get quote spans
    spans, _ = find_spans(normalized)

    # Extract only dialogue spans
    quote_spans: list[QuoteSpan] = []
    quote_idx = 0
    char_offset = 0

    for span_text, kind in spans:
        span_len = len(span_text)
        if kind == SegmentKind.DIALOGUE:
            quote_id = f"Q{quote_idx + 1}"
            # Find the actual quote boundaries in the span
            # (find_spans includes quotes but we need to locate the quote marks)
            stripped = span_text.strip()
            if stripped.startswith('"') and stripped.endswith('"'):
                # Strip the quote marks for the text content
                content = stripped[1:-1]
                # Find where this content starts in the original span
                content_start = span_text.find(content)
                if content_start >= 0:
                    start = char_offset + content_start
                    end = start + len(content)
                    quote_spans.append(
                        QuoteSpan(
                            quote_id=quote_id,
                            text=content,
                            start_char=start,
                            end_char=end,
                        )
                    )
                    quote_idx += 1
        char_offset += span_len

    return quote_spans


def align_gold_with_extractor(
    gold_file: Path,
    raw_text: str,
) -> AlignmentResult:
    """
    Align gold labels with production DialogueSegmenter output.

    Parameters
    ----------
    gold_file:
        Path to gold JSON file with labels.
    raw_text:
        Raw chapter text (before cleaning/segmentation).

    Returns
    -------
    AlignmentResult
        Precision, recall, and lists of missed/invented quotes.
    """
    import json

    # Load gold labels
    with open(gold_file) as f:
        gold_data = json.load(f)

    gold_quotes = gold_data.get("quotes", [])
    gold_quote_ids = {q["quote_id"] for q in gold_quotes}

    # Extract quotes using production logic
    extractor_quotes = extract_quotes_from_text(raw_text)
    extractor_quote_ids = {q.quote_id for q in extractor_quotes}

    # Find matches (exact ID match)
    matched = gold_quote_ids & extractor_quote_ids

    # Missed: in gold but not in extractor
    missed = list(gold_quote_ids - extractor_quote_ids)

    # Invented: in extractor but not in gold
    invented = list(extractor_quote_ids - gold_quote_ids)

    # Calculate metrics
    total_gold = len(gold_quote_ids)
    total_extractor = len(extractor_quote_ids)
    total_matched = len(matched)

    precision = total_matched / total_extractor if total_extractor > 0 else 0.0
    recall = total_matched / total_gold if total_gold > 0 else 0.0

    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    return AlignmentResult(
        total_gold_quotes=total_gold,
        total_extractor_quotes=total_extractor,
        matched_quotes=total_matched,
        missed_quotes=sorted(missed),
        invented_quotes=sorted(invented),
        precision=precision,
        recall=recall,
        f1=f1,
    )


def compute_text_hash(text: str) -> str:
    """Compute SHA-256 hash of text for verification."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def verify_source_integrity(
    gold_file: Path,
    raw_text: str,
) -> bool:
    """
    Verify that the raw text matches the hash stored in gold file.

    Returns True if hash matches, False otherwise.
    """
    import json

    with open(gold_file) as f:
        gold_data = json.load(f)

    stored_hash = gold_data.get("source_hash")
    if not stored_hash:
        logger.warning(f"No source_hash in {gold_file}")
        return False

    computed_hash = compute_text_hash(raw_text)
    return stored_hash == computed_hash
