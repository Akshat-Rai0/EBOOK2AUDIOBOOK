"""Quote normalisation and span finding.

Analogy: before a proofreader can find speech, every kind of quotation mark
(curly, straight, German-style) is rewritten to one standard pen.
"""
from __future__ import annotations

import logging
from typing import NamedTuple

from ebook2audiobook.models.segment import SegmentKind

logger = logging.getLogger(__name__)


class QuoteStyle(NamedTuple):
    """Detected quote style for a paragraph."""

    style: str  # "double" or "single"


def normalise_double_style(text: str) -> str:
    """Curly double quotes -> straight '"'. Curly apostrophes -> straight "'"."""
    out: list[str] = []
    for ch in text:
        if ch == "\u201c" or ch == "\u201d" or ch == "\u201e":
            out.append('"')
        elif ch == "\u2018" or ch == "\u2019":
            out.append("'")
        else:
            out.append(ch)
    return "".join(out)


def normalise_single_style(text: str) -> str:
    """Normalise single quotes: ' ' ' ' -> straight ' and handle apostrophes.

    Apostrophe (') between two letters is kept (didn't, Harry's).
    Opening ' at start or after whitespace/punctuation is opening quote.
    Closing ' is a closing quote only if a quote is open and next char is not a letter.
    """
    out: list[str] = []
    for i, ch in enumerate(text):
        if ch == "\u2018" or ch == "\u2019":
            # Check if it's an apostrophe (between letters)
            if i > 0 and i < len(text) - 1:
                prev = text[i - 1]
                next_ch = text[i + 1]
                if prev.isalpha() and next_ch.isalpha():
                    out.append("'")  # Apostrophe
                    continue
            # Otherwise it's a quote mark
            out.append("'")
        else:
            out.append(ch)
    return "".join(out)


def detect_quote_style(paragraph_texts: list[str]) -> QuoteStyle:
    """Detect whether text uses double or single quotes by counting."""
    double_count = 0
    single_count = 0

    for text in paragraph_texts:
        for i, ch in enumerate(text):
            if ch in ('"', "\u201c", "\u201d", "\u201e"):
                # Count if it's at start or after whitespace
                if i == 0 or text[i - 1].isspace():
                    double_count += 1
            elif ch in ("'", "\u2018", "\u2019"):
                if i == 0 or text[i - 1].isspace():
                    single_count += 1

    return QuoteStyle(style="double" if double_count >= single_count else "single")


def find_spans(text: str) -> tuple[list[tuple[str, SegmentKind]], bool]:
    """Split one paragraph into (text, kind) spans.

    Returns (spans, ends_inside_quote). ends_inside_quote is True when the
    paragraph opens a quote and never closes it (a speech continuing into
    the next paragraph, a normal convention in novels).

    Note: Spans are not stripped - whitespace is preserved for round-trip
    correctness. The dialogue_segmenter handles stripping at the segment level.
    """
    spans: list[tuple[str, SegmentKind]] = []
    buf: list[str] = []
    inside = False
    for ch in text:
        if ch == '"':
            if inside:
                buf.append(ch)
                piece = "".join(buf)
                if piece:
                    spans.append((piece, SegmentKind.DIALOGUE))
                buf = []
                inside = False
            else:
                piece = "".join(buf)
                if piece:
                    spans.append((piece, SegmentKind.NARRATION))
                buf = [ch]
                inside = True
        else:
            buf.append(ch)
    tail = "".join(buf)
    if tail:
        kind = SegmentKind.DIALOGUE if inside else SegmentKind.NARRATION
        spans.append((tail, kind))
    return spans, inside
