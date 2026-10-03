"""
Synthesis text normaliser — expands numbers and abbreviations for TTS.

**Why this is separate from ingest cleaning:**
Ingest cleaning (ingest/cleaning.py) normalises text *before* it's stored in book.json.
This normaliser runs *only* during synthesis, so the original text remains unchanged
for resume/debugging purposes.

Analogy: like a director's notes that tell the actor how to say a line, without
changing the script itself.

**Rules:**
- Ellipses → comma (pause word)
- Em/en dashes → space-dash-space
- Titles (Mr., Mrs., Dr., St.) → expanded form
- Integers → words (42 → forty-two)
- Years 1100-2099 → spoken form (1999 → nineteen ninety-nine)
- All-caps words length ≥ 4 → title case (WAIT → Wait, but not NASA)
- Stray symbols (* _ # ~) → removed
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compiled regexes
# ---------------------------------------------------------------------------

# Ellipses: ... or … (single character)
_RE_ELLIPS = re.compile(r"\.{3,}|…")

# Em and en dashes
_RE_EM_DASH = re.compile(r"[—–]")

# Titles with word boundaries (avoid matching inside words)
_RE_TITLES = re.compile(
    r"\b(Mr\.|Mrs\.|Ms\.|Dr\.|St\.)\b"
)

# Integers (standalone numbers, not part of larger alphanumeric)
_RE_INTEGER = re.compile(r"\b\d+\b")

# Years: 4 digits in range 1100-2099
_RE_YEAR = re.compile(r"\b(1[1-9]\d\d|20\d\d)\b")

# All-caps words length ≥ 4 (but not common acronyms)
_RE_ALL_CAPS = re.compile(r"\b[A-Z]{4,}\b")

# Common acronyms to leave as-is (extend as needed)
_ACRONYMS = {
    "NASA",
    "USA",
    "UK",
    "US",
    "USSR",
    "EU",
    "UN",
    "WHO",
    "BBC",
    "CNN",
    "CIA",
    "FBI",
    "NASA",
    "RAM",
    "CPU",
    "GPU",
    "AI",
    "API",
    "HTML",
    "XML",
    "JSON",
    "SQL",
    "GUI",
    "PDF",
    "MP3",
    "WAV",
    "URL",
    "HTTP",
    "HTTPS",
    "FTP",
}

# Stray symbols to remove
_RE_STRAY_SYMBOLS = re.compile(r"[*_#~]")

# Title expansions
_TITLE_EXPANSIONS = {
    "Mr.": "Mister",
    "Mrs.": "Missus",
    "Ms.": "Miss",
    "Dr.": "Doctor",
    "St.": "Saint",
}


def normalise_for_synthesis(text: str) -> str:
    """
    Apply synthesis normalisation rules to *text*.

    This is called inside the synthesize() method of TTS adapters.
    The original Segment.text is not modified.

    Parameters
    ----------
    text:
        Text to normalise.

    Returns
    -------
    str
        Normalised text suitable for TTS synthesis.

    Raises
    ------
    ValueError
        If the result is empty after normalisation.
    """
    if not text or not text.strip():
        raise ValueError("Text is empty before normalisation")

    result = text

    # 1. Ellipses → comma (pause word)
    result = _RE_ELLIPS.sub(", ", result)

    # 2. Em/en dashes → space-dash-space
    result = _RE_EM_DASH.sub(" - ", result)

    # 3. Expand titles
    result = _RE_TITLES.sub(lambda m: _TITLE_EXPANSIONS.get(m.group(), m.group()), result)

    # 4. Convert integers to words
    result = _RE_INTEGER.sub(_replace_integer, result)

    # 5. Convert years to spoken form
    result = _RE_YEAR.sub(_replace_year, result)

    # 6. All-caps to title case (but not acronyms)
    result = _RE_ALL_CAPS.sub(_replace_all_caps, result)

    # 7. Remove stray symbols
    result = _RE_STRAY_SYMBOLS.sub("", result)

    # Clean up extra spaces
    result = re.sub(r"\s+", " ", result).strip()

    if not result:
        raise ValueError("Text is empty after normalisation")

    return result


def _replace_integer(match: re.Match) -> str:
    """Replace an integer with its word form."""
    try:
        from num2words import num2words

        num = int(match.group())
        return num2words(num)
    except ImportError:
        # Fallback: return as-is if num2words not available
        logger.debug("num2words not available, leaving integer as-is")
        return match.group()


def _replace_year(match: re.Match) -> str:
    """Replace a year (1100-2099) with its spoken form."""
    year = int(match.group())

    if 1100 <= year < 2000:
        # 1100-1999: "nineteen ninety-nine"
        thousands = year // 100
        remainder = year % 100
        if remainder == 0:
            # 1900 → "nineteen hundred"
            try:
                from num2words import num2words

                return num2words(thousands) + " hundred"
            except ImportError:
                return match.group()
        else:
            # 1999 → "nineteen ninety-nine"
            try:
                from num2words import num2words

                return num2words(thousands) + " " + num2words(remainder)
            except ImportError:
                return match.group()
    else:
        # 2000-2099: "two thousand twenty-four"
        thousands = year // 1000
        remainder = year % 1000
        try:
            from num2words import num2words

            if remainder == 0:
                return num2words(thousands) + " thousand"
            else:
                return num2words(thousands) + " thousand " + num2words(remainder)
        except ImportError:
            return match.group()


def _replace_all_caps(match: re.Match) -> str:
    """Replace an all-caps word with title case, unless it's a known acronym."""
    word = match.group()
    if word in _ACRONYMS:
        return word
    return word.title()
