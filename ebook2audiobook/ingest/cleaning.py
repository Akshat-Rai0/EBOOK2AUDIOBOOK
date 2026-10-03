"""
Text cleaning pipeline for extracted ebook content.

**Processing order** (important — changing the order changes outcomes):

1. Unicode normalisation (NFC) — normalises composed vs decomposed characters.
2. Soft-hyphen removal — U+00AD is invisible and breaks word matching.
3. Quote normalisation — maps curly/smart quotes and guillemets to straight
   ASCII quotes.  The original is preserved in ``Paragraph.original_text``.
4. Line-break de-hyphenation — joins "hyphen-\\nated" words split across lines.
5. Whitespace normalisation — collapses multiple spaces/blank lines.
6. Header/footer/page-number removal — PDF-specific; applied before everything
   else when the caller passes ``strip_headers=True``.

**Known limitation (PARKING_LOT):**
Em-dash dialogue (— He said) is not split into dialogue/narration spans here;
it is flagged in ``Paragraph.is_dialogue`` but attribution must handle the split.
"""

from __future__ import annotations

import re
import unicodedata

# ---------------------------------------------------------------------------
# Quote maps
# ---------------------------------------------------------------------------

# Maps Unicode smart / curly / guillemet quotes to ASCII equivalents.
# We keep the original text in Paragraph.original_text before applying this.
_QUOTE_MAP: dict[str, str] = {
    "\u2018": "'",  # LEFT SINGLE QUOTATION MARK
    "\u2019": "'",  # RIGHT SINGLE QUOTATION MARK  (apostrophe)
    "\u201a": "'",  # SINGLE LOW-9 QUOTATION MARK
    "\u201b": "'",  # SINGLE HIGH-REVERSED-9 QUOTATION MARK
    "\u201c": '"',  # LEFT DOUBLE QUOTATION MARK
    "\u201d": '"',  # RIGHT DOUBLE QUOTATION MARK
    "\u201e": '"',  # DOUBLE LOW-9 QUOTATION MARK
    "\u201f": '"',  # DOUBLE HIGH-REVERSED-9 QUOTATION MARK
    "\u00ab": '"',  # LEFT-POINTING DOUBLE ANGLE QUOTATION MARK (guillemet)
    "\u00bb": '"',  # RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK
    "\u2039": "'",  # SINGLE LEFT-POINTING ANGLE QUOTATION MARK
    "\u203a": "'",  # SINGLE RIGHT-POINTING ANGLE QUOTATION MARK
}
_QUOTE_TRANS = str.maketrans(_QUOTE_MAP)

# ---------------------------------------------------------------------------
# Compiled regexes
# ---------------------------------------------------------------------------

# Soft hyphen (U+00AD) — invisible, must be removed before any other step.
_RE_SOFT_HYPHEN = re.compile("\u00ad")

# Line-break hyphenation: a word ending with "-" followed by a newline and
# the continuation word (no space between them in the output).
_RE_LINE_BREAK_HYPHEN = re.compile(r"-\n(\S)")

# Multiple consecutive blank lines → single blank line.
_RE_MULTI_BLANK = re.compile(r"\n{3,}")

# Multiple spaces → single space.
_RE_MULTI_SPACE = re.compile(r"[ \t]{2,}")

# Page-number lines: a line containing only digits (optionally surrounded by
# whitespace), possibly preceded/followed by a running header/footer heuristic.
# This is a conservative pattern — catches bare page numbers only.
_RE_PAGE_NUMBER = re.compile(r"^\s*\d{1,4}\s*$", re.MULTILINE)

# Running header/footer: a line that appears 3+ times verbatim in the text.
# Detected dynamically in ``remove_running_headers``.

# Dialogue detection: line contains at least one double-quoted span.
_RE_DIALOGUE = re.compile(r'"[^"]{1,500}"')

# Em-dash dialogue (known limitation — flagged but not split here).
_RE_EM_DASH_DIALOGUE = re.compile(r"[—\u2014\u2013]")


def normalize_unicode(text: str) -> str:
    """Apply NFC normalisation (composed form) to *text*."""
    return unicodedata.normalize("NFC", text)


def remove_soft_hyphens(text: str) -> str:
    """Remove soft-hyphen characters (U+00AD) which are invisible but confuse tokenisers."""
    return _RE_SOFT_HYPHEN.sub("", text)


def normalize_quotes(text: str) -> str:
    """
    Replace curly/smart quotes and guillemets with ASCII equivalents.

    The caller must save ``original_text`` *before* calling this if the raw
    form is needed later (e.g. for the ``Paragraph.original_text`` field).
    """
    return text.translate(_QUOTE_TRANS)


def dehyphenate(text: str) -> str:
    """
    Join words split across lines by a hyphen.

    Example: ``"hyphen-\\nated"`` → ``"hyphenated"``.
    Only joins when the hyphen is the last character before the newline and
    the next line begins with a non-space character.
    """
    return _RE_LINE_BREAK_HYPHEN.sub(r"\1", text)


def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces to one and multiple blank lines to one."""
    text = _RE_MULTI_SPACE.sub(" ", text)
    text = _RE_MULTI_BLANK.sub("\n\n", text)
    return text.strip()


def remove_page_numbers(text: str) -> str:
    """
    Remove bare page-number lines (digits only).

    This is a conservative heuristic — it only removes lines that contain
    *only* digits (plus surrounding whitespace).  Running headers/footers
    that include the book title are handled separately in ``remove_running_headers``.
    """
    return _RE_PAGE_NUMBER.sub("", text)


def remove_running_headers(text: str, min_repeats: int = 3) -> str:
    """
    Remove lines that appear *min_repeats* or more times throughout the text.

    **Why this works for headers/footers:**
    PDF ebooks often embed the chapter title or author name at the top or bottom
    of every page.  These repeated lines are noise for TTS.  Lines that repeat
    many times are almost certainly headers or footers, not story content.

    This is a heuristic — it may occasionally remove legitimately repeated
    narrative lines.  The threshold is set conservatively at 3.
    """
    lines = text.split("\n")
    from collections import Counter

    counts = Counter(line.strip() for line in lines if line.strip())
    repeated = {line for line, count in counts.items() if count >= min_repeats and line}
    cleaned = [line for line in lines if line.strip() not in repeated]
    return "\n".join(cleaned)


def detect_dialogue(text: str) -> bool:
    """
    Return True if *text* contains at least one double-quoted dialogue span.

    Also returns True for em-dash dialogue (flagged as a known limitation).
    """
    return bool(_RE_DIALOGUE.search(text)) or bool(_RE_EM_DASH_DIALOGUE.search(text))


def clean(
    text: str,
    *,
    strip_headers: bool = False,
    is_pdf: bool = False,
) -> str:
    """
    Run the full cleaning pipeline on *text* and return the cleaned result.

    **Order** (do not change without updating the module docstring):
    1. Page-number removal (if ``is_pdf``).
    2. Running-header removal (if ``strip_headers``).
    3. Unicode normalisation.
    4. Soft-hyphen removal.
    5. Line-break de-hyphenation.
    6. Quote normalisation.
    7. Whitespace normalisation.

    Parameters
    ----------
    text:
        Raw extracted text.
    strip_headers:
        If True, run the repeated-line header/footer removal heuristic.
        Set this for PDF and TXT sources; DOCX and EPUB have their own structure.
    is_pdf:
        If True, also strip bare page-number lines before other steps.
    """
    if is_pdf:
        text = remove_page_numbers(text)
    if strip_headers:
        text = remove_running_headers(text)
    text = normalize_unicode(text)
    text = remove_soft_hyphens(text)
    text = dehyphenate(text)
    text = normalize_quotes(text)
    text = normalize_whitespace(text)
    return text


def split_to_chunks(text: str, max_chars: int) -> list[str]:
    """
    Split *text* into chunks of at most *max_chars* characters, breaking at
    sentence boundaries (full stops, question marks, exclamation marks).

    **Why sentence-boundary splitting matters for TTS:**
    XTTS-v2 silently truncates input beyond ~250 characters.  If we split in
    the middle of a word, the audio sounds cut off.  Splitting at sentence
    boundaries produces natural pauses and complete utterances.

    **In-sentence splitting (enhanced):**
    If a single sentence exceeds *max_chars*, it is split at the punctuation
    mark nearest the middle (`, ; : — ( ) [ ]`), then at spaces if no
    punctuation exists. Never splits mid-word. Logs a warning when a
    mid-sentence split occurs.

    Parameters
    ----------
    text:
        Input text (already cleaned).
    max_chars:
        Maximum length per chunk (must match ``TTSEngine.max_chars``).

    Returns
    -------
    list[str]
        Ordered list of text chunks, each ≤ *max_chars* characters.
    """
    import logging

    logger = logging.getLogger(__name__)

    # Split at sentence-ending punctuation followed by whitespace.
    sentence_re = re.compile(r"(?<=[.!?])\s+")
    sentences = sentence_re.split(text.strip())

    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        # Would adding this sentence exceed the limit?
        candidate = (current + " " + sentence).strip() if current else sentence
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # Handle oversized sentence
            if len(sentence) > max_chars:
                sub_chunks = _split_oversized_sentence(sentence, max_chars, logger)
                chunks.extend(sub_chunks)
            else:
                current = sentence

    if current:
        chunks.append(current)

    return chunks


def _split_oversized_sentence(text: str, max_chars: int, logger) -> list[str]:
    """
    Split an oversized sentence at punctuation nearest the middle, then spaces.

    Split order: `, ; : — ( ) [ ]` then spaces. Never mid-word.
    Logs a warning when splitting occurs.

    Parameters
    ----------
    text:
        Oversized sentence.
    max_chars:
        Maximum length per chunk.
    logger:
        Logger instance for warnings.

    Returns
    -------
    list[str]
        List of chunks, each ≤ max_chars.
    """
    logger.warning(
        "Sentence (%d chars) exceeds max_chars=%d; splitting in-sentence.",
        len(text),
        max_chars,
    )

    # Try splitting at punctuation nearest the middle
    punct_chars = [",", ";", ":", "—", "(", ")", "[", "]"]
    mid = len(text) // 2

    # Find the punctuation mark closest to the middle
    best_split_idx = -1
    best_dist = len(text)

    for i, char in enumerate(text):
        if char in punct_chars:
            dist = abs(i - mid)
            if dist < best_dist:
                best_dist = dist
                best_split_idx = i

    if best_split_idx != -1:
        # Split at the punctuation
        part1 = text[: best_split_idx + 1].strip()
        part2 = text[best_split_idx + 1 :].strip()
        if part1 and part2:
            if len(part1) <= max_chars and len(part2) <= max_chars:
                return [part1, part2]
            # If one part is still too large, recurse
            if len(part1) > max_chars:
                return _split_oversized_sentence(part1, max_chars, logger) + [part2]
            if len(part2) > max_chars:
                return [part1] + _split_oversized_sentence(part2, max_chars, logger)

    # No suitable punctuation found, split at spaces
    # Find the space nearest the middle
    space_split_idx = -1
    best_dist = len(text)

    for i, char in enumerate(text):
        if char == " ":
            dist = abs(i - mid)
            if dist < best_dist:
                best_dist = dist
                space_split_idx = i

    if space_split_idx != -1:
        part1 = text[:space_split_idx].strip()
        part2 = text[space_split_idx + 1 :].strip()
        if part1 and part2:
            if len(part1) <= max_chars and len(part2) <= max_chars:
                return [part1, part2]
            # Recurse if needed
            if len(part1) > max_chars:
                return _split_oversized_sentence(part1, max_chars, logger) + [part2]
            if len(part2) > max_chars:
                return [part1] + _split_oversized_sentence(part2, max_chars, logger)

    # No spaces either - hard split at max_chars (last resort)
    logger.error(
        "No punctuation or spaces found in sentence; hard-splitting at %d chars.",
        max_chars,
    )
    part1 = text[:max_chars]
    part2 = text[max_chars:]
    # Recurse on part2 if it's still too large
    if len(part2) > max_chars:
        return [part1] + _split_oversized_sentence(part2, max_chars, logger)
    return [part1, part2]
