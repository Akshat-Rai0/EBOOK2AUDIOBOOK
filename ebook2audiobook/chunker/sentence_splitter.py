"""
Sentence splitter — splits paragraph text into individual sentences using spaCy.

**Why spaCy instead of simple regex splitting?**
Simple regex (split on ". ") fails on abbreviations ("Dr. Smith"), initials
("J.R.R. Tolkien"), and decimal numbers ("3.14").  spaCy's rule-based
sentence segmenter handles these correctly.
Analogy: like a careful editor who knows the difference between a period
ending a sentence and a period in "Mr." or "e.g."

**Offline use:**
``en_core_web_sm`` is a small statistical model (~15 MB) downloaded once with::

    python -m spacy download en_core_web_sm

It does NOT require network access at runtime after download.

**Fallback:**
If the spaCy model is not installed (e.g. in a stripped CI environment where
we cannot download models), the splitter falls back to a simple regex.
Tests mock spaCy to avoid requiring the model in CI.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Simple regex fallback: split on sentence-ending punctuation + whitespace.
_RE_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

_spacy_nlp = None
_spacy_load_attempted = False


def _get_nlp():
    """
    Lazily load the spaCy model.  Returns None if the model is not available.

    Lazy loading means the model is only loaded when the first sentence split
    is actually needed, and the import cost is not paid at module import time.
    """
    global _spacy_nlp, _spacy_load_attempted
    if _spacy_load_attempted:
        return _spacy_nlp
    _spacy_load_attempted = True
    try:
        import spacy

        _spacy_nlp = spacy.load(
            "en_core_web_sm",
            disable=["ner", "tagger", "parser", "lemmatizer", "attribute_ruler"],
        )
        # Enable the sentencizer (fast rule-based) instead of the DependencyParser.
        if "sentencizer" not in _spacy_nlp.pipe_names:
            _spacy_nlp.add_pipe("sentencizer")
        logger.debug("spaCy en_core_web_sm loaded successfully.")
    except (ImportError, OSError) as exc:
        logger.warning(
            "spaCy model 'en_core_web_sm' not available (%s). "
            "Using regex fallback. Run: python -m spacy download en_core_web_sm",
            exc,
        )
        _spacy_nlp = None
    return _spacy_nlp


def split_sentences(text: str) -> list[str]:
    """
    Split *text* into a list of sentences.

    Uses ``en_core_web_sm`` if available, otherwise falls back to a simple
    regex splitter.  The caller should not need to know which path was taken.

    Parameters
    ----------
    text:
        Cleaned paragraph text.

    Returns
    -------
    list[str]
        Ordered list of sentence strings, stripped of leading/trailing whitespace.
        Empty strings are excluded.
    """
    if not text.strip():
        return []

    nlp = _get_nlp()
    if nlp is not None:
        doc = nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    # Regex fallback.
    parts = _RE_SENTENCE_END.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def split_paragraphs(text: str) -> list[str]:
    """
    Split *text* into paragraph blocks (separated by one or more blank lines).

    Returns
    -------
    list[str]
        Ordered list of non-empty paragraph strings.
    """
    return [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
