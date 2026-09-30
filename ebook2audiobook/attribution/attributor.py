"""
Abstract base class for speaker attribution (the LLM layer).

The Attributor receives one paragraph of text plus the cast known so far, and
returns a list of Segment-like dicts identifying who says what.

**Constrained decoding** (Week 4 concept):
Small local LLMs sometimes produce text that isn't valid JSON.  Constrained
decoding forces the model to only emit tokens that keep the output valid JSON,
the same way autocomplete on a phone only suggests real words.  Ollama's JSON
mode does this automatically.

**Retry + narrator fallback** (always implemented by concrete classes):
If the model returns invalid JSON after two retries, every span in the
paragraph is attributed to the narrator so the book can still be produced.
This is logged as a warning, not silently dropped.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Attributor(ABC):
    """Abstract base for speaker attribution backends."""

    @abstractmethod
    def attribute(
        self,
        paragraph: str,
        cast_so_far: list[str],
        context: str = "",
    ) -> list[dict]:
        """
        Attribute each span of *paragraph* to a speaker.

        Parameters
        ----------
        paragraph:
            The raw paragraph text (may contain dialogue and narration mixed).
        cast_so_far:
            List of canonical character names seen so far in the book.
            Passed to the LLM to bias towards known characters.
        context:
            Optional preceding paragraph for context (keeps LLM grounded).

        Returns
        -------
        list[dict]
            Each dict has keys:
              - ``text``    (str)   — the span of text.
              - ``speaker`` (str)   — canonical name, ``"narrator"``, or ``"unknown"``.
              - ``kind``    (str)   — ``"dialogue"``, ``"narration"``, or ``"thought"``.

        Raises
        ------
        AttributionError
            Raised only for unrecoverable errors (e.g., model not loaded).
            Temporary failures (bad JSON) are handled internally with retries
            and narrator fallback; they do NOT raise.
        """


class AttributionError(RuntimeError):
    """Raised for unrecoverable attribution failures (model crash, not loaded, etc.)."""
