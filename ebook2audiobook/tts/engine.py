"""
Abstract base class for TTS (Text-to-Speech) engines.

**What this interface guarantees:**
Every engine accepts text + a ``VoiceRef`` and returns raw WAV bytes.
Splitting long text into chunks at sentence boundaries is the *caller's*
responsibility (the ``synthesize`` contract is for one chunk ≤ ``max_chars``).

**Adding a new engine** means writing a new adapter that implements this ABC.
The rest of the pipeline never imports engine-specific code directly.

Concepts:
  ``max_chars``  — the engine's per-call character limit (e.g. 250 for XTTS-v2).
                   Analogy: like a text-message character limit — longer text
                   must be split into separate messages.
  ``sample_rate`` — audio samples per second; 22 050 Hz is CD-quality speech.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ebook2audiobook.models.cast import VoiceRef


class TTSEngine(ABC):
    """Abstract base for all TTS synthesis engines."""

    @property
    @abstractmethod
    def engine_name(self) -> str:
        """Short identifier matching ``VoiceRef.engine``, e.g. ``'fake'``."""

    @property
    @abstractmethod
    def max_chars(self) -> int:
        """
        Maximum characters this engine accepts in a single ``synthesize`` call.

        The caller must split longer text before calling ``synthesize``.
        Use ``split_to_chunks`` from ``ebook2audiobook.ingest.cleaning`` for this.
        """

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Output WAV sample rate in Hz."""

    @abstractmethod
    def list_voices(self) -> list[dict]:
        """
        Return available voices.

        Each entry is a dict with at least ``{"id": str, "name": str}``.
        """

    @abstractmethod
    def synthesize(self, text: str, voice: str | VoiceRef) -> bytes:
        """
        Synthesise *text* in the given *voice* and return WAV bytes.

        Parameters
        ----------
        text:
            Plain text to speak.  Must be ≤ ``max_chars`` characters.
            Must not contain SSML or markup.
        voice:
            VoiceRef pointing to the specific speaker, or a string voice ID.
            For convenience, adapters accept both VoiceRef and str.

        Returns
        -------
        bytes
            Raw WAV file bytes (RIFF header + PCM data).

        Raises
        ------
        TTSError
            On synthesis failure (model error, bad voice id, etc.).
        ValueError
            If ``len(text) > max_chars``.
        """


class TTSError(RuntimeError):
    """Raised when a TTS engine fails to synthesise a segment."""
