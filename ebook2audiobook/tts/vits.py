"""
VITS-VCTK TTS adapter for CastBook.

Uses the idiap/coqui-ai-TTS fork with the VITS-VCTK multi-speaker model.
Model: tts_models/en/vctk/vits (109 speakers, MIT licence).

**Speaker metadata:**
- Coqui's VITS-VCTK speaker IDs (p225 … p376) do not reliably match VCTK
  speaker-info.txt gender/accent (Coqui issue #2258).
- This adapter returns only {id, name} for now. Gender/accent will be added
  in M4 after listen-labelling.
- Narrator maps to p225 by default (configurable via config.toml).

**Usage:**
    from ebook2audiobook.tts.vits import VITS
    from ebook2audiobook.models.cast import VoiceRef

    tts = VITS()
    voices = tts.list_voices()
    voice = VoiceRef(engine="vits", voice_id="p225")
    wav = tts.synthesize("Hello world", voice)
"""

from __future__ import annotations

import logging
import struct

from ebook2audiobook.config import get_vits_narrator
from ebook2audiobook.models.cast import VoiceRef
from ebook2audiobook.tts.engine import TTSEngine, TTSError

logger = logging.getLogger(__name__)

# Lazy import of TTS to avoid requiring the tts extra for basic imports
_TTS = None


def _get_tts():
    """Lazy import of TTS library."""
    global _TTS
    if _TTS is not None:
        return _TTS
    try:
        from TTS.api import TTS

        _TTS = TTS
        return _TTS
    except ImportError as exc:
        raise TTSError(
            "coqui-tts not installed. Run: uv sync --extra tts"
        ) from exc


class VITS(TTSEngine):
    """
    VITS-VCTK TTS adapter using idiap/coqui-ai-TTS.

    Model: tts_models/en/vctk/vits (109 speakers, 22.05 kHz output).
    """

    _MODEL_NAME = "tts_models/en/vctk/vits"
    _PRACTICAL_MAX_CHARS = 500  # Conservative cap; increased from 400 to handle normalized text expansion

    def __init__(self, device: str = "cpu") -> None:
        """
        Parameters
        ----------
        device:
            Device to run on: "cpu" or "cuda" (MPS not supported for VITS).
        """
        self._device = device
        self._tts_api = None
        self._model = None
        self._speaker_ids = None

    def _load_model(self) -> None:
        """Lazy-load the VITS model on first use."""
        if self._model is not None:
            return

        TTS = _get_tts()  # noqa: N806
        logger.info(f"Loading VITS model: {self._MODEL_NAME} on {self._device}")
        self._tts_api = TTS(self._MODEL_NAME).to(self._device)
        self._model = self._tts_api.synthesizer
        # Speaker IDs are in tts_model.speaker_manager.speaker_names
        speaker_manager = self._model.tts_model.speaker_manager
        self._speaker_ids = speaker_manager.speaker_names if speaker_manager else []
        logger.info(f"VITS loaded with {len(self._speaker_ids)} speakers")

    @property
    def engine_name(self) -> str:
        return "vits"

    @property
    def max_chars(self) -> int:
        return self._PRACTICAL_MAX_CHARS

    @property
    def sample_rate(self) -> int:
        """Read sample rate from loaded config (expected 22050 Hz)."""
        self._load_model()
        # VITS outputs at 22050 Hz (hardcoded in the model)
        return 22050

    def list_voices(self) -> list[dict]:
        """
        Return available VITS speakers.

        Each entry is a dict with {id, name}. Gender/accent are not included
        due to Coqui issue #2258 (IDs don't reliably match VCTK speaker-info.txt).

        Returns
        -------
        list[dict]
            List of speaker dictionaries.
        """
        self._load_model()
        voices = []
        for speaker_id in self._speaker_ids:
            voices.append({"id": speaker_id, "name": f"Speaker {speaker_id}"})
        return voices

    def synthesize(self, text: str, voice: str | VoiceRef) -> bytes:
        """
        Synthesise *text* in the given *voice* and return WAV bytes.

        Parameters
        ----------
        text:
            Plain text to speak. Must be ≤ max_chars.
        voice:
            VoiceRef with engine="vits" and voice_id (e.g., "p225"),
            or a string speaker ID.
            Special case: voice_id="narrator" maps to the configured default.

        Returns
        -------
        bytes
            Raw WAV file bytes (RIFF header + PCM data).

        Raises
        ------
        TTSError
            On synthesis failure or unknown voice.
        ValueError
            If len(text) > max_chars.
        """
        if len(text) > self.max_chars:
            raise ValueError(f"Text length {len(text)} exceeds max_chars={self.max_chars}")

        if not text.strip():
            raise TTSError("Cannot synthesize empty text")

        self._load_model()

        # Accept either VoiceRef or string speaker ID
        if isinstance(voice, VoiceRef):
            speaker_id = voice.voice_id
        else:
            speaker_id = voice

        # Resolve narrator to the configured default speaker
        if speaker_id == "narrator":
            speaker_id = get_vits_narrator()
            logger.debug(f"Resolved narrator to VITS speaker: {speaker_id}")

        # Validate speaker ID
        if speaker_id not in self._speaker_ids:
            available = ", ".join(self._speaker_ids[:5]) + "..."
            raise TTSError(
                f"Unknown VITS speaker: {speaker_id}. Available: {available}"
            )

        # Synthesize
        try:
            wav = self._tts_api.tts(text=text, speaker=speaker_id)
        except Exception as exc:
            raise TTSError(f"VITS synthesis failed: {exc}") from exc

        # Convert numpy array to WAV bytes
        import numpy as np

        if isinstance(wav, list):
            wav = np.array(wav)

        # Ensure 16-bit PCM
        if wav.dtype != np.int16:
            wav = (wav * 32767).astype(np.int16)

        pcm_data = wav.tobytes()
        return self._wrap_in_wav(pcm_data, self.sample_rate)

    def _wrap_in_wav(self, pcm_data: bytes, sample_rate: int) -> bytes:
        """Wrap raw 16-bit mono PCM bytes in a RIFF/WAV header."""
        data_size = len(pcm_data)
        num_channels = 1
        bits_per_sample = 16
        byte_rate = sample_rate * num_channels * bits_per_sample // 8
        block_align = num_channels * bits_per_sample // 8

        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",
            36 + data_size,
            b"WAVE",
            b"fmt ",
            16,  # subchunk1 size (PCM)
            1,  # audio format: 1 = PCM
            num_channels,
            sample_rate,
            byte_rate,
            block_align,
            bits_per_sample,
            b"data",
            data_size,
        )
        return header + pcm_data
