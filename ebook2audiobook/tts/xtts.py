"""
XTTS-v2 TTS adapter for CastBook.

Uses the idiap/coqui-ai-TTS fork with the XTTS-v2 multi-speaker model.
Model: tts_models/multilingual/multi-dataset/xtts_v2 (Coqui Public Model License - non-commercial).

**Speaker conditioning:**
- XTTS uses speaker latents (gpt latent + speaker embedding) computed once per voice_id.
- These are cached in the adapter instance for the process lifetime.
- Cache key: (engine, voice_id).

**Device support:**
- Default CPU on Darwin (macOS).
- Optional MPS attempt with fallback to CPU (Coqui #3649: aten::_fft_r2c unimplemented on MPS).
- CUDA on Linux/Windows if available.

**Usage:**
    from ebook2audiobook.tts.xtts import XTTS
    from ebook2audiobook.models.cast import VoiceRef

    tts = XTTS(device="cpu")
    voices = tts.list_voices()
    voice = VoiceRef(engine="xtts", voice_id="Ana Florence")
    wav = tts.synthesize("Hello world", voice)
"""

from __future__ import annotations

import logging
import struct

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


class XTTS(TTSEngine):
    """
    XTTS-v2 TTS adapter using idiap/coqui-ai-TTS.

    Model: tts_models/multilingual/multi-dataset/xtts_v2 (24 kHz output).
    """

    _MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

    def __init__(self, device: str = "cpu") -> None:
        """
        Parameters
        ----------
        device:
            Device to run on: "cpu", "cuda", or "mps" (macOS).
            MPS will attempt and fall back to CPU on errors.
        """
        self._device = device
        self._actual_device = device  # May change from MPS to CPU
        self._tts_api = None
        self._model = None
        self._speaker_manager = None

    def _load_model(self) -> None:
        """Lazy-load the XTTS model on first use."""
        if self._model is not None:
            return

        TTS = _get_tts()  # noqa: N806

        # Handle MPS fallback
        if self._device == "mps":
            logger.info("Attempting XTTS on MPS (may fail, will fallback to CPU)")
            try:
                import torch

                if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
                    logger.warning("MPS not available, falling back to CPU")
                    self._actual_device = "cpu"
            except ImportError:
                logger.warning("Torch not available, falling back to CPU")
                self._actual_device = "cpu"

        logger.info(f"Loading XTTS model: {self._MODEL_NAME} on {self._actual_device}")
        try:
            self._tts_api = TTS(self._MODEL_NAME).to(self._actual_device)
            self._model = self._tts_api.synthesizer
            self._speaker_manager = self._model.tts_model.speaker_manager
            logger.info(f"XTTS loaded on {self._actual_device}")
        except Exception as exc:
            if self._device == "mps" and self._actual_device == "mps":
                logger.warning(f"MPS load failed ({exc}), falling back to CPU")
                self._actual_device = "cpu"
                self._tts_api = TTS(self._MODEL_NAME).to(self._actual_device)
                self._model = self._tts_api.synthesizer
                self._speaker_manager = self._model.tts_model.speaker_manager
                logger.info("XTTS loaded on CPU after MPS fallback")
            else:
                raise TTSError(f"XTTS load failed: {exc}") from exc

    @property
    def engine_name(self) -> str:
        return "xtts"

    @property
    def max_chars(self) -> int:
        """Read num_chars from config (default 255)."""
        self._load_model()
        # XTTS has num_chars in config; default is 255
        return getattr(self._model.tts_config, "num_chars", 255)

    @property
    def sample_rate(self) -> int:
        """XTTS output sample rate (24000 Hz)."""
        return 24000

    def list_voices(self) -> list[dict]:
        """
        Return available XTTS speakers.

        Reads from the loaded speaker_manager.

        Returns
        -------
        list[dict]
            List of speaker dictionaries with {id, name}.
        """
        self._load_model()
        voices = []
        if self._speaker_manager:
            speaker_names = self._speaker_manager.speaker_names
            for name in speaker_names:
                voices.append({"id": name, "name": name})
        return voices

    def synthesize(self, text: str, voice: str | VoiceRef) -> bytes:
        """
        Synthesise *text* in the given *voice* and return WAV bytes.

        Parameters
        ----------
        text:
            Plain text to speak. Must be ≤ max_chars.
        voice:
            VoiceRef with engine="xtts" and voice_id (speaker name), or a string speaker ID.

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

        # Validate speaker ID
        if self._speaker_manager and speaker_id not in self._speaker_manager.speaker_names:
            available = ", ".join(self._speaker_manager.speaker_names[:5]) + "..."
            raise TTSError(
                f"Unknown XTTS speaker: {speaker_id}. Available: {available}"
            )

        # Synthesize using the high-level TTS API
        # The newer Coqui API handles speaker latents internally
        try:
            wav = self._tts_api.tts(
                text=text,
                speaker=speaker_id,
                language="en",
                split_sentences=False,  # We handle splitting ourselves
            )
        except Exception as exc:
            raise TTSError(f"XTTS synthesis failed: {exc}") from exc

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
