"""
FakeTTS — a TTS engine that generates silence instead of speech.

**Why this exists:**
Real TTS models (XTTS-v2, VITS) are large and slow to load.  Tests and CI must
never download models or take minutes to run.  FakeTTS produces valid WAV bytes
(silence or a simple tone) in microseconds, so the full pipeline can be tested
end-to-end without any model.

Analogy: like a film stand-in who walks through scenes so the camera and lighting
can be set up, without the real actor needing to be on set.

``mode="silence"``  — every call produces a WAV of silence.
``mode="tone"``     — every call produces a 440 Hz sine wave (A4 note).
                      Useful when you want to hear that audio is being generated.
"""

from __future__ import annotations

import math
import struct

from ebook2audiobook.models.cast import VoiceRef
from ebook2audiobook.tts.engine import TTSEngine


class FakeTTS(TTSEngine):
    """
    A no-op TTS engine for tests and CI.

    Generates WAV bytes containing silence (or a tone) without loading any model.
    Duration is proportional to word count: 100 ms per word, minimum 200 ms.
    """

    _FAKE_VOICES = [
        {"id": "fake-narrator", "name": "Fake Narrator (silence)"},
        {"id": "fake-char-1", "name": "Fake Character 1 (silence)"},
        {"id": "fake-char-2", "name": "Fake Character 2 (silence)"},
    ]

    def __init__(
        self,
        sample_rate: int = 22050,
        mode: str = "silence",
    ) -> None:
        """
        Parameters
        ----------
        sample_rate:
            Output WAV sample rate in Hz.  22 050 is the default for VITS/XTTS.
        mode:
            ``"silence"`` (default) or ``"tone"`` (440 Hz sine wave).
        """
        if mode not in ("silence", "tone"):
            raise ValueError(f"mode must be 'silence' or 'tone', got {mode!r}")
        self._sample_rate = sample_rate
        self.mode = mode

    # ------------------------------------------------------------------
    # TTSEngine interface
    # ------------------------------------------------------------------

    @property
    def engine_name(self) -> str:
        return "fake"

    @property
    def max_chars(self) -> int:
        # No real limit — FakeTTS accepts any length.
        return 100_000

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def list_voices(self) -> list[dict]:
        return list(self._FAKE_VOICES)

    def synthesize(self, text: str, voice: VoiceRef) -> bytes:
        """
        Generate WAV bytes for *text* (silence or tone, no actual speech).

        The duration approximates real speech: 100 ms per word, minimum 200 ms.
        """
        if len(text) > self.max_chars:
            raise ValueError(f"Text length {len(text)} exceeds FakeTTS.max_chars={self.max_chars}")
        word_count = max(1, len(text.split()))
        duration_s = max(0.2, word_count * 0.1)
        num_samples = int(duration_s * self._sample_rate)

        if self.mode == "tone":
            pcm_data = self._generate_tone(num_samples, frequency=440.0)
        else:
            pcm_data = b"\x00\x00" * num_samples  # 16-bit silence

        return self._wrap_in_wav(pcm_data)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _generate_tone(self, num_samples: int, frequency: float) -> bytes:
        """Generate 16-bit PCM samples for a sine wave at *frequency* Hz."""
        samples = []
        for i in range(num_samples):
            value = int(32767 * math.sin(2 * math.pi * frequency * i / self._sample_rate))
            samples.append(struct.pack("<h", value))
        return b"".join(samples)

    def _wrap_in_wav(self, pcm_data: bytes) -> bytes:
        """Wrap raw 16-bit mono PCM bytes in a RIFF/WAV header."""
        data_size = len(pcm_data)
        num_channels = 1
        bits_per_sample = 16
        byte_rate = self._sample_rate * num_channels * bits_per_sample // 8
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
            self._sample_rate,
            byte_rate,
            block_align,
            bits_per_sample,
            b"data",
            data_size,
        )
        return header + pcm_data
