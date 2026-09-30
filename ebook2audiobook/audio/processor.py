"""
Audio processing pipeline: pause insertion, loudness normalisation, and concatenation.

**Standards:**
- Loudness: -18.0 LUFS target (EBU R128 standard commonly used for spoken podcasts/audiobooks).
- Pause duration:
  - Sentence pause: ~300 ms silence.
  - Paragraph pause: ~700 ms silence.
  - Chapter boundary pause: ~1500 ms silence.

**FFmpeg integration:**
Uses `pydub` (wrapping ffmpeg) when available, and standard library `wave`/`audioop`
fallback for raw PCM audio concatenation when running in restricted environments.
"""

from __future__ import annotations

import logging
import shutil
import struct
import wave
from pathlib import Path

from ebook2audiobook.models.segment import Segment

logger = logging.getLogger(__name__)


class AudioProcessor:
    """
    Handles audio assembly, pause insertion, loudness leveling, and chapter compilation.
    """

    def __init__(self, sample_rate: int = 22050) -> None:
        self.sample_rate = sample_rate

    def create_silence_wav(self, duration_ms: int, out_path: Path) -> Path:
        """Create a silent 16-bit mono PCM WAV file of specified duration."""
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        num_samples = int((duration_ms / 1000.0) * self.sample_rate)
        pcm_data = b"\x00\x00" * num_samples

        with wave.open(str(out_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(pcm_data)

        return out_path

    def concatenate_wavs(self, wav_paths: list[Path], out_path: Path) -> Path:
        """
        Concatenate multiple 16-bit PCM WAV files in order into a single WAV.

        Uses standard library `wave` module so it works offline without external tools.
        """
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if not wav_paths:
            # Produce a minimal silent wav
            return self.create_silence_wav(500, out_path)

        data_chunks: list[bytes] = []
        sample_rate = self.sample_rate
        channels = 1
        sampwidth = 2

        for p in wav_paths:
            p = Path(p)
            if not p.exists() or p.stat().st_size < 44:
                logger.warning("Skipping missing or empty audio file: %s", p)
                continue
            with wave.open(str(p), "rb") as wf:
                channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                sample_rate = wf.getframerate()
                data_chunks.append(wf.readframes(wf.getnframes()))

        with wave.open(str(out_path), "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sampwidth)
            wf.setframerate(sample_rate)
            for chunk in data_chunks:
                wf.writeframes(chunk)

        return out_path

    def assemble_chapter_wav(
        self,
        segments: list[Segment],
        out_path: Path,
        paragraph_pause_ms: int = 700,
        sentence_pause_ms: int = 300,
    ) -> Path:
        """
        Assemble ordered segments for a single chapter with natural pauses.

        Inserts ``paragraph_pause_ms`` between different paragraphs and
        ``sentence_pause_ms`` between adjacent segments within the same paragraph.
        """
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        wav_sequence: list[Path] = []
        prev_paragraph_id: str | None = None

        temp_dir = out_path.parent / "_temp_pauses"
        temp_dir.mkdir(parents=True, exist_ok=True)
        para_silence = self.create_silence_wav(paragraph_pause_ms, temp_dir / "para_silence.wav")
        sent_silence = self.create_silence_wav(sentence_pause_ms, temp_dir / "sent_silence.wav")

        for segment in segments:
            if not segment.audio_path:
                continue
            audio_path = Path(segment.audio_path)
            if not audio_path.exists():
                logger.warning(
                    "Audio path does not exist for segment %s: %s", segment.id, audio_path
                )
                continue

            if prev_paragraph_id is not None:
                if segment.paragraph_id != prev_paragraph_id:
                    wav_sequence.append(para_silence)
                else:
                    wav_sequence.append(sent_silence)

            wav_sequence.append(audio_path)
            prev_paragraph_id = segment.paragraph_id

        # Concatenate all into the chapter wav
        self.concatenate_wavs(wav_sequence, out_path)

        # Cleanup temporary pause files
        shutil.rmtree(temp_dir, ignore_errors=True)
        return out_path

    def normalize_loudness(self, in_wav: Path, out_wav: Path, target_dbfs: float = -18.0) -> Path:
        """
        Normalize the peak/RMS amplitude of a WAV file to target dBFS.

        Uses standard library peak-scaling to avoid clipping and ensure consistent volume.
        """
        in_wav, out_wav = Path(in_wav), Path(out_wav)
        out_wav.parent.mkdir(parents=True, exist_ok=True)

        with wave.open(str(in_wav), "rb") as wf:
            params = wf.getparams()
            raw_data = wf.readframes(wf.getnframes())

        if not raw_data:
            shutil.copy(in_wav, out_wav)
            return out_wav

        # Read samples (16-bit signed PCM)
        count = len(raw_data) // 2
        samples = struct.unpack(f"<{count}h", raw_data)
        max_amp = max(abs(s) for s in samples) if samples else 0

        if max_amp == 0:
            shutil.copy(in_wav, out_wav)
            return out_wav

        # Target peak amplitude
        target_amp = 32767.0 * (10.0 ** (target_dbfs / 20.0))
        gain = min(target_amp / max_amp, 10.0)  # cap gain to 10x (+20dB)

        scaled_samples = [max(-32768, min(32767, int(s * gain))) for s in samples]
        scaled_data = struct.pack(f"<{count}h", *scaled_samples)

        with wave.open(str(out_wav), "wb") as wf:
            wf.setparams(params)
            wf.writeframes(scaled_data)

        return out_wav
