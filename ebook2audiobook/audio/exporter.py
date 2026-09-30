"""
Audio exporter: Converts WAV files into chapter MP3s and creates an M4B file with chapter markers.

**Formats:**
- Chapter MP3: Individual MP3 files for each chapter (``chapter_01.mp3``).
- M4B: Single audiobook file with standard metadata chapters (``book.m4b``).

Uses FFmpeg when installed for fast encoding and metadata tagging.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class AudioExporter:
    """Exports WAV files to chapter MP3s and unified M4B with chapter markers."""

    def __init__(self, ffmpeg_path: str | None = None) -> None:
        self.ffmpeg_path = ffmpeg_path or shutil.which("ffmpeg") or "ffmpeg"

    def has_ffmpeg(self) -> bool:
        """Check if FFmpeg executable is accessible on the system."""
        return shutil.which(self.ffmpeg_path) is not None

    def wav_to_mp3(self, in_wav: Path, out_mp3: Path, bitrate: str = "128k") -> Path:
        """
        Convert WAV to MP3 using FFmpeg.

        If FFmpeg is not installed, creates a placeholder or copies the file with warning.
        """
        in_wav, out_mp3 = Path(in_wav), Path(out_mp3)
        out_mp3.parent.mkdir(parents=True, exist_ok=True)

        if not self.has_ffmpeg():
            logger.warning("FFmpeg not found; writing WAV copy with .mp3 suffix for fallback.")
            shutil.copy(in_wav, out_mp3)
            return out_mp3

        cmd = [
            self.ffmpeg_path,
            "-y",
            "-i",
            str(in_wav),
            "-codec:a",
            "libmp3lame",
            "-b:a",
            bitrate,
            str(out_mp3),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("FFmpeg error encoding MP3: %s", result.stderr)
            raise RuntimeError(f"FFmpeg MP3 export failed: {result.stderr}")

        return out_mp3

    def build_m4b(
        self,
        chapter_wavs: list[tuple[str, Path, float]],
        out_m4b: Path,
        title: str = "Audiobook",
        author: str = "Author",
    ) -> Path:
        """
        Create a single M4B file with embedded chapter markers.

        Parameters
        ----------
        chapter_wavs:
            List of tuples ``(chapter_title, wav_path, duration_seconds)``.
        out_m4b:
            Destination path for the .m4b file.
        title:
            Book title.
        author:
            Book author.
        """
        out_m4b = Path(out_m4b)
        out_m4b.parent.mkdir(parents=True, exist_ok=True)

        if not self.has_ffmpeg() or not chapter_wavs:
            logger.warning("FFmpeg not found or no chapters provided; creating stub M4B.")
            out_m4b.write_bytes(b"STUB_M4B")
            return out_m4b

        temp_dir = out_m4b.parent / "_m4b_build"
        temp_dir.mkdir(parents=True, exist_ok=True)

        # 1. Create ffmpeg concat list file
        concat_list = temp_dir / "concat_list.txt"
        with open(concat_list, "w", encoding="utf-8") as f:
            for _, wav_path, _ in chapter_wavs:
                f.write(f"file '{wav_path.resolve()}'\n")

        # 2. Build FFmetadata file for chapters
        metadata_file = temp_dir / "ffmetadata.txt"
        metadata_lines = [
            ";FFMETADATA1",
            f"title={title}",
            f"artist={author}",
            f"album={title}",
            "",
        ]

        current_time_ms = 0
        for ch_title, _, duration_sec in chapter_wavs:
            duration_ms = int(duration_sec * 1000)
            end_time_ms = current_time_ms + duration_ms
            metadata_lines.extend(
                [
                    "[CHAPTER]",
                    "TIMEBASE=1/1000",
                    f"START={current_time_ms}",
                    f"END={end_time_ms}",
                    f"title={ch_title}",
                    "",
                ]
            )
            current_time_ms = end_time_ms

        metadata_file.write_text("\n".join(metadata_lines), encoding="utf-8")

        # 3. Concatenate and encode to AAC/M4B with metadata
        cmd = [
            self.ffmpeg_path,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list),
            "-i",
            str(metadata_file),
            "-map_metadata",
            "1",
            "-codec:a",
            "aac",
            "-b:a",
            "128k",
            str(out_m4b),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        shutil.rmtree(temp_dir, ignore_errors=True)

        if result.returncode != 0:
            logger.error("FFmpeg error building M4B: %s", result.stderr)
            raise RuntimeError(f"FFmpeg M4B export failed: {result.stderr}")

        return out_m4b
