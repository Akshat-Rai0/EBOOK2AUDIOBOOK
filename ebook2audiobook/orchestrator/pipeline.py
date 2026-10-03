"""
State machine and pipeline orchestrator for audiobook conversion jobs.

Pipeline stages for Narrator-Only (M2):
1. ``ingest``: Parses ebook file to book.json.
2. ``segmentation``: Splits paragraphs into synthesizable segments.
3. ``synthesis``: Generates audio per segment via TTSEngine, tracking progress in state.sqlite.
4. ``assembly``: Combines segment audio with natural pauses per chapter.
5. ``export``: Encodes chapter MP3s and unified M4B with chapter markers.
"""

from __future__ import annotations

import logging
import time
import wave
from collections.abc import Callable
from pathlib import Path

from ebook2audiobook.audio.exporter import AudioExporter
from ebook2audiobook.audio.processor import AudioProcessor
from ebook2audiobook.chunker.sentence_splitter import split_sentences
from ebook2audiobook.ingest.cleaning import split_to_chunks
from ebook2audiobook.models.book import Book
from ebook2audiobook.models.cast import VoiceRef
from ebook2audiobook.models.job import ConversionMode, Job, StageStatus
from ebook2audiobook.models.segment import Segment, SegmentKind, SegmentStatus
from ebook2audiobook.store.db import JobDatabase
from ebook2audiobook.tts.engine import TTSEngine

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Coordinates job execution, segment synthesis, pause insertion, and export."""

    def __init__(
        self,
        project_dir: Path,
        tts_engine: TTSEngine,
        mode: ConversionMode = ConversionMode.NARRATOR_ONLY,
    ) -> None:
        self.project_dir = Path(project_dir)
        self.audio_dir = self.project_dir / "audio"
        self.output_dir = self.project_dir / "output"
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.db = JobDatabase(self.project_dir / "state.sqlite")
        self.db.init_schema()

        self.tts = tts_engine
        self.mode = mode
        self.processor = AudioProcessor(sample_rate=self.tts.sample_rate)
        self.exporter = AudioExporter()

    def run_narrator_pipeline(
        self,
        book: Book,
        progress_callback: Callable[[int, int, float], None] | None = None,
    ) -> Job:
        """
        Run the complete narrator-only pipeline from book model to finished audiobook.

        Parameters
        ----------
        book:
            The parsed Book object.
        progress_callback:
            Optional callback receiving (done_count, total_count, eta_seconds).
        """
        job_id = f"job-{book.id[:8]}"
        job = self.db.get_job(job_id)
        if not job:
            job = Job(
                id=job_id,
                book_id=book.id,
                project_name=self.project_dir.name,
                mode=self.mode,
                current_stage="segmentation",
                stage_status=StageStatus.RUNNING,
            )
            self.db.save_job(job)

        # Stage 1: Segmentation
        segments = self._prepare_narrator_segments(book)
        self.db.register_segments(segments)

        # Stage 2: Synthesis (with per-segment resume support)
        job.current_stage = "synthesis"
        self.db.save_job(job)
        self._synthesize_segments(progress_callback=progress_callback)

        # Stage 3: Audio Assembly per chapter
        job.current_stage = "assembly"
        self.db.save_job(job)
        chapter_wavs = self._assemble_chapters(book)

        # Stage 4: Export MP3s and M4B
        job.current_stage = "export"
        self.db.save_job(job)
        self._export_audio(book, chapter_wavs)

        job.current_stage = "complete"
        job.stage_status = StageStatus.DONE
        self.db.save_job(job)
        return job

    def _prepare_narrator_segments(self, book: Book) -> list[Segment]:
        """
        Convert chapters & paragraphs of a book into sequential narrator segments.

        Respects the TTS engine's ``max_chars`` limit by splitting sentences if needed.
        Uses stable sub-ids (s00, s01, s02) for sentences, and (s00a, s00b) for split pieces.
        """
        segments: list[Segment] = []
        for ch in book.chapters:
            for para in ch.paragraphs:
                sentences = split_sentences(para.text)
                sub_idx = 0
                for sent in sentences:
                    # Break sentence down if it exceeds the engine limit
                    chunks = split_to_chunks(sent, max_chars=self.tts.max_chars)
                    if len(chunks) == 1:
                        # No split needed
                        seg_id = f"{para.id}-s{sub_idx:02d}"
                        segments.append(
                            Segment(
                                id=seg_id,
                                chapter=ch.index,
                                paragraph_id=para.id,
                                speaker="narrator",
                                kind=SegmentKind.NARRATION,
                                text=chunks[0],
                                status=SegmentStatus.PENDING,
                            )
                        )
                        sub_idx += 1
                    else:
                        # Sentence was split - use letter suffixes (s00a, s00b, s00c)
                        for chunk_idx, chunk in enumerate(chunks):
                            suffix = chr(ord("a") + chunk_idx)
                            seg_id = f"{para.id}-s{sub_idx:02d}{suffix}"
                            segments.append(
                                Segment(
                                    id=seg_id,
                                    chapter=ch.index,
                                    paragraph_id=para.id,
                                    speaker="narrator",
                                    kind=SegmentKind.NARRATION,
                                    text=chunk,
                                    status=SegmentStatus.PENDING,
                                )
                            )
                        sub_idx += 1
        return segments

    def _synthesize_segments(
        self,
        progress_callback: Callable[[int, int, float], None] | None = None,
    ) -> None:
        """Synthesise pending/failed segments, recording audio files and updating SQLite."""
        voice = VoiceRef(engine=self.tts.engine_name, voice_id="narrator")
        all_segments = self.db.get_all_segments()
        total_count = len(all_segments)
        completed = [s for s in all_segments if s.status == SegmentStatus.DONE]
        done_count = len(completed)

        pending = self.db.get_pending_or_failed_segments()
        if not pending:
            return

        durations: list[float] = []

        for seg in pending:
            t0 = time.time()
            self.db.update_segment_status(seg.id, SegmentStatus.RUNNING)

            out_wav = self.audio_dir / f"{seg.id}.wav"
            try:
                # Normalise text for synthesis (if not FakeTTS)
                if self.tts.engine_name != "fake":
                    from ebook2audiobook.tts.normalise import normalise_for_synthesis

                    text_to_synth = normalise_for_synthesis(seg.text)
                else:
                    text_to_synth = seg.text

                wav_bytes = self.tts.synthesize(text_to_synth, voice)
                out_wav.write_bytes(wav_bytes)
                self.db.update_segment_status(seg.id, SegmentStatus.DONE, audio_path=str(out_wav))
                done_count += 1
            except Exception as exc:
                logger.error("Failed synthesizing segment %s: %s", seg.id, exc)
                self.db.update_segment_status(seg.id, SegmentStatus.FAILED, increment_retry=True)

            elapsed = time.time() - t0
            durations.append(elapsed)

            avg_sec = sum(durations) / len(durations) if durations else 0.5
            remaining = total_count - done_count
            eta = remaining * avg_sec

            if progress_callback:
                progress_callback(done_count, total_count, eta)

    def _assemble_chapters(self, book: Book) -> list[tuple[str, Path, float]]:
        """
        Assemble and normalize chapter WAVs from segment audio files.

        Returns list of tuples: ``(chapter_title, wav_path, duration_seconds)``.
        """
        assembled_chapters: list[tuple[str, Path, float]] = []

        for ch in book.chapters:
            ch_segments = self.db.get_all_segments(chapter=ch.index)
            if not ch_segments:
                continue

            raw_ch_wav = self.audio_dir / f"chapter_{ch.index + 1:02d}_raw.wav"
            norm_ch_wav = self.audio_dir / f"chapter_{ch.index + 1:02d}.wav"

            self.processor.assemble_chapter_wav(
                ch_segments, raw_ch_wav, split_pause_ms=150
            )
            self.processor.normalize_loudness(raw_ch_wav, norm_ch_wav)

            # Calculate duration in seconds
            duration_s = 0.0
            if norm_ch_wav.exists():
                with wave.open(str(norm_ch_wav), "rb") as wf:
                    frames = wf.getnframes()
                    rate = wf.getframerate()
                    duration_s = frames / float(rate) if rate else 0.0

            title = ch.title or f"Chapter {ch.index + 1}"
            assembled_chapters.append((title, norm_ch_wav, duration_s))

        return assembled_chapters

    def _export_audio(self, book: Book, chapter_wavs: list[tuple[str, Path, float]]) -> None:
        """Convert chapter WAVs to MP3s and single M4B file with chapter markers."""
        # 1. Export chapter MP3s
        for i, (_, wav_path, _) in enumerate(chapter_wavs):
            out_mp3 = self.output_dir / f"chapter_{i + 1:02d}.mp3"
            self.exporter.wav_to_mp3(wav_path, out_mp3)

        # 2. Build M4B
        out_m4b = self.output_dir / f"{self.project_dir.name}.m4b"
        self.exporter.build_m4b(
            chapter_wavs=chapter_wavs,
            out_m4b=out_m4b,
            title=book.title or self.project_dir.name,
            author=book.author or "CastBook",
        )
