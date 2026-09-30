"""
Tests for M2 narrator-only audio pipeline: SQLite persistence, AudioProcessor,
PipelineOrchestrator, and resume logic.
"""

from __future__ import annotations

import wave
from pathlib import Path

import pytest

from ebook2audiobook.audio.processor import AudioProcessor
from ebook2audiobook.models.book import Book, Chapter, Paragraph
from ebook2audiobook.models.job import ConversionMode, Job, StageStatus
from ebook2audiobook.models.segment import Segment, SegmentKind, SegmentStatus
from ebook2audiobook.orchestrator.pipeline import PipelineOrchestrator
from ebook2audiobook.store.db import JobDatabase
from ebook2audiobook.tts.fake_tts import FakeTTS


@pytest.fixture()
def sample_book() -> Book:
    c1_paras = [
        Paragraph(
            id="c00-p000",
            text="Mira set the lamp down.",
            original_text="Mira set the lamp down.",
        ),
        Paragraph(
            id="c00-p001",
            text="She heard a soft sound outside the window.",
            original_text="She heard a soft sound outside the window.",
        ),
    ]
    c2_paras = [
        Paragraph(
            id="c01-p000",
            text="The next morning the sun was bright.",
            original_text="The next morning the sun was bright.",
        ),
    ]
    chapters = [
        Chapter(id="c00", index=0, title="Chapter One", paragraphs=c1_paras),
        Chapter(id="c01", index=1, title="Chapter Two", paragraphs=c2_paras),
    ]
    return Book(
        id="test-book-uuid-1234",
        title="Test Story",
        author="Author A",
        source_path="/fake/path.epub",
        source_format="epub",
        chapters=chapters,
    )


class TestJobDatabase:
    def test_job_persistence(self, tmp_path: Path):
        db_path = tmp_path / "state.sqlite"
        db = JobDatabase(db_path)
        db.init_schema()

        job = Job(
            id="job-1",
            book_id="book-1",
            project_name="proj-1",
            mode=ConversionMode.NARRATOR_ONLY,
            current_stage="synthesis",
            stage_status=StageStatus.RUNNING,
        )
        db.save_job(job)

        loaded = db.get_job("job-1")
        assert loaded is not None
        assert loaded.book_id == "book-1"
        assert loaded.current_stage == "synthesis"
        assert loaded.stage_status == StageStatus.RUNNING

    def test_segment_registration_and_update(self, tmp_path: Path):
        db = JobDatabase(tmp_path / "state.sqlite")
        db.init_schema()

        segments = [
            Segment(
                id="c00-p000-s00",
                chapter=0,
                paragraph_id="c00-p000",
                speaker="narrator",
                kind=SegmentKind.NARRATION,
                text="Hello world.",
                status=SegmentStatus.PENDING,
            ),
            Segment(
                id="c00-p001-s00",
                chapter=0,
                paragraph_id="c00-p001",
                speaker="narrator",
                kind=SegmentKind.NARRATION,
                text="Second segment.",
                status=SegmentStatus.PENDING,
            ),
        ]
        db.register_segments(segments)

        all_segs = db.get_all_segments()
        assert len(all_segs) == 2

        db.update_segment_status(
            "c00-p000-s00",
            SegmentStatus.DONE,
            audio_path="/fake/audio.wav",
        )
        updated = db.get_segment("c00-p000-s00")
        assert updated is not None
        assert updated.status == SegmentStatus.DONE
        assert updated.audio_path == "/fake/audio.wav"

        stats = db.get_stats()
        assert stats["done"] == 1
        assert stats["pending"] == 1


class TestAudioProcessor:
    def test_create_silence_wav(self, tmp_path: Path):
        proc = AudioProcessor(sample_rate=22050)
        silence_wav = proc.create_silence_wav(500, tmp_path / "silence.wav")
        assert silence_wav.exists()

        with wave.open(str(silence_wav), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration = frames / float(rate)
            assert 0.49 <= duration <= 0.51

    def test_concatenate_wavs(self, tmp_path: Path):
        proc = AudioProcessor(sample_rate=22050)
        s1 = proc.create_silence_wav(200, tmp_path / "s1.wav")
        s2 = proc.create_silence_wav(300, tmp_path / "s2.wav")
        out = proc.concatenate_wavs([s1, s2], tmp_path / "combined.wav")

        with wave.open(str(out), "rb") as wf:
            duration = wf.getnframes() / float(wf.getframerate())
            assert 0.49 <= duration <= 0.51

    def test_normalize_loudness(self, tmp_path: Path):
        proc = AudioProcessor(sample_rate=22050)
        tts = FakeTTS(mode="tone")
        from ebook2audiobook.models.cast import VoiceRef

        tone_wav = tmp_path / "tone.wav"
        tone_wav.write_bytes(
            tts.synthesize("Test sentence for audio", VoiceRef(engine="fake", voice_id="v1"))
        )

        norm_wav = tmp_path / "norm.wav"
        proc.normalize_loudness(tone_wav, norm_wav, target_dbfs=-18.0)
        assert norm_wav.exists()


class TestPipelineOrchestrator:
    def test_end_to_end_narrator_pipeline(self, tmp_path: Path, sample_book: Book):
        project_dir = tmp_path / "test_proj"
        tts = FakeTTS()
        orch = PipelineOrchestrator(project_dir=project_dir, tts_engine=tts)

        progress_calls = []

        def cb(done, total, eta):
            progress_calls.append((done, total))

        job = orch.run_narrator_pipeline(sample_book, progress_callback=cb)
        assert job.stage_status == StageStatus.DONE
        assert len(progress_calls) > 0

        # Verify output files
        mp3_1 = project_dir / "output" / "chapter_01.mp3"
        mp3_2 = project_dir / "output" / "chapter_02.mp3"
        m4b_file = project_dir / "output" / "test_proj.m4b"

        assert mp3_1.exists()
        assert mp3_2.exists()
        assert m4b_file.exists()

    def test_resume_does_not_repeat_done_segments(self, tmp_path: Path, sample_book: Book):
        project_dir = tmp_path / "resume_proj"
        tts = FakeTTS()
        orch = PipelineOrchestrator(project_dir=project_dir, tts_engine=tts)

        # 1. First run
        orch.run_narrator_pipeline(sample_book)
        db = JobDatabase(project_dir / "state.sqlite")
        initial_stats = db.get_stats()
        assert initial_stats["done"] > 0
        assert initial_stats["pending"] == 0

        # Track modification times of generated segment wavs (matching c*-p*-s*.wav)
        seg_files = list((project_dir / "audio").glob("c*-p*-s*.wav"))
        assert len(seg_files) > 0
        first_mtimes = {p: p.stat().st_mtime for p in seg_files}

        # 2. Second run (simulate resuming a completed or partially completed job)
        orch2 = PipelineOrchestrator(project_dir=project_dir, tts_engine=tts)
        orch2.run_narrator_pipeline(sample_book)

        second_mtimes = {p: p.stat().st_mtime for p in seg_files}
        # Already done segment wavs must NOT have been re-synthesized / touched
        for p in seg_files:
            assert first_mtimes[p] == second_mtimes[p]
