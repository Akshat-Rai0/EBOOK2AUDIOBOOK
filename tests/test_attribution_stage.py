"""
Tests for AttributionStage (M4 Step 3).

Verifies production attribution execution, resume logic, user edit protection,
stage metadata recording, and explicit model eviction.
"""

from __future__ import annotations

import json
from pathlib import Path

from ebook2audiobook.attribution.attributor import Attributor
from ebook2audiobook.attribution.stage import AttributionStage
from ebook2audiobook.models.book import Book, Chapter, Paragraph
from ebook2audiobook.models.job import ConversionMode, Job, StageStatus
from ebook2audiobook.models.segment import Segment, SegmentKind, SegmentSource
from ebook2audiobook.store.db import JobDatabase


class FakeAttributor(Attributor):
    """Deterministic attributor for testing AttributionStage."""

    def __init__(self, mapping: dict[str, str] | None = None) -> None:
        self.model = "fake-llama3.2:3b"
        self.mapping = mapping or {}
        self.closed = False
        self.call_count = 0

    def attribute(self, paragraph: str, cast_so_far: list[str], context: str = "") -> list[dict]:
        self.call_count += 1
        # Extract quoted substrings
        import re

        quotes = re.findall(r'"([^"]+)"', paragraph)
        spans = []
        for q in quotes:
            speaker = self.mapping.get(q, "Mira")
            spans.append({"text": f'"{q}"', "speaker": speaker, "kind": "dialogue"})
        return spans

    def close(self) -> None:
        self.closed = True


class TestAttributionStage:
    def _create_sample_book(self) -> Book:
        return Book(
            id="book-attr-test",
            title="Spike Book",
            author="Author",
            source_path="/fake/book.txt",
            source_format="txt",
            chapters=[
                Chapter(
                    id="c00",
                    index=0,
                    title="Chapter One",
                    paragraphs=[
                        Paragraph(
                            id="c00-p000",
                            text="The clock struck eight.",
                            original_text="The clock struck eight.",
                        ),
                        Paragraph(
                            id="c00-p001",
                            text='"I am here," said Mira.',
                            original_text='"I am here," said Mira.',
                        ),
                    ],
                )
            ],
        )

    def test_attribution_stage_end_to_end(self, tmp_path: Path):
        db = JobDatabase(tmp_path / "state.sqlite")
        db.init_schema()

        job = Job(
            id="job-attr-1",
            book_id="book-attr-test",
            project_name="test_project",
            mode=ConversionMode.MULTI_VOICE,
            current_stage="ingest",
            stage_status=StageStatus.DONE,
        )
        db.save_job(job)

        book = self._create_sample_book()
        attributor = FakeAttributor(mapping={"I am here,": "Mira"})
        stage = AttributionStage(db=db, attributor=attributor)

        segments = stage.run(book, job)

        # 1 paragraph narration (1 seg) + 1 paragraph dialogue with tag (2 segs) = 3 segs
        assert len(segments) == 3

        # Seg 0: narration
        assert segments[0].speaker_id == "narrator"
        assert segments[0].kind == SegmentKind.NARRATION

        # Seg 1: dialogue attributed to Mira
        assert segments[1].speaker_id == "Mira"
        assert segments[1].kind == SegmentKind.DIALOGUE
        assert segments[1].source == SegmentSource.LLM

        # Seg 2: narration tag
        assert segments[2].speaker_id == "narrator"
        assert segments[2].kind == SegmentKind.NARRATION

        # Check stage status and metadata
        saved_job = db.get_job(job.id)
        assert saved_job.current_stage == "attribution"
        assert saved_job.stage_status == StageStatus.DONE

        with db._connection() as conn:
            row = conn.execute("SELECT * FROM stages WHERE job_id = ?", (job.id,)).fetchone()
            assert row is not None
            meta = json.loads(row["metadata"])
            assert meta["model"] == "fake-llama3.2:3b"
            assert "commit_hash" in meta

        # Assert memory separation: attributor was closed/unloaded
        assert attributor.closed is True

    def test_resume_does_not_repeat_attributed_paragraphs(self, tmp_path: Path):
        db = JobDatabase(tmp_path / "state.sqlite")
        db.init_schema()

        job = Job(
            id="job-resume",
            book_id="b1",
            project_name="proj",
            mode=ConversionMode.MULTI_VOICE,
            current_stage="chunk",
            stage_status=StageStatus.DONE,
        )
        db.save_job(job)

        # Pre-populate database with already attributed segment
        db.register_segments(
            [
                Segment(
                    id="c00-p001-s00",
                    chapter=0,
                    paragraph_id="c00-p001",
                    speaker_id="Mira",
                    kind=SegmentKind.DIALOGUE,
                    text='"I am here,"',
                    source=SegmentSource.LLM,
                ),
                Segment(
                    id="c00-p001-s01",
                    chapter=0,
                    paragraph_id="c00-p001",
                    speaker_id="narrator",
                    kind=SegmentKind.NARRATION,
                    text="said Mira.",
                    source=SegmentSource.RULE,
                ),
            ]
        )

        book = self._create_sample_book()
        attributor = FakeAttributor()
        stage = AttributionStage(db=db, attributor=attributor)

        stage.run(book, job)

        # Paragraph c00-p001 already has dialogue attributed -> attributor was NOT called for it
        assert attributor.call_count == 0

    def test_user_locked_segments_are_never_overwritten_by_attribution(self, tmp_path: Path):
        db = JobDatabase(tmp_path / "state.sqlite")
        db.init_schema()

        job = Job(
            id="job-user-lock",
            book_id="b1",
            project_name="proj",
            mode=ConversionMode.MULTI_VOICE,
            current_stage="chunk",
            stage_status=StageStatus.DONE,
        )
        db.save_job(job)

        # User locked segment
        db.register_segments(
            [
                Segment(
                    id="c00-p001-s00",
                    chapter=0,
                    paragraph_id="c00-p001",
                    speaker_id="Tomas",
                    kind=SegmentKind.DIALOGUE,
                    text='"I am here,"',
                    source=SegmentSource.USER,
                )
            ]
        )

        book = self._create_sample_book()
        # Attributor would claim "Mira"
        attributor = FakeAttributor(mapping={"I am here,": "Mira"})
        stage = AttributionStage(db=db, attributor=attributor)

        stage.run(book, job)

        seg = db.get_segment("c00-p001-s00")
        assert seg.speaker_id == "Tomas"
        assert seg.source == SegmentSource.USER
