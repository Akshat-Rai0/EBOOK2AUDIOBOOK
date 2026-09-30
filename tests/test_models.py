"""Tests for pydantic models: Book, Chapter, Paragraph, Segment, Cast, Job."""

from __future__ import annotations

import json

import pytest

from ebook2audiobook.models import (
    Book,
    Cast,
    Chapter,
    Character,
    ConversionMode,
    Job,
    Paragraph,
    Segment,
    SegmentKind,
    SegmentStatus,
    StageStatus,
    VoiceRef,
)

# ---------------------------------------------------------------------------
# Book / Chapter / Paragraph
# ---------------------------------------------------------------------------


class TestParagraph:
    def test_basic_creation(self):
        p = Paragraph(id="c00-p000", text="Hello.", original_text="Hello.")
        assert p.id == "c00-p000"
        assert not p.is_dialogue

    def test_dialogue_flag(self):
        p = Paragraph(id="c00-p000", text='"Hi," she said.', original_text='"Hi," she said.')
        assert p.is_dialogue is False  # flag is set externally by the extractor


class TestChapter:
    def test_chapter_defaults(self):
        ch = Chapter(id="c00", index=0, title="Prologue")
        assert ch.paragraphs == []
        assert ch.confidence == 1.0

    def test_confidence_bounds(self):
        with pytest.raises(ValueError):
            Chapter(id="c00", index=0, title="X", confidence=1.5)


class TestBook:
    def _make_book(self) -> Book:
        paras = [
            Paragraph(id=f"c00-p{i:03d}", text=f"Sentence {i}.", original_text=f"Sentence {i}.")
            for i in range(3)
        ]
        ch = Chapter(id="c00", index=0, title="Chapter One", paragraphs=paras)
        return Book(
            id="test-uuid",
            source_path="/tmp/test.txt",
            source_format="txt",
            chapters=[ch],
        )

    def test_word_count(self):
        book = self._make_book()
        # "Sentence 0." "Sentence 1." "Sentence 2." → 2 words each = 6
        assert book.word_count == 6

    def test_chapter_count(self):
        book = self._make_book()
        assert book.chapter_count == 1

    def test_get_chapter(self):
        book = self._make_book()
        assert book.get_chapter(0) is not None
        assert book.get_chapter(99) is None

    def test_json_roundtrip(self):
        book = self._make_book()
        data = json.loads(book.model_dump_json())
        restored = Book.model_validate(data)
        assert restored.id == book.id
        assert restored.chapter_count == 1

    def test_schema_version(self):
        book = self._make_book()
        assert book.schema_version == "1.0"


# ---------------------------------------------------------------------------
# Segment
# ---------------------------------------------------------------------------


class TestSegment:
    def test_default_status(self):
        s = Segment(
            id="c00-p000-s0",
            chapter=0,
            paragraph_id="c00-p000",
            speaker="narrator",
            kind=SegmentKind.NARRATION,
            text="Hello.",
        )
        assert s.status == SegmentStatus.PENDING
        assert s.audio_path is None

    def test_kind_values(self):
        assert SegmentKind.DIALOGUE == "dialogue"
        assert SegmentKind.NARRATION == "narration"
        assert SegmentKind.THOUGHT == "thought"


# ---------------------------------------------------------------------------
# Cast / Character / VoiceRef
# ---------------------------------------------------------------------------


class TestVoiceRef:
    def test_str_representation(self):
        v = VoiceRef(engine="fake", voice_id="fake-narrator")
        assert str(v) == "fake:fake-narrator"


class TestCharacter:
    def test_matches_canonical(self):
        c = Character(name="Mira", aliases=["Ms. Vane", "Mira Vane"])
        assert c.matches("Mira")
        assert c.matches("mira")  # case-insensitive
        assert c.matches("Ms. Vane")
        assert not c.matches("Tomas")

    def test_narrator_special(self):
        c = Character(name="narrator")
        assert c.matches("narrator")
        assert c.matches("NARRATOR")


class TestCast:
    def _make_cast(self) -> Cast:
        return Cast(
            characters=[
                Character(name="Mira", aliases=["Ms. Vane"], line_count=10),
                Character(name="Tomas", line_count=8),
            ]
        )

    def test_get_narrator(self):
        cast = self._make_cast()
        char = cast.get_character("narrator")
        assert char is not None
        assert char.name == "narrator"

    def test_get_by_alias(self):
        cast = self._make_cast()
        char = cast.get_character("Ms. Vane")
        assert char is not None
        assert char.name == "Mira"

    def test_get_missing(self):
        cast = self._make_cast()
        assert cast.get_character("unknown_person") is None

    def test_all_speakers_order(self):
        cast = self._make_cast()
        speakers = cast.all_speakers()
        assert speakers[0].name == "narrator"
        assert len(speakers) == 3

    def test_json_roundtrip(self):
        cast = self._make_cast()
        data = json.loads(cast.model_dump_json())
        restored = Cast.model_validate(data)
        assert len(restored.characters) == 2


# ---------------------------------------------------------------------------
# Job
# ---------------------------------------------------------------------------


class TestJob:
    def test_defaults(self):
        from datetime import datetime

        job = Job(
            id="job-001",
            book_id="book-001",
            project_name="dracula",
        )
        assert job.stage_status == StageStatus.PENDING
        assert job.mode == ConversionMode.MULTI_VOICE
        assert job.error is None
        assert isinstance(job.created_at, datetime)
