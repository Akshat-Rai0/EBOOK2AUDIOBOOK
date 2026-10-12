"""Tests for Caster."""

from __future__ import annotations

from ebook2audiobook.audio.catalogue import VoiceCatalogue
from ebook2audiobook.casting.caster import (
    EXTRAS_POOL_SIZE,
    Caster,
)
from ebook2audiobook.models.cast import Cast, Character
from ebook2audiobook.models.segment import Segment, SegmentKind, SegmentStatus


def _create_sample_segment(seg_id: str, speaker_id: str) -> Segment:
    return Segment(
        id=seg_id,
        chapter=0,
        paragraph_id="c00-p000",
        speaker_id=speaker_id,
        kind=SegmentKind.DIALOGUE,
        text="Sample dialogue text.",
        status=SegmentStatus.PENDING,
    )


class TestCaster:
    def test_frequency_ranking_gives_first_pick_to_top_speaker(self):
        cat = VoiceCatalogue()
        caster = Caster(catalogue=cat)

        cast = Cast(
            characters=[
                Character(name="Minor", line_count=2),
                Character(name="Major", line_count=50),
            ]
        )

        caster.cast_characters(cast, engine="fake")
        major = cast.get_character("Major")
        assert major is not None
        assert major.voice is not None
        assert major.voice.voice_id == "fake-char-1"

    def test_protagonist_i_policy_defaults_to_narrator(self):
        cat = VoiceCatalogue()
        caster = Caster(catalogue=cat)

        cast = Cast(
            characters=[
                Character(name="I", line_count=100),
                Character(name="Friend", line_count=20),
            ]
        )

        caster.cast_characters(cast, engine="fake")
        protagonist = cast.get_character("I")
        assert protagonist is not None
        assert protagonist.voice is not None
        assert protagonist.voice.voice_id == "fake-narrator"

    def test_cooccurrence_distinctness_in_sliding_window(self):
        cat = VoiceCatalogue()
        caster = Caster(catalogue=cat)

        # Alice and Bob talk repeatedly within a 20-segment window
        segs: list[Segment] = []
        for i in range(10):
            segs.append(_create_sample_segment(f"c00-p000-s{i*2:02d}", "Alice"))
            segs.append(_create_sample_segment(f"c00-p000-s{i*2+1:02d}", "Bob"))

        cast = Cast(
            characters=[
                Character(name="Alice", line_count=10),
                Character(name="Bob", line_count=10),
            ]
        )

        caster.cast_characters(cast, engine="fake", segments=segs)
        alice = cast.get_character("Alice")
        bob = cast.get_character("Bob")

        assert alice is not None and bob is not None
        assert alice.voice is not None and bob.voice is not None
        assert alice.voice.voice_id != bob.voice.voice_id

        # Verify acoustic distance is distinct
        v_alice = cat.get_voice("fake", alice.voice.voice_id)
        v_bob = cat.get_voice("fake", bob.voice.voice_id)
        assert v_alice is not None and v_bob is not None
        assert v_alice.acoustic_distance_to(v_bob) >= 1.0

    def test_minor_characters_not_demoted_when_voices_plentiful(self):
        cat = VoiceCatalogue()
        caster = Caster(catalogue=cat)

        # 3 characters total, fake has 6 character voices available -> enough for all
        cast = Cast(
            characters=[
                Character(name="Main", line_count=20),
                Character(name="Minor1", line_count=2),  # <= MINOR_CHARACTER_THRESHOLD
                Character(name="Minor2", line_count=1),  # <= MINOR_CHARACTER_THRESHOLD
            ]
        )

        caster.cast_characters(cast, engine="fake")
        # Every character gets a distinct voice
        assigned = {c.voice.voice_id for c in cast.characters if c.voice}
        assert len(assigned) == 3

    def test_minor_characters_demoted_to_extras_when_pool_exceeded(self):
        cat = VoiceCatalogue()
        caster = Caster(catalogue=cat)

        # Create 8 characters; fake catalogue only has 6 character voices available
        characters = [Character(name=f"Main{i}", line_count=20) for i in range(4)]
        characters += [Character(name=f"Minor{i}", line_count=2) for i in range(4)]

        cast = Cast(characters=characters)
        caster.cast_characters(cast, engine="fake")

        # Minor characters should draw from extras
        minor_voices = [c.voice.voice_id for c in cast.characters if c.line_count <= 2 and c.voice]
        assert len(set(minor_voices)) <= EXTRAS_POOL_SIZE

    def test_user_locked_is_never_overwritten(self):
        cat = VoiceCatalogue()
        caster = Caster(catalogue=cat)

        cast = Cast(
            characters=[
                Character(
                    name="Alice",
                    line_count=20,
                    user_locked=True,
                    voice_assignments={"fake": "fake-char-4"},
                ),
                Character(name="Bob", line_count=10),
            ]
        )

        caster.cast_characters(cast, engine="fake")
        alice = cast.get_character("Alice")
        assert alice is not None
        assert alice.voice is not None
        assert alice.voice.voice_id == "fake-char-4"  # Preserved

    def test_engine_switching_preserves_other_engines(self):
        cat = VoiceCatalogue()
        caster = Caster(catalogue=cat)

        cast = Cast(
            characters=[
                Character(
                    name="Alice",
                    line_count=20,
                    voice_assignments={"vits": "p230"},
                )
            ]
        )

        # Cast on fake engine
        caster.cast_characters(cast, engine="fake")
        alice = cast.get_character("Alice")
        assert alice is not None
        assert alice.voice_assignments.get("vits") == "p230"
        assert alice.voice_assignments.get("fake") is not None
