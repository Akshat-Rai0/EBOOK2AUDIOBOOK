"""
Pydantic models for the Cast — the mapping of characters to voices.

Concepts:
  VoiceRef         — a pointer to one voice in one TTS engine.
  CharacterProfile — inferred or assigned demographic profile (gender, age bracket).
  Character        — a named speaker with profile, aliases, and voice assignment.
  Cast             — the complete set of characters for one book, plus the narrator.

User edits are authoritative: a re-run NEVER overwrites a character with user_locked=True.
"""

from __future__ import annotations

from pathlib import Path
from typing import Self

from pydantic import BaseModel, Field, model_validator


class VoiceRef(BaseModel):
    """
    A pointer to a voice in a specific TTS engine.

    ``engine`` must match the adapter's self-reported name (e.g. ``"xtts"``,
    ``"vits"``, ``"fake"``).  ``voice_id`` is engine-specific (e.g. a speaker
    index for VITS, a speaker name for XTTS).
    """

    engine: str = Field(..., description="TTS engine name: xtts | vits | fake")
    voice_id: str = Field(..., description="Engine-specific voice identifier.")
    language: str = Field(default="en", description="BCP-47 language tag.")

    def __str__(self) -> str:
        return f"{self.engine}:{self.voice_id}"


class CharacterProfile(BaseModel):
    """
    Demographic profile of a character, used to recommend matching voices.

    Inferred only from textual evidence; confidence < 1.0; defaults to 'unknown'.
    """

    gender: str = Field(
        default="unknown",
        description="Gender: male | female | nonbinary | unknown",
    )
    gender_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence in gender inference (0.0 to 1.0).",
    )
    age_bracket: str = Field(
        default="unknown",
        description="Age bracket: child | young_adult | adult | elderly | unknown",
    )
    age_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence in age bracket inference (0.0 to 1.0).",
    )


class Character(BaseModel):
    """
    One speaker in the book.

    ``id`` is a stable, slugified identifier (e.g. "mira_vane").
    ``name`` is retained for backwards compatibility.
    ``display_name`` is the human-readable canonical name.
    ``user_locked`` indicates manual user curation; automated re-runs must not overwrite it.
    """

    id: str = Field(default="", description="Unique canonical character identifier.")
    name: str = Field(..., description="Canonical character name.")
    display_name: str = Field(default="", description="User-facing display name.")
    aliases: list[str] = Field(
        default_factory=list,
        description="Other names that refer to this character.",
    )
    profile: CharacterProfile = Field(
        default_factory=CharacterProfile,
        description="Inferred or user-specified demographic profile.",
    )
    line_count: int = Field(
        default=0,
        description="Number of segments attributed to this speaker.",
    )
    first_chapter: int = Field(
        default=0,
        description="0-based index of the first chapter where character speaks.",
    )
    user_locked: bool = Field(
        default=False,
        description="If True, automated casting/merge passes will not overwrite this character.",
    )
    voice: VoiceRef | None = Field(
        default=None,
        description="Assigned voice for the currently active engine.",
    )
    voice_assignments: dict[str, str] = Field(
        default_factory=dict,
        description="Engine-keyed voice assignments, e.g. {'xtts': 'Daisy', 'vits': 'p225'}.",
    )

    @model_validator(mode="after")
    def _populate_defaults(self) -> Self:
        """Ensure id and display_name are populated from name if empty."""
        if not self.id:
            # Clean slug for identifier
            self.id = self.name.lower().replace(" ", "_").replace(".", "")
        if not self.display_name:
            self.display_name = self.name
        return self

    def matches(self, name: str) -> bool:
        """Return True if ``name`` is the canonical name, id, or any alias (case-insensitive)."""
        lower = name.lower()
        if lower in (self.name.lower(), self.id.lower(), self.display_name.lower()):
            return True
        return lower in {a.lower() for a in self.aliases}


class Cast(BaseModel):
    """
    The complete voice cast for one book.

    Written to ``projects/<name>/cast.json`` by the casting agent and read by
    the TTS agent.  The narrator is always present; characters are sorted by
    ``line_count`` descending on the review screen.
    """

    narrator: Character = Field(
        default_factory=lambda: Character(
            name="narrator", display_name="Narrator", user_locked=True
        ),
        description="The narrator character. Always present.",
    )
    characters: list[Character] = Field(
        default_factory=list,
        description="Named characters, excluding the narrator.",
    )
    schema_version: str = Field(default="1.1")

    def get_character(self, name: str) -> Character | None:
        """Look up a character by canonical name, id, or alias. Returns None if not found."""
        if name.lower() in ("narrator", self.narrator.id.lower()):
            return self.narrator
        for char in self.characters:
            if char.matches(name):
                return char
        return None

    def all_speakers(self) -> list[Character]:
        """Return narrator + all characters, narrator first."""
        return [self.narrator, *self.characters]

    @classmethod
    def schema_path(cls) -> Path:
        return Path("docs/schemas/cast.schema.json")
