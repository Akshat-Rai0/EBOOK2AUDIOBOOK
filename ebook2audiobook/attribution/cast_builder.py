"""Cast seed builder using NER for character discovery.

Analogy: a casting director reads the whole script and writes the cast list
before anyone auditions, so we know who we need voices for before attribution.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ebook2audiobook.models.book import Book

logger = logging.getLogger(__name__)


@dataclass
class CharacterSeed:
    """A candidate character from NER or manual seed file."""

    name: str
    aliases: list[str]
    mention_count: int


class CastSeedFile(BaseModel):
    """Schema for projects/<p>/cast_seed.json hand-edited file."""

    characters: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of character entries with name and optional aliases.",
    )


def build_cast_seed(
    book: Book,
    min_mentions: int = 3,
    spacy_model: str = "en_core_web_sm",
    seed_file_path: Path | None = None,
    ner_func: Callable[[str], list[tuple[str, int]]] | None = None,
) -> list[CharacterSeed]:
    """
    Build a cast seed list using NER or a fake NER function for testing.

    Parameters
    ----------
    book:
        The Book object with chapters and paragraphs.
    min_mentions:
        Minimum mention count to include a character.
    spacy_model:
        spaCy model name for NER (default: en_core_web_sm).
    seed_file_path:
        Optional path to hand-edited cast_seed.json file.
    ner_func:
        Optional fake NER function for testing (takes text, returns list of (name, count)).

    Returns
    -------
    list[CharacterSeed]
        Sorted by mention count descending.
    """
    # Load hand-edited seed file if present
    manual_seeds: dict[str, list[str]] = {}
    if seed_file_path and seed_file_path.exists():
        try:
            seed_data = CastSeedFile.model_validate_json(seed_file_path.read_text())
            for entry in seed_data.characters:
                name = entry.get("name", "")
                aliases = entry.get("aliases", [])
                if name:
                    manual_seeds[name] = aliases
            logger.info(f"Loaded {len(manual_seeds)} manual seeds from {seed_file_path}")
        except Exception as exc:
            logger.warning(f"Failed to load seed file {seed_file_path}: {exc}")

    # Use fake NER if provided (for testing)
    if ner_func is not None:
        return _build_with_fake_ner(book, min_mentions, manual_seeds, ner_func)

    # Use real spaCy NER
    try:
        import spacy

        nlp = spacy.load(spacy_model)
    except ImportError:
        logger.error(
            f"spaCy model '{spacy_model}' not found. "
            f"Run: uv run python -m spacy download {spacy_model}"
        )
        return []
    except OSError:
        logger.error(
            f"spaCy model '{spacy_model}' not downloaded. "
            f"Run: uv run python -m spacy download {spacy_model}"
        )
        return []

    mention_counts: dict[str, int] = {}
    entity_to_canonical: dict[str, str] = {}

    for chapter in book.chapters:
        chapter_text = "\n".join(p.text for p in chapter.paragraphs)
        doc = nlp(chapter_text)

        for ent in doc.ents:
            if ent.label_ == "PERSON":
                name = ent.text.strip()
                if not name:
                    continue

                # Use existing canonical if already seen
                canonical = entity_to_canonical.get(name, name)
                mention_counts[canonical] = mention_counts.get(canonical, 0) + 1

                # Track aliases (same name variations)
                if name != canonical:
                    entity_to_canonical[name] = canonical

    # Filter by min_mentions
    filtered = {name: count for name, count in mention_counts.items() if count >= min_mentions}

    # Merge with manual seeds (manual seeds win)
    for name, aliases in manual_seeds.items():
        if name not in filtered:
            filtered[name] = 0  # Manual seeds included even with 0 mentions
        # Add aliases from manual seed
        for alias in aliases:
            entity_to_canonical[alias] = name

    # Build CharacterSeed list
    seeds: list[CharacterSeed] = []
    for name, count in filtered.items():
        aliases = [k for k, v in entity_to_canonical.items() if v == name and k != name]
        seeds.append(CharacterSeed(name=name, aliases=aliases, mention_count=count))

    # Sort by mention count descending
    seeds.sort(key=lambda s: s.mention_count, reverse=True)

    logger.info(f"Built cast seed: {len(seeds)} characters with >={min_mentions} mentions")
    return seeds


def _build_with_fake_ner(
    book: Book,
    min_mentions: int,
    manual_seeds: dict[str, list[str]],
    ner_func: Callable[[str], list[tuple[str, int]]],
) -> list[CharacterSeed]:
    """Build cast seed using a fake NER function for testing."""
    mention_counts: dict[str, int] = {}

    for chapter in book.chapters:
        chapter_text = "\n".join(p.text for p in chapter.paragraphs)
        for name, count in ner_func(chapter_text):
            mention_counts[name] = mention_counts.get(name, 0) + count

    # Filter by min_mentions
    filtered = {name: count for name, count in mention_counts.items() if count >= min_mentions}

    # Merge with manual seeds
    for name, aliases in manual_seeds.items():
        if name not in filtered:
            filtered[name] = 0
        for alias in aliases:
            mention_counts[alias] = name

    # Build CharacterSeed list
    seeds: list[CharacterSeed] = []
    for name, count in filtered.items():
        aliases = [k for k, v in mention_counts.items() if v == name and k != name]
        seeds.append(CharacterSeed(name=name, aliases=aliases, mention_count=count))

    seeds.sort(key=lambda s: s.mention_count, reverse=True)
    return seeds
