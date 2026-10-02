"""
Character registry — canonical name management, alias merging, and audit history (M4).

**The problem it solves:**
A single character may be referred to by many names in a novel:
    "Mira", "Ms. Vane", "Mira Vane", "she" (when unambiguous), etc.
The registry merges these into one canonical entry so the voice-casting
step always assigns the same voice to the same person.

**Ambiguity Guard (M4):**
If an incoming name matches multiple characters (e.g. "Smith" when both
"John Smith" and "Jane Smith" exist) or is a generic title ("the Captain",
"Doctor"), it is flagged for manual review rather than erroneously merged.

**Merge History & Undo:**
All manual and automatic merges append to an audit trail so users can
review merges or undo an incorrect merge via ``castbook cast undo``.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Tokens we ignore when comparing names (too generic to be identifying).
_STOP_WORDS = {"the", "a", "an", "mr", "mrs", "ms", "dr", "sir", "lady", "lord"}

# Standalone titles that should not be merged automatically if bare
_AMBIGUOUS_TITLES = {
    "captain",
    "doctor",
    "dr",
    "professor",
    "inspector",
    "major",
    "colonel",
    "judge",
    "priest",
    "father",
    "officer",
    "detective",
    "the old man",
}


@dataclass
class MergeRecord:
    """Record of a merge operation for audit and undo."""

    timestamp: str
    source_name: str
    target_name: str
    source_aliases: list[str]
    source_mentions: int


@dataclass
class CharacterEntry:
    """
    One canonical character in the book.

    Attributes
    ----------
    canonical_name:
        The "best" form of the name — the longest, most specific alias seen.
    aliases:
        All name strings that have been merged into this entry.
    mention_count:
        Total number of attribution spans assigned to this character.
    is_unknown:
        True when the character represents the catch-all ``"unknown"`` bucket.
    """

    canonical_name: str
    aliases: set[str] = field(default_factory=set)
    mention_count: int = 0
    is_unknown: bool = False

    def add_alias(self, name: str, entries: dict[str, CharacterEntry] | None = None) -> None:
        """Add *name* as an alias and update canonical_name if *name* is longer."""
        self.aliases.add(name)
        if len(name) > len(self.canonical_name) and name.lower() not in _STOP_WORDS:
            old_canonical = self.canonical_name
            self.canonical_name = name
            # Re-index the parent dict if provided.
            if entries is not None and old_canonical in entries:
                entries[name] = entries.pop(old_canonical)


class CharacterRegistry:
    """
    Tracks canonical characters encountered during attribution.

    Provides alias merging, collision detection, and reversible merge history.
    """

    def __init__(self) -> None:
        self._entries: dict[str, CharacterEntry] = {
            "narrator": CharacterEntry(canonical_name="narrator"),
            "unknown": CharacterEntry(canonical_name="unknown", is_unknown=True),
        }
        self._alias_index: dict[str, str] = {
            "narrator": "narrator",
            "unknown": "unknown",
        }
        self._merge_history: list[MergeRecord] = []
        self._flagged_ambiguities: list[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(self, name: str) -> str:
        """
        Return the canonical name for *name*, creating or merging entries as needed.
        """
        name = name.strip()
        if not name:
            return "unknown"

        lower = name.lower()

        # 1. Direct match in alias index.
        if lower in self._alias_index:
            return self._alias_index[lower]

        # 2. Fuzzy match with ambiguity guard
        candidate = self._fuzzy_match(lower)
        if candidate is not None:
            entry = self._entries[candidate]
            entry.add_alias(name, entries=self._entries)
            new_canon = entry.canonical_name
            for alias_lower, canon in list(self._alias_index.items()):
                if canon == candidate:
                    self._alias_index[alias_lower] = new_canon
            self._alias_index[lower] = new_canon
            logger.debug("Registry: %r merged into %r", name, new_canon)
            return new_canon

        # 3. No match or ambiguous — create fresh entry.
        entry = CharacterEntry(canonical_name=name, aliases={name})
        self._entries[name] = entry
        self._alias_index[lower] = name
        logger.debug("Registry: new character %r", name)
        return name

    def manual_merge(self, source_name: str, target_name: str) -> bool:
        """
        Explicitly merge *source_name* into *target_name*.

        Records the merge in history so it can be reversed with ``undo()``.
        """
        source_canon = self.resolve(source_name)
        target_canon = self.resolve(target_name)

        if source_canon == target_canon:
            return False

        if source_canon not in self._entries or target_canon not in self._entries:
            return False

        source_entry = self._entries[source_canon]
        target_entry = self._entries[target_canon]

        # Record merge before applying
        record = MergeRecord(
            timestamp=datetime.utcnow().isoformat(),
            source_name=source_canon,
            target_name=target_canon,
            source_aliases=list(source_entry.aliases),
            source_mentions=source_entry.mention_count,
        )
        self._merge_history.append(record)

        # Merge aliases and mentions
        target_entry.aliases.update(source_entry.aliases)
        target_entry.aliases.add(source_canon)
        target_entry.mention_count += source_entry.mention_count

        # Re-index all aliases pointing to source_canon
        for alias_lower, canon in list(self._alias_index.items()):
            if canon == source_canon:
                self._alias_index[alias_lower] = target_canon

        # Remove source entry from entries map
        del self._entries[source_canon]
        logger.info("Registry: merged %r into %r", source_canon, target_canon)
        return True

    def undo(self) -> MergeRecord | None:
        """
        Revert the most recent merge operation, restoring the separated character.
        """
        if not self._merge_history:
            return None

        record = self._merge_history.pop()

        # Recreate source entry
        source_entry = CharacterEntry(
            canonical_name=record.source_name,
            aliases=set(record.source_aliases),
            mention_count=record.source_mentions,
        )
        self._entries[record.source_name] = source_entry

        # Restore target mention count
        if record.target_name in self._entries:
            self._entries[record.target_name].mention_count = max(
                0, self._entries[record.target_name].mention_count - record.source_mentions
            )
            # Remove restored aliases from target
            self._entries[record.target_name].aliases.difference_update(record.source_aliases)
            self._entries[record.target_name].aliases.discard(record.source_name)

        # Re-point restored aliases
        for alias in record.source_aliases:
            self._alias_index[alias.lower()] = record.source_name
        self._alias_index[record.source_name.lower()] = record.source_name

        logger.info(
            "Registry: reversed merge of %r into %r", record.source_name, record.target_name
        )
        return record

    def record_mention(self, canonical_name: str) -> None:
        """Increment the mention counter for *canonical_name*."""
        if canonical_name in self._entries:
            self._entries[canonical_name].mention_count += 1

    def canonical_names(self) -> list[str]:
        """Return all canonical names (including narrator and unknown)."""
        return list(self._entries.keys())

    def character_names(self) -> list[str]:
        """Return canonical names excluding narrator and unknown."""
        return [n for n, e in self._entries.items() if not e.is_unknown and n != "narrator"]

    def unknowns_count(self) -> int:
        """Number of mentions attributed to 'unknown'."""
        return self._entries["unknown"].mention_count

    def flagged_ambiguities(self) -> list[dict]:
        """Return ambiguous name occurrences flagged for human review."""
        return list(self._flagged_ambiguities)

    def summary(self) -> list[dict]:
        """
        Return a list of dicts summarising each character entry.
        """
        rows = []
        for entry in self._entries.values():
            rows.append(
                {
                    "canonical_name": entry.canonical_name,
                    "aliases": sorted(entry.aliases),
                    "mention_count": entry.mention_count,
                    "is_unknown": entry.is_unknown,
                }
            )
        return rows

    def save_history(self, path: Path) -> None:
        """Serialize merge history to JSON."""
        data = [
            {
                "timestamp": r.timestamp,
                "source_name": r.source_name,
                "target_name": r.target_name,
                "source_aliases": r.source_aliases,
                "source_mentions": r.source_mentions,
            }
            for r in self._merge_history
        ]
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load_history(self, path: Path) -> None:
        """Deserialize merge history from JSON."""
        if not path.exists():
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        self._merge_history = [MergeRecord(**d) for d in data]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fuzzy_match(self, lower_name: str) -> str | None:
        """
        Return the canonical_name of an existing character if *lower_name* is
        a plausible, unambiguous alias, else None.
        """
        # Ambiguity guard 1: standalone generic title
        if lower_name in _AMBIGUOUS_TITLES:
            self._flagged_ambiguities.append(
                {"name": lower_name, "reason": "ambiguous_generic_title"}
            )
            return None

        incoming_tokens = _significant_tokens(lower_name)
        if not incoming_tokens:
            return None

        # If incoming tokens consist solely of ambiguous titles
        if incoming_tokens.issubset(_AMBIGUOUS_TITLES):
            self._flagged_ambiguities.append(
                {"name": lower_name, "reason": "ambiguous_generic_title"}
            )
            return None

        matching_candidates: list[str] = []
        for canon, entry in self._entries.items():
            if entry.is_unknown or canon == "narrator":
                continue
            existing_tokens = _significant_tokens(entry.canonical_name.lower())
            if not existing_tokens:
                continue

            shared = incoming_tokens & existing_tokens
            if not shared:
                continue

            # If the only overlap is a generic title, do not auto-merge
            if shared.issubset(_AMBIGUOUS_TITLES):
                continue

            # Conflicting token guard: e.g. "Jane Smith" vs "John Smith"
            incoming_unique = incoming_tokens - shared - _AMBIGUOUS_TITLES
            existing_unique = existing_tokens - shared - _AMBIGUOUS_TITLES
            if incoming_unique and existing_unique:
                continue

            matching_candidates.append(canon)

        # Ambiguity guard 2: multiple candidates share the token (e.g. shared surname)
        if len(matching_candidates) > 1:
            self._flagged_ambiguities.append(
                {
                    "name": lower_name,
                    "candidates": matching_candidates,
                    "reason": "multiple_matching_characters",
                }
            )
            logger.warning(
                "Ambiguous match for %r: candidates=%s. Flagged for review.",
                lower_name,
                matching_candidates,
            )
            return None

        return matching_candidates[0] if matching_candidates else None


def _significant_tokens(name: str) -> set[str]:
    """Return the set of meaningful tokens from *name*."""
    raw = re.sub(r"[^a-z0-9 ]", " ", name.lower())
    return {t for t in raw.split() if len(t) >= 2 and t not in _STOP_WORDS}


def build_registry(speaker_lists: Sequence[list[str]]) -> CharacterRegistry:
    """Build a CharacterRegistry pre-seeded with all known speaker names."""
    registry = CharacterRegistry()
    for names in speaker_lists:
        for name in names:
            registry.resolve(name)
    return registry
