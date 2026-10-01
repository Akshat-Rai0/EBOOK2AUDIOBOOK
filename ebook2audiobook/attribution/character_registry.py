"""
Character registry — canonical name management and alias merging (M3).

**The problem it solves:**
A single character may be referred to by many names in a novel:
    "Mira", "Ms. Vane", "Mira Vane", "she" (when unambiguous), etc.
The registry merges these into one canonical entry so the voice-casting
step always assigns the same voice to the same person.

**Alias merging strategy (everyday analogy):**
Think of it like a contacts app.  If you have "John Smith", "John", and
"J. Smith" in your contacts, you merge them into one card — even though
the three strings are different.  We do the same by checking whether one
name is a sub-sequence of another or they share a common significant token.

**Unknown speaker policy:**
If the LLM cannot determine who is speaking, it returns ``"unknown"``.
The registry tracks unknowns, labels them ``"unknown"``, and will assign
them the narrator voice.  They are listed in the review output so the user
can manually assign a name later.

Design note: We deliberately keep this module dependency-free (no LLM,
no network) so it can be unit-tested with plain string inputs.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Tokens we ignore when comparing names (too generic to be identifying).
_STOP_WORDS = {"the", "a", "an", "mr", "mrs", "ms", "dr", "sir", "lady", "lord"}


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


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
        """
        Add *name* as an alias and update canonical_name if *name* is longer.

        *entries* is the parent registry's ``_entries`` dict; pass it so this
        method can re-index the key when the canonical name changes.
        """
        self.aliases.add(name)
        if len(name) > len(self.canonical_name) and name.lower() not in _STOP_WORDS:
            old_canonical = self.canonical_name
            self.canonical_name = name
            # Re-index the parent dict if provided.
            if entries is not None and old_canonical in entries:
                entries[name] = entries.pop(old_canonical)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class CharacterRegistry:
    """
    Tracks canonical characters encountered during attribution.

    Thread safety: not thread-safe.  The pipeline is single-threaded per job.

    Usage::

        registry = CharacterRegistry()
        canon = registry.resolve("Mira Vane")   # → "Mira Vane" (new entry)
        canon = registry.resolve("Mira")         # → "Mira Vane" (merged)
        registry.record_mention("Mira Vane")
        summary = registry.summary()
    """

    def __init__(self) -> None:
        # Maps canonical_name → CharacterEntry.  The narrator has its own
        # permanent entry so it is always tracked separately.
        self._entries: dict[str, CharacterEntry] = {
            "narrator": CharacterEntry(canonical_name="narrator"),
            "unknown": CharacterEntry(canonical_name="unknown", is_unknown=True),
        }
        # Secondary index: every alias (lower-cased) → canonical_name.
        self._alias_index: dict[str, str] = {
            "narrator": "narrator",
            "unknown": "unknown",
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(self, name: str) -> str:
        """
        Return the canonical name for *name*, creating or merging entries as needed.

        Parameters
        ----------
        name:
            Raw speaker string from the LLM (e.g. ``"Ms. Vane"``).

        Returns
        -------
        str
            Canonical name (e.g. ``"Mira Vane"``).
        """
        name = name.strip()
        if not name:
            return "unknown"

        lower = name.lower()

        # 1. Direct match in alias index.
        if lower in self._alias_index:
            return self._alias_index[lower]

        # 2. Fuzzy match — see if *name* is an alias of an existing character.
        candidate = self._fuzzy_match(lower)
        if candidate is not None:
            # Merge: add this name as another alias.
            entry = self._entries[candidate]
            entry.add_alias(name, entries=self._entries)
            new_canon = entry.canonical_name
            # Re-index ALL alias_index entries that pointed to the old canonical name
            # (including the earlier short alias that prompted this fuzzy match).
            for alias_lower, canon in list(self._alias_index.items()):
                if canon == candidate:
                    self._alias_index[alias_lower] = new_canon
            self._alias_index[lower] = new_canon
            logger.debug("Registry: %r merged into %r", name, new_canon)
            return new_canon

        # 3. No match — create a fresh entry.
        entry = CharacterEntry(canonical_name=name, aliases={name})
        self._entries[name] = entry
        self._alias_index[lower] = name
        logger.debug("Registry: new character %r", name)
        return name

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

    def summary(self) -> list[dict]:
        """
        Return a list of dicts summarising each character entry.

        Suitable for serialising to ``attribution.json``.
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fuzzy_match(self, lower_name: str) -> str | None:
        """
        Return the canonical_name of an existing character if *lower_name* is
        a plausible alias, else None.

        Strategy:
        - Extract significant tokens (≥2 chars, not in _STOP_WORDS) from both.
        - If the incoming name shares ≥1 significant token with an existing
          character *and* the existing character has ≥1 significant token,
          treat them as the same person.

        This is intentionally conservative: a single shared significant token
        is sufficient because false-positives (merging two different characters)
        are more harmful than false-negatives (two separate entries for the same
        person).
        """
        incoming_tokens = _significant_tokens(lower_name)
        if not incoming_tokens:
            return None

        for canon, entry in self._entries.items():
            if entry.is_unknown or canon == "narrator":
                continue
            existing_tokens = _significant_tokens(entry.canonical_name.lower())
            if not existing_tokens:
                continue
            if incoming_tokens & existing_tokens:
                return canon
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _significant_tokens(name: str) -> set[str]:
    """
    Return the set of "meaningful" tokens from *name*.

    Strips punctuation, lower-cases, removes stop words and single-char tokens.
    """
    raw = re.sub(r"[^a-z0-9 ]", " ", name.lower())
    return {t for t in raw.split() if len(t) >= 2 and t not in _STOP_WORDS}  # noqa: C401


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def build_registry(speaker_lists: Sequence[list[str]]) -> CharacterRegistry:
    """
    Build a ``CharacterRegistry`` pre-seeded with all known speaker names.

    *speaker_lists* is a list-of-lists because the attribution step processes
    chapters one at a time; each chapter contributes its own speaker list.

    Parameters
    ----------
    speaker_lists:
        E.g. ``[["Mira", "narrator"], ["Mira Vane", "James"]]``.

    Returns
    -------
    CharacterRegistry
        Registry with all names resolved and merged.
    """
    registry = CharacterRegistry()
    for names in speaker_lists:
        for name in names:
            registry.resolve(name)
    return registry
