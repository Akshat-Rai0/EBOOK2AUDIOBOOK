"""
Library functions for managing the cast, reviews, merges, splits, and approvals.

All CLI commands in `castbook cast` delegate to these pure library functions.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ebook2audiobook.casting.caster import Caster
from ebook2audiobook.models.book import Book
from ebook2audiobook.models.cast import Cast, Character, VoiceRef
from ebook2audiobook.store.db import JobDatabase

logger = logging.getLogger(__name__)


def compute_cast_hash(cast: Cast) -> str:
    """Return SHA-256 of cast.json serialized content."""
    payload = cast.model_dump_json(indent=2)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_cast(project_dir: Path) -> Cast | None:
    """Load cast.json from project directory if it exists."""
    cast_path = project_dir / "cast.json"
    if not cast_path.exists():
        return None
    try:
        data = json.loads(cast_path.read_text(encoding="utf-8"))
        return Cast.model_validate(data)
    except Exception as exc:
        logger.warning("Failed to load cast.json: %s", exc)
        return None


def save_cast(project_dir: Path, cast: Cast) -> None:
    """Save cast to cast.json."""
    cast_path = project_dir / "cast.json"
    cast_path.write_text(cast.model_dump_json(indent=2), encoding="utf-8")


def suggest_cast(project_dir: Path, engine: str = "fake") -> Cast:
    """
    Run automated casting algorithm over book and state.sqlite segments.
    Writes projects/<p>/cast.json and returns the updated Cast.
    """
    project_dir = Path(project_dir)
    cast = load_cast(project_dir)
    if cast is None:
        # Build initial cast from characters in state.sqlite or book.json
        db_path = project_dir / "state.sqlite"
        characters: list[Character] = []
        if db_path.exists():
            db = JobDatabase(db_path)
            segments = db.get_all_segments()
            speaker_counts: dict[str, int] = {}
            first_chapters: dict[str, int] = {}
            for s in segments:
                if s.speaker_id and s.speaker_id not in ("narrator", "unknown"):
                    speaker_counts[s.speaker_id] = speaker_counts.get(s.speaker_id, 0) + 1
                    if s.speaker_id not in first_chapters:
                        first_chapters[s.speaker_id] = s.chapter

            for spk_id, count in speaker_counts.items():
                characters.append(
                    Character(
                        id=spk_id.lower().replace(" ", "_"),
                        name=spk_id,
                        line_count=count,
                        first_chapter=first_chapters.get(spk_id, 0),
                    )
                )
        cast = Cast(characters=characters)

    # Profile from book.json if available
    book_json_path = project_dir / "book.json"
    caster = Caster()
    if book_json_path.exists():
        try:
            book = Book.model_validate_json(book_json_path.read_text(encoding="utf-8"))
            caster.profile_characters(cast, book)
        except Exception as exc:
            logger.debug("Profiling from book.json failed: %s", exc)

    # Load segments from sqlite if available
    db_path = project_dir / "state.sqlite"
    segments = None
    if db_path.exists():
        db = JobDatabase(db_path)
        segments = db.get_all_segments()

    cast = caster.cast_characters(cast, engine=engine, segments=segments)
    save_cast(project_dir, cast)
    return cast


def set_character_voice(
    project_dir: Path,
    character_name: str,
    voice_id: str,
    engine: str = "fake",
) -> Character:
    """
    Assign a specific voice to a character, marking user_locked=True.
    """
    project_dir = Path(project_dir)
    cast = load_cast(project_dir)
    if cast is None:
        raise ValueError("cast.json not found. Run 'castbook cast suggest' first.")

    char = cast.get_character(character_name)
    if char is None:
        raise ValueError(f"Character {character_name!r} not found in cast.")

    char.voice_assignments[engine] = voice_id
    char.voice = VoiceRef(engine=engine, voice_id=voice_id)
    char.user_locked = True
    save_cast(project_dir, cast)
    return char


def merge_characters(
    project_dir: Path,
    source_name: str,
    target_name: str,
) -> tuple[Character, dict[str, Any]]:
    """
    Merge source character into target character transactionally across cast.json
    and state.sqlite. Appends record to merge_history.json.

    Winner rule:
    - If both are user_locked: raise ValueError (abort).
    - If source is user_locked: source wins (target merged into source).
    - If target is user_locked or neither: target wins.
    """
    project_dir = Path(project_dir)
    cast = load_cast(project_dir)
    if cast is None:
        raise ValueError("cast.json not found. Run 'castbook cast suggest' first.")

    source_char = cast.get_character(source_name)
    target_char = cast.get_character(target_name)

    if source_char is None or target_char is None:
        raise ValueError(
            f"Both characters must exist (source: {source_name!r}, target: {target_name!r})."
        )

    if source_char.id == target_char.id:
        raise ValueError("Cannot merge a character into itself.")

    # Rule: If both are user_locked, abort!
    if source_char.user_locked and target_char.user_locked:
        raise ValueError(
            f"Both {source_char.name!r} and {target_char.name!r} are user_locked. "
            "Manual resolution required."
        )

    # Decide winner
    if source_char.user_locked and not target_char.user_locked:
        winner = source_char
        loser = target_char
    else:
        winner = target_char
        loser = source_char

    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "action": "merge",
        "winner_id": winner.id,
        "loser_id": loser.id,
        "loser_name": loser.name,
        "loser_aliases": list(loser.aliases),
        "loser_line_count": loser.line_count,
        "loser_voice_assignments": dict(loser.voice_assignments),
        "loser_user_locked": loser.user_locked,
    }

    # Update cast
    winner.line_count += loser.line_count
    winner.aliases.append(loser.name)
    winner.aliases.extend(loser.aliases)
    winner.aliases = sorted(set(winner.aliases))
    cast.characters = [c for c in cast.characters if c.id != loser.id]
    save_cast(project_dir, cast)

    # Update state.sqlite atomically
    db_path = project_dir / "state.sqlite"
    if db_path.exists():
        db = JobDatabase(db_path)
        with db._connection() as conn:
            conn.execute(
                """
                UPDATE segments
                SET speaker_id = ?, speaker = ?
                WHERE speaker_id = ? OR speaker = ?;
                """,
                (winner.name, winner.name, loser.name, loser.name),
            )

    # Append to merge_history.json
    _append_history(project_dir, record)
    return winner, record


def undo_last_operation(project_dir: Path) -> dict[str, Any] | None:
    """
    Revert the last merge or rename from merge_history.json.
    """
    project_dir = Path(project_dir)
    history = _load_history(project_dir)
    if not history:
        return None

    record = history.pop()
    action = record.get("action")

    if action == "merge":
        cast = load_cast(project_dir)
        if cast is not None:
            winner = cast.get_character(record["winner_id"])
            if winner:
                winner.line_count = max(0, winner.line_count - record["loser_line_count"])
                for a in [record["loser_name"], *record["loser_aliases"]]:
                    if a in winner.aliases:
                        winner.aliases.remove(a)

            # Recreate loser
            loser = Character(
                id=record["loser_id"],
                name=record["loser_name"],
                aliases=record["loser_aliases"],
                line_count=record["loser_line_count"],
                voice_assignments=record["loser_voice_assignments"],
                user_locked=record["loser_user_locked"],
            )
            cast.characters.append(loser)
            save_cast(project_dir, cast)

    _save_history(project_dir, history)
    return record


def rename_character(project_dir: Path, character_name: str, new_name: str) -> Character:
    """Rename a character's display name."""
    project_dir = Path(project_dir)
    cast = load_cast(project_dir)
    if cast is None:
        raise ValueError("cast.json not found.")

    char = cast.get_character(character_name)
    if char is None:
        raise ValueError(f"Character {character_name!r} not found.")

    old_name = char.display_name or char.name
    char.display_name = new_name
    save_cast(project_dir, cast)

    _append_history(
        project_dir,
        {
            "timestamp": datetime.now(UTC).isoformat(),
            "action": "rename",
            "character_id": char.id,
            "old_name": old_name,
            "new_name": new_name,
        },
    )
    return char


def approve_cast(project_dir: Path, auto: bool = False) -> str:
    """
    Approve cast: store SHA-256 of cast.json content in state.sqlite / sidecar.
    If auto=True, set unknown speakers to narrator first.
    """
    project_dir = Path(project_dir)
    cast = load_cast(project_dir)
    if cast is None:
        cast = suggest_cast(project_dir)

    if auto:
        # Route unknown speakers to narrator
        db_path = project_dir / "state.sqlite"
        if db_path.exists():
            db = JobDatabase(db_path)
            with db._connection() as conn:
                conn.execute(
                    "UPDATE segments SET speaker_id = 'narrator', speaker = 'narrator' "
                    "WHERE speaker_id = 'unknown' OR speaker = 'unknown';"
                )

    cast_hash = compute_cast_hash(cast)
    sidecar_path = project_dir / "cast_approval.json"
    approval_data = {
        "approved_at": datetime.now(UTC).isoformat(),
        "cast_hash": cast_hash,
        "auto": auto,
    }
    sidecar_path.write_text(json.dumps(approval_data, indent=2), encoding="utf-8")
    return cast_hash


def verify_cast_approval(project_dir: Path) -> bool:
    """Verify stored cast approval hash matches current cast.json content."""
    project_dir = Path(project_dir)
    sidecar_path = project_dir / "cast_approval.json"
    if not sidecar_path.exists():
        return False
    cast = load_cast(project_dir)
    if cast is None:
        return False

    try:
        data = json.loads(sidecar_path.read_text(encoding="utf-8"))
        stored_hash = data.get("cast_hash")
        return stored_hash == compute_cast_hash(cast)
    except Exception:
        return False


def _load_history(project_dir: Path) -> list[dict[str, Any]]:
    path = project_dir / "merge_history.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def _save_history(project_dir: Path, history: list[dict[str, Any]]) -> None:
    path = project_dir / "merge_history.json"
    path.write_text(json.dumps(history, indent=2), encoding="utf-8")


def _append_history(project_dir: Path, record: dict[str, Any]) -> None:
    history = _load_history(project_dir)
    history.append(record)
    _save_history(project_dir, history)
