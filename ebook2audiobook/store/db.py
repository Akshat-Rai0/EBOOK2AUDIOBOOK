"""
SQLite database storage for jobs, stages, and per-segment synthesis state.

File location: ``projects/<project_name>/state.sqlite``

**Tables:**
- ``job``: High-level job record (status, mode, stage, error, timestamps).
- ``segments``: Per-segment execution record (chapter, speaker, status, retries, audio path).
- ``stages``: Detailed stage execution timing and status history.

**Why SQLite?**
SQLite is serverless, zero-dependency (part of the Python standard library),
ACID-compliant, and self-contained in a single file inside the project folder.
If the application crashes, SQLite rolls back uncommitted changes, preserving
data integrity so the pipeline can safely resume.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from ebook2audiobook.models.job import ConversionMode, Job, StageStatus
from ebook2audiobook.models.segment import Segment, SegmentKind, SegmentSource, SegmentStatus


class JobDatabase:
    """
    Manages state.sqlite for a specific project.

    Usage::

        db = JobDatabase(Path("projects/mybook/state.sqlite"))
        db.init_schema()
        db.save_job(job)
        db.register_segments(segments)
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager providing a SQLite connection with WAL mode and foreign keys."""
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_schema(self) -> None:
        """Create tables and indexes if they do not already exist."""
        with self._connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS job (
                    id TEXT PRIMARY KEY,
                    book_id TEXT NOT NULL,
                    project_name TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    current_stage TEXT NOT NULL,
                    stage_status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    error TEXT
                );

                CREATE TABLE IF NOT EXISTS segments (
                    id TEXT PRIMARY KEY,
                    chapter INTEGER NOT NULL,
                    paragraph_id TEXT NOT NULL,
                    speaker TEXT NOT NULL,
                    speaker_id TEXT NOT NULL DEFAULT 'narrator',
                    kind TEXT NOT NULL,
                    text TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 1.0,
                    source TEXT NOT NULL DEFAULT 'rule',
                    evidence TEXT,
                    voice_hash TEXT,
                    status TEXT NOT NULL,
                    audio_path TEXT,
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_segments_status ON segments(status);
                CREATE INDEX IF NOT EXISTS idx_segments_chapter ON segments(chapter);

                CREATE TABLE IF NOT EXISTS stages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    stage_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    error TEXT,
                    metadata TEXT,
                    FOREIGN KEY(job_id) REFERENCES job(id)
                );
                """
            )
            # Automatic schema migration for existing databases
            cols = {row["name"] for row in conn.execute("PRAGMA table_info(segments);").fetchall()}
            if "speaker_id" not in cols:
                conn.execute(
                    "ALTER TABLE segments ADD COLUMN speaker_id TEXT NOT NULL DEFAULT 'narrator';"
                )
            if "confidence" not in cols:
                conn.execute(
                    "ALTER TABLE segments ADD COLUMN confidence REAL NOT NULL DEFAULT 1.0;"
                )
            if "source" not in cols:
                conn.execute("ALTER TABLE segments ADD COLUMN source TEXT NOT NULL DEFAULT 'rule';")
            if "evidence" not in cols:
                conn.execute("ALTER TABLE segments ADD COLUMN evidence TEXT;")
            if "voice_hash" not in cols:
                conn.execute("ALTER TABLE segments ADD COLUMN voice_hash TEXT;")
            if "continues_previous" not in cols:
                conn.execute(
                    "ALTER TABLE segments ADD COLUMN "
                    "continues_previous INTEGER DEFAULT 0;"
                )

            conn.execute("CREATE INDEX IF NOT EXISTS idx_segments_speaker ON segments(speaker_id);")

            stage_cols = {
                row["name"] for row in conn.execute("PRAGMA table_info(stages);").fetchall()
            }
            if "metadata" not in stage_cols:
                conn.execute("ALTER TABLE stages ADD COLUMN metadata TEXT;")

    # ------------------------------------------------------------------
    # Job CRUD
    # ------------------------------------------------------------------

    def save_job(self, job: Job) -> None:
        """Insert or update a Job record."""
        now = datetime.now(UTC).isoformat()
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO job (
                    id, book_id, project_name, mode, current_stage, stage_status,
                    created_at, updated_at, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    current_stage = excluded.current_stage,
                    stage_status = excluded.stage_status,
                    updated_at = excluded.updated_at,
                    error = excluded.error;
                """,
                (
                    job.id,
                    job.book_id,
                    job.project_name,
                    job.mode.value,
                    job.current_stage,
                    job.stage_status.value,
                    job.created_at.isoformat(),
                    now,
                    job.error,
                ),
            )

    def get_job(self, job_id: str) -> Job | None:
        """Fetch a Job by ID, or None if not found."""
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM job WHERE id = ?", (job_id,)).fetchone()
            if not row:
                return None
            return Job(
                id=row["id"],
                book_id=row["book_id"],
                project_name=row["project_name"],
                mode=ConversionMode(row["mode"]),
                current_stage=row["current_stage"],
                stage_status=StageStatus(row["stage_status"]),
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
                error=row["error"],
            )

    # ------------------------------------------------------------------
    # Segments CRUD
    # ------------------------------------------------------------------

    def register_segments(self, segments: list[Segment]) -> None:
        """
        Register initial segments in bulk.

        User-locked segments (source='user') are NEVER overwritten.
        """
        now = datetime.now(UTC).isoformat()
        records = [
            (
                s.id,
                s.chapter,
                s.paragraph_id,
                s.speaker,
                s.speaker_id,
                s.kind.value,
                s.text,
                s.confidence,
                s.source.value,
                s.evidence,
                s.voice_hash,
                s.continues_previous,
                s.status.value,
                s.audio_path,
                s.retry_count,
                now,
                now,
            )
            for s in segments
        ]
        with self._connection() as conn:
            conn.executemany(
                """
                INSERT INTO segments (
                    id, chapter, paragraph_id, speaker, speaker_id, kind, text,
                    confidence, source, evidence, voice_hash, continues_previous,
                    status, audio_path, retry_count, created_at, updated_at
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                ON CONFLICT(id) DO UPDATE SET
                    chapter = excluded.chapter,
                    paragraph_id = excluded.paragraph_id,
                    speaker = excluded.speaker,
                    speaker_id = excluded.speaker_id,
                    kind = excluded.kind,
                    text = excluded.text,
                    confidence = excluded.confidence,
                    source = excluded.source,
                    evidence = excluded.evidence,
                    voice_hash = excluded.voice_hash,
                    continues_previous = excluded.continues_previous,
                    updated_at = excluded.updated_at
                WHERE segments.source != 'user';
                """,
                records,
            )

    def get_segment(self, segment_id: str) -> Segment | None:
        """Fetch a single segment by ID."""
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM segments WHERE id = ?", (segment_id,)).fetchone()
            if not row:
                return None
            return self._row_to_segment(row)

    def get_all_segments(self, chapter: int | None = None) -> list[Segment]:
        """Fetch all segments ordered by id, optionally filtered by chapter."""
        query = "SELECT * FROM segments"
        params: tuple = ()
        if chapter is not None:
            query += " WHERE chapter = ?"
            params = (chapter,)
        query += " ORDER BY id ASC;"

        with self._connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_segment(r) for r in rows]

    def get_pending_or_failed_segments(self) -> list[Segment]:
        """Return all segments needing synthesis (status is pending or failed)."""
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM segments
                WHERE status IN ('pending', 'failed')
                ORDER BY id ASC;
                """
            ).fetchall()
            return [self._row_to_segment(r) for r in rows]

    def update_segment_status(
        self,
        segment_id: str,
        status: SegmentStatus,
        audio_path: str | None = None,
        increment_retry: bool = False,
    ) -> None:
        """Update a segment's progress status, optional audio path, and retry counter."""
        now = datetime.now(UTC).isoformat()
        with self._connection() as conn:
            retry_sql = "retry_count = retry_count + 1," if increment_retry else ""
            if audio_path is not None:
                conn.execute(
                    f"""
                    UPDATE segments
                    SET status = ?, audio_path = ?, {retry_sql} updated_at = ?
                    WHERE id = ?;
                    """,
                    (status.value, audio_path, now, segment_id),
                )
            else:
                conn.execute(
                    f"""
                    UPDATE segments
                    SET status = ?, {retry_sql} updated_at = ?
                    WHERE id = ?;
                    """,
                    (status.value, now, segment_id),
                )

    def update_segment_speaker(
        self,
        segment_id: str,
        speaker_id: str,
        source: SegmentSource = SegmentSource.USER,
        confidence: float = 1.0,
    ) -> None:
        """Update a segment's speaker assignment. Defaults to source=user (authoritative)."""
        now = datetime.now(UTC).isoformat()
        with self._connection() as conn:
            conn.execute(
                """
                UPDATE segments
                SET speaker_id = ?, speaker = ?, source = ?, confidence = ?, updated_at = ?
                WHERE id = ?;
                """,
                (speaker_id, speaker_id, source.value, confidence, now, segment_id),
            )

    def get_stats(self) -> dict[str, int]:
        """Return counts of segments grouped by status."""
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT status, COUNT(*) as count
                FROM segments
                GROUP BY status;
                """
            ).fetchall()
            stats = {status.value: 0 for status in SegmentStatus}
            for row in rows:
                stats[row["status"]] = row["count"]
            stats["total"] = sum(stats.values())
            return stats

    @staticmethod
    def _row_to_segment(row: sqlite3.Row) -> Segment:
        row_keys = row.keys()
        speaker_id = row["speaker_id"] if "speaker_id" in row_keys else row["speaker"]
        confidence = (
            float(row["confidence"])
            if "confidence" in row_keys and row["confidence"] is not None
            else 1.0
        )
        source_val = row["source"] if "source" in row_keys and row["source"] is not None else "rule"
        evidence = row["evidence"] if "evidence" in row_keys else None
        voice_hash = row["voice_hash"] if "voice_hash" in row_keys else None
        continues_previous = (
            bool(row["continues_previous"])
            if "continues_previous" in row_keys and row["continues_previous"] is not None
            else False
        )

        return Segment(
            id=row["id"],
            chapter=row["chapter"],
            paragraph_id=row["paragraph_id"],
            speaker=row["speaker"],
            speaker_id=speaker_id,
            kind=SegmentKind(row["kind"]),
            confidence=confidence,
            source=SegmentSource(source_val),
            evidence=evidence,
            voice_hash=voice_hash,
            continues_previous=continues_previous,
            text=row["text"],
            status=SegmentStatus(row["status"]),
            audio_path=row["audio_path"],
            retry_count=row["retry_count"],
        )
