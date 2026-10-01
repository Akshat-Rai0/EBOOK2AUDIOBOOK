"""
Production Attribution Stage (M4 Step 3).

Applies the calibrated M3 LLM recipe to assign character speakers to dialogue segments:
- Model: llama3.2:3b (or configured local model).
- Constrained JSON output format with running cast list and paragraph context.
- Retry twice, then fall back to 'unknown'.
- User-edited segments (source=USER) are never overwritten.
- Stage metadata (model tag, prompt version, commit hash) recorded in state.sqlite.
- Explicit model eviction upon completion to guarantee memory separation before TTS.
"""

from __future__ import annotations

import json
import logging
import subprocess
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from ebook2audiobook.attribution.attributor import Attributor
from ebook2audiobook.attribution.character_registry import CharacterRegistry
from ebook2audiobook.chunker.dialogue_segmenter import DialogueSegmenter
from ebook2audiobook.models.book import Book
from ebook2audiobook.models.job import Job, StageStatus
from ebook2audiobook.models.segment import Segment, SegmentKind, SegmentSource
from ebook2audiobook.store.db import JobDatabase

logger = logging.getLogger(__name__)

PROMPT_VERSION = "1.0-m3-constrained"


def get_git_commit_hash(repo_dir: Path | None = None) -> str:
    """Return the current short git commit hash, or 'unknown'."""
    try:
        cmd = ["git", "rev-parse", "--short", "HEAD"]
        result = subprocess.run(
            cmd,
            cwd=str(repo_dir or Path.cwd()),
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


class AttributionStage:
    """
    Orchestrates the speaker attribution phase across all chapters of a book.
    """

    def __init__(
        self,
        db: JobDatabase,
        attributor: Attributor,
        segmenter: DialogueSegmenter | None = None,
        registry: CharacterRegistry | None = None,
    ) -> None:
        self.db = db
        self.attributor = attributor
        self.segmenter = segmenter or DialogueSegmenter()
        self.registry = registry or CharacterRegistry()

    def run(
        self,
        book: Book,
        job: Job,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[Segment]:
        """
        Execute attribution on *book*, recording state in *db*.

        Resumable: segments already attributed (or locked by user) are not re-queried.
        """
        job.current_stage = "attribution"
        job.stage_status = StageStatus.RUNNING
        self.db.save_job(job)

        started_at = datetime.utcnow().isoformat()
        commit_hash = get_git_commit_hash()
        model_name = getattr(self.attributor, "model", "custom")

        stage_meta = {
            "model": model_name,
            "prompt_version": PROMPT_VERSION,
            "commit_hash": commit_hash,
            "started_at": started_at,
        }

        # 1. Prepare initial segments if not already registered in DB
        existing_segs = self.db.get_all_segments()
        if not existing_segs:
            initial_segs = self.segmenter.segment_book(book)
            self.db.register_segments(initial_segs)
            all_segments = initial_segs
        else:
            all_segments = existing_segs

        # Index segments by paragraph_id for in-order attribution
        para_to_segs: dict[str, list[Segment]] = {}
        for seg in all_segments:
            para_to_segs.setdefault(seg.paragraph_id, []).append(seg)

        total_paragraphs = sum(len(ch.paragraphs) for ch in book.chapters)
        processed_paras = 0

        context_prev = ""

        # Pre-seed registry from existing segments (for resume safety)
        for seg in all_segments:
            if seg.speaker_id and seg.speaker_id not in ("narrator", "unknown"):
                self.registry.resolve(seg.speaker_id)

        try:
            for ch in book.chapters:
                for para in ch.paragraphs:
                    p_segs = para_to_segs.get(para.id, [])
                    # Check if paragraph has any dialogue segments needing attribution
                    needs_attribution = any(
                        s.kind == SegmentKind.DIALOGUE
                        and s.source != SegmentSource.USER
                        and s.speaker_id == "unknown"
                        for s in p_segs
                    )

                    if needs_attribution:
                        cast_so_far = self.registry.character_names()
                        spans = self.attributor.attribute(
                            paragraph=para.text,
                            cast_so_far=cast_so_far,
                            context=context_prev,
                        )

                        # Filter dialogue spans from model output
                        diag_spans = [s for s in spans if s.get("kind") == "dialogue"]
                        diag_idx = 0

                        for seg in p_segs:
                            if (
                                seg.kind == SegmentKind.DIALOGUE
                                and seg.source != SegmentSource.USER
                            ):
                                if diag_idx < len(diag_spans):
                                    raw_speaker = diag_spans[diag_idx]["speaker"]
                                    canon_speaker = self.registry.resolve(raw_speaker)
                                    self.registry.record_mention(canon_speaker)
                                    seg.speaker_id = canon_speaker
                                    seg.speaker = canon_speaker
                                    seg.source = SegmentSource.LLM
                                    seg.confidence = 0.9 if canon_speaker != "unknown" else 0.0
                                    diag_idx += 1
                                else:
                                    # Fallback if spans didn't match count
                                    seg.speaker_id = "unknown"
                                    seg.speaker = "unknown"
                                    seg.source = SegmentSource.LLM
                                    seg.confidence = 0.0

                        # Save updated paragraph segments to SQLite
                        self.db.register_segments(p_segs)

                    context_prev = para.text
                    processed_paras += 1
                    if progress_callback:
                        progress_callback(processed_paras, total_paragraphs)

            # Record stage completion
            stage_meta["completed_at"] = datetime.utcnow().isoformat()
            stage_meta["characters_found"] = len(self.registry.character_names())
            self._record_stage_history(job.id, "attribution", "done", stage_meta)

            job.stage_status = StageStatus.DONE
            self.db.save_job(job)

        except Exception as exc:
            logger.exception("Attribution stage failed: %s", exc)
            stage_meta["error"] = str(exc)
            self._record_stage_history(job.id, "attribution", "failed", stage_meta)
            job.stage_status = StageStatus.FAILED
            job.error = str(exc)
            self.db.save_job(job)
            raise
        finally:
            # Memory separation guard: unload LLM model upon stage termination
            if hasattr(self.attributor, "close"):
                try:
                    self.attributor.close()
                except Exception as e:
                    logger.debug("Error closing attributor client: %s", e)

        return self.db.get_all_segments()

    def _record_stage_history(self, job_id: str, stage_name: str, status: str, meta: dict) -> None:
        """Write stage metadata row to SQLite stages table."""
        with self.db._connection() as conn:
            conn.execute(
                """
                INSERT INTO stages (
                    job_id, stage_name, status, started_at, completed_at, error, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    job_id,
                    stage_name,
                    status,
                    meta.get("started_at", datetime.utcnow().isoformat()),
                    meta.get("completed_at"),
                    meta.get("error"),
                    json.dumps(meta),
                ),
            )
