"""
Production Attribution Stage (M4 Step 3 - V2 ID-based).

Applies the calibrated M4 LLM recipe to assign character speakers to dialogue segments:
- Model: llama3.2:3b (or configured local model).
- Quote markers [Q1], [Q2] for ID-based attribution (fixes P3).
- Closed-set speaker list from seeded cast (fixes P6).
- Rules before LLM for high-precision attribution.
- Alternation correction after LLM for A/?/A patterns.
- Real confidence buckets: high (0.95), medium (0.70), low (0.40) (fixes P5).
- User-edited segments (source=USER) are never overwritten.
- Stage metadata (model tag, prompt version, commit hash) recorded in state.sqlite.
- Explicit model eviction upon completion to guarantee memory separation before TTS.
"""

from __future__ import annotations

import json
import logging
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from ebook2audiobook.attribution.alternation import apply_alternation_correction
from ebook2audiobook.attribution.attributor import Attributor
from ebook2audiobook.attribution.character_registry import CharacterRegistry
from ebook2audiobook.attribution.rules import apply_rules
from ebook2audiobook.chunker.dialogue_segmenter import DialogueSegmenter
from ebook2audiobook.models.book import Book
from ebook2audiobook.models.job import Job, StageStatus
from ebook2audiobook.models.segment import Segment, SegmentKind, SegmentSource
from ebook2audiobook.models_manager.manager import ModelManager
from ebook2audiobook.store.db import JobDatabase

logger = logging.getLogger(__name__)

PROMPT_VERSION = "2.0-m4-id-based"


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
        model_manager: ModelManager | None = None,
    ) -> None:
        self.db = db
        self.attributor = attributor
        self.segmenter = segmenter or DialogueSegmenter()
        self.registry = registry or CharacterRegistry()
        self.model_manager = model_manager

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

        started_at = datetime.now(UTC).isoformat()
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
            initial_segs = [s for s in initial_segs if s.text and s.text.strip()]
            self.db.register_segments(initial_segs)
            all_segments = initial_segs
        else:
            all_segments = [s for s in existing_segs if s.text and s.text.strip()]

        # Index segments by paragraph_id for in-order attribution
        para_to_segs: dict[str, list[Segment]] = {}
        for seg in all_segments:
            para_to_segs.setdefault(seg.paragraph_id, []).append(seg)

        total_paragraphs = sum(len(ch.paragraphs) for ch in book.chapters)
        processed_paras = 0

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
                        # V2: Step 1 - Apply rules first (high-precision)
                        rule_attributions = apply_rules(p_segs, self.registry)

                        # Apply rule attributions
                        for seg_id, (speaker, confidence, evidence) in rule_attributions.items():
                            for seg in p_segs:
                                if seg.id == seg_id:
                                    seg.speaker_id = speaker
                                    seg.speaker = speaker
                                    seg.confidence = confidence
                                    seg.source = SegmentSource.RULE
                                    seg.evidence = evidence
                                    self.registry.record_mention(speaker)

                        # V2: Step 2 - Mark quotes with IDs for LLM
                        quote_id_map: dict[str, Segment] = {}
                        quote_idx = 0

                        # Build marked paragraph by replacing unknown dialogue
                        # segments with [Q1], [Q2], etc. We use the original
                        # paragraph text and perform sequential replacements
                        # to preserve original spacing exactly.
                        marked_paragraph = para.text
                        for seg in p_segs:
                            is_unknown_dialogue = (
                                seg.kind == SegmentKind.DIALOGUE
                                and seg.source != SegmentSource.USER
                                and seg.speaker_id == "unknown"
                            )
                            if is_unknown_dialogue:
                                quote_id = f"Q{quote_idx + 1}"
                                quote_id_map[quote_id] = seg
                                # Replace first occurrence of segment text with marker
                                # This preserves original spacing. We use replace() with count=1
                                # to ensure we only replace the first occurrence, which corresponds
                                # to the current segment being processed in order.
                                marker = f"[{quote_id}]"
                                marked_paragraph = marked_paragraph.replace(seg.text, marker, 1)
                                quote_idx += 1

                        # V2: Step 3 - Call LLM with quote markers
                        if quote_id_map and hasattr(self.attributor, "attribute_quotes"):
                            allowed_speakers = (
                                self.registry.character_names() + ["unknown"]
                            )
                            attributions = self.attributor.attribute_quotes(
                                marked_paragraph=marked_paragraph,
                                allowed_speakers=allowed_speakers,
                            )

                            # Apply LLM attributions by ID
                            for attr in attributions:
                                if attr.quote_id in quote_id_map:
                                    seg = quote_id_map[attr.quote_id]
                                    # Rule: unknown speakers must be labeled "unknown",
                                    # never silently collapsed into "narrator".
                                    resolved_speaker = (
                                        "unknown"
                                        if attr.speaker_id in ("narrator", "unknown")
                                        else attr.speaker_id
                                    )
                                    seg.speaker_id = resolved_speaker
                                    seg.speaker = resolved_speaker
                                    seg.confidence = attr.confidence
                                    seg.source = SegmentSource.LLM
                                    seg.evidence = "llm"
                                    self.registry.record_mention(resolved_speaker)

                        # V2: Step 4 - Apply alternation correction
                        alternation_corrections = apply_alternation_correction(p_segs)
                        for seg_id, (speaker, confidence, evidence) in (
                            alternation_corrections.items()
                        ):
                            for seg in p_segs:
                                if seg.id == seg_id:
                                    seg.speaker_id = speaker
                                    seg.speaker = speaker
                                    seg.confidence = confidence
                                    seg.source = SegmentSource.RULE
                                    seg.evidence = evidence

                        # Save updated paragraph segments to SQLite
                        self.db.register_segments(p_segs)

                    processed_paras += 1
                    if progress_callback:
                        progress_callback(processed_paras, total_paragraphs)

            # Record stage completion
            stage_meta["completed_at"] = datetime.now(UTC).isoformat()
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

            # If ModelManager is provided, unload the LLM model
            if self.model_manager is not None:
                ollama_model = getattr(self.attributor, "model", None)
                if ollama_model:
                    try:
                        self.model_manager.unload_ollama(ollama_model)
                    except Exception as e:
                        logger.debug("Error unloading Ollama model via ModelManager: %s", e)

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
                    meta.get("started_at", datetime.now(UTC).isoformat()),
                    meta.get("completed_at"),
                    meta.get("error"),
                    json.dumps(meta),
                ),
            )
