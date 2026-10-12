"""Speaker alternation heuristic for correcting ambiguous attribution.

Analogy: like a tennis referee who notices when two players keep
serving back and forth, and can predict who serves next based on
the pattern.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def apply_alternation_correction(
    segments: list[Any],
    min_confidence: float = 0.0,
    max_confidence: float = 0.75,
) -> dict[str, tuple[str, float, str]]:
    """
    Apply A/?/A pattern correction to low-confidence speaker assignments.

    If we have pattern Speaker A / Unknown / Speaker A, the middle line
    is likely also Speaker A (two characters talking back and forth).

    Parameters
    ----------
    segments:
        List of Segment objects with id, kind, speaker_id, confidence, source.
    min_confidence:
        Minimum confidence to consider a segment as "known" (default 0.0).
    max_confidence:
        Maximum confidence to correct a segment (don't touch high-confidence
        segments even if they'd fit the pattern, default 0.75).

    Returns
    -------
    dict[str, tuple[str, float, str]]
        Mapping from segment_id to (speaker, confidence, evidence).
        Only includes segments where alternation correction was applied.
    """
    corrections: dict[str, tuple[str, float, str]] = {}

    # Only consider dialogue segments
    dialogue_segs = [s for s in segments if s.kind.value == "dialogue"]

    for i in range(1, len(dialogue_segs) - 1):
        prev_seg = dialogue_segs[i - 1]
        curr_seg = dialogue_segs[i]
        next_seg = dialogue_segs[i + 1]

        # Skip if middle segment is user-locked (manual assignment)
        if curr_seg.source == "user":
            continue

        # Skip if middle segment is already high-confidence
        if curr_seg.confidence >= max_confidence:
            continue

        # Check for A/?/A pattern
        if (
            prev_seg.speaker_id
            and prev_seg.speaker_id == next_seg.speaker_id
            and prev_seg.confidence >= min_confidence
            and next_seg.confidence >= min_confidence
            and prev_seg.speaker_id != curr_seg.speaker_id
        ):
            # Apply correction
            speaker = prev_seg.speaker_id
            confidence = 0.75  # Alternation is a strong heuristic but not certain
            evidence = "alternation: A/?/A pattern suggests same speaker"

            corrections[curr_seg.id] = (speaker, confidence, evidence)
            logger.debug(
                f"Alternation correction for {curr_seg.id}: "
                f"{curr_seg.speaker_id} -> {speaker} ({evidence})"
            )

    return corrections
