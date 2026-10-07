"""Rule-based speaker attribution with high precision.

Analogy: like a detective who looks for obvious clues (someone saying
"I did it" while pointing at themselves) before calling in the expert
(LLM) for the hard cases.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from ebook2audiobook.attribution.character_registry import CharacterRegistry
from ebook2audiobook.models.segment import SegmentKind

logger = logging.getLogger(__name__)

# Speech verbs that indicate someone spoke
_SPEECH_VERBS = (
    r"(said|asked|replied|answered|whispered|shouted|yelled|cried|muttered|"
    r"murmured|exclaimed|continued|added|remarked|noted|observed|commented|"
    r"declared|stated|explained|argued|insisted|agreed|disagreed|promised|"
    r"warned|threatened|begged|pleaded|laughed|sighed|groaned|snapped|growled)"
)


def apply_rules(
    segments: list[Any],
    registry: CharacterRegistry,
) -> dict[str, tuple[str, float, str]]:
    """
    Apply high-precision rules to attribute speakers to dialogue segments.

    Rules run before the LLM to reduce LLM calls and improve accuracy on
    obvious cases (e.g., '"Hello," said Harry.').

    Parameters
    ----------
    segments:
        List of Segment objects with id, text, kind, speaker_id, etc.
    registry:
        CharacterRegistry for name resolution.

    Returns
    -------
    dict[str, tuple[str, float, str]]
        Mapping from segment_id to (speaker, confidence, evidence).
        Only includes segments where rules matched.
    """
    attributions: dict[str, tuple[str, float, str]] = {}

    for seg in segments:
        if seg.kind != SegmentKind.DIALOGUE:
            continue

        # Rule R1: "..." said NAME or NAME said, "..."
        match = _match_dialogue_tag(seg.text, registry)
        if match:
            speaker, evidence = match
            attributions[seg.id] = (speaker, 0.98, evidence)
            logger.debug(f"Rule R1 matched for {seg.id}: {speaker} ({evidence})")
            continue

        # Rule R2: Quote ending in `,` then short narration with speech verb then another quote
        # This requires looking at neighboring segments - handled in the stage logic

    return attributions


def _match_dialogue_tag(text: str, registry: CharacterRegistry) -> tuple[str, str] | None:
    """
    Match patterns like '"Hello," said Harry.' or 'Harry said, "Hello."'.

    Returns (speaker, evidence) or None if no match.
    """
    # Pattern 1: "...", said NAME
    pattern1 = re.compile(
        rf'"([^"]+)",\s+{_SPEECH_VERBS}\s+(\S+)'
    )
    match1 = pattern1.search(text)
    if match1:
        quote, verb, name = match1.groups()
        speaker = registry.resolve(name.strip())
        evidence = f'tag: "{quote}" {verb} {name}'
        return speaker, evidence

    # Pattern 2: NAME said, "..."
    pattern2 = re.compile(
        rf'^([A-Z][a-zA-Z\s]+)\s+{_SPEECH_VERBS},\s+"([^"]+)"'
    )
    match2 = pattern2.search(text)
    if match2:
        name, verb, quote = match2.groups()
        speaker = registry.resolve(name.strip())
        evidence = f'tag: {name} {verb} "{quote}"'
        return speaker, evidence

    return None


def check_two_quote_pattern(
    current_seg: Any,
    next_seg: Any,
    registry: CharacterRegistry,
) -> tuple[str, float, str] | None:
    """
    Check if current quote ends in `,` and next segment is a quote after
    a short narration with a speech verb.

    Rule R2: Quote A ends in `,` then short narration with speech verb then Quote B
    Implies same speaker for both quotes.

    Returns (speaker, confidence, evidence) or None.
    """
    if current_seg.kind != SegmentKind.DIALOGUE or next_seg.kind != SegmentKind.DIALOGUE:
        return None

    # Check if current quote ends with comma
    if not current_seg.text.rstrip().endswith(","):
        return None

    # This would need to look at the narration between them
    # For now, this is a placeholder - the full implementation
    # would need access to the narration segment between the two dialogue segments
    return None
