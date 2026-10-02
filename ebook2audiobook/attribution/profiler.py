"""Character profile inference from textual evidence.

Uses pure regex/rule-based heuristics (no LLM, no network).
Profiles are *advisory only* — the user can always override with a single
command.  Never present inferred gender or age as authoritative facts.

Everyday analogy for the confidence score:
    Think of it like a jury verdict by show of hands.  If 8 out of 10 clues
    point to "female" and 2 are neutral, confidence = 0.80.  A single
    contradictory clue lowers the score further.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence

from ebook2audiobook.models.cast import CharacterProfile

# ---------------------------------------------------------------------------
# Compiled patterns (module-level so they compile once)
# ---------------------------------------------------------------------------

_MALE_PRONOUNS = re.compile(
    r"\b(he|him|his)\b",
    re.IGNORECASE,
)
_FEMALE_PRONOUNS = re.compile(
    r"\b(she|her|hers)\b",
    re.IGNORECASE,
)
_NONBINARY_PRONOUNS = re.compile(
    r"\b(they|them|their|theirs)\b",
    re.IGNORECASE,
)

_CHILD_MARKERS = re.compile(
    r"\b(boy|girl|child|children|kid|kids|toddler|infant|baby|juvenile|pupil)\b",
    re.IGNORECASE,
)
_YOUNG_ADULT_MARKERS = re.compile(
    r"\b(young\s+(?:man|woman|lady|fellow)|teen|teenager|adolescent|youth|lad|lass)\b",
    re.IGNORECASE,
)
_ADULT_MARKERS = re.compile(
    r"\b(man|woman|gentleman|lady|sir|madam|mister|mrs|miss|ms)\b",
    re.IGNORECASE,
)
_ELDERLY_MARKERS = re.compile(
    r"\b(old(?:\s+(?:man|woman|lady|fellow|chap))?|elder|elderly|aged|ancient|"
    r"grandmother|grandfather|grandma|grandpa|granny|gran|granddad|geezer|crone)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class CharacterProfiler:
    """Infer gender and age bracket from raw text snippets.

    Parameters
    ----------
    snippets:
        Sentences or paragraphs in which the character appears.  Pass as many
        as available — more evidence yields higher confidence.
    """

    def infer(self, snippets: Sequence[str]) -> CharacterProfile:
        """Return a :class:`CharacterProfile` inferred from *snippets*.

        Confidence is the fraction of evidence tokens (pronoun / age-marker
        hits) that agree with the chosen label divided by the total hits
        examined.  When there are no hits the label defaults to ``"unknown"``
        with confidence ``0.0``.

        Parameters
        ----------
        snippets:
            Iterable of text passages associated with the character.
        """
        text = " ".join(snippets)
        gender, gender_conf = self._infer_gender(text)
        age, age_conf = self._infer_age(text)
        return CharacterProfile(
            gender=gender,
            gender_confidence=gender_conf,
            age_bracket=age,
            age_confidence=age_conf,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_gender(text: str) -> tuple[str, float]:
        """Count pronoun hits and return (label, confidence).

        Each regex match counts as one vote.  The winning category wins the
        label; confidence = winner_votes / total_votes.  Ties and zero-total
        both return ``("unknown", 0.0)``.
        """
        votes: Counter[str] = Counter()
        votes["male"] += len(_MALE_PRONOUNS.findall(text))
        votes["female"] += len(_FEMALE_PRONOUNS.findall(text))
        votes["nonbinary"] += len(_NONBINARY_PRONOUNS.findall(text))

        total = sum(votes.values())
        if total == 0:
            return "unknown", 0.0

        winner, winner_count = votes.most_common(1)[0]
        confidence = round(winner_count / total, 4)

        # Exact tie between top two → lower confidence to reflect ambiguity
        top_two = votes.most_common(2)
        if len(top_two) == 2 and top_two[0][1] == top_two[1][1]:
            confidence = round(confidence * 0.5, 4)

        return winner, confidence

    @staticmethod
    def _infer_age(text: str) -> tuple[str, float]:
        """Count age-marker hits and return (label, confidence).

        Priority order when counts tie: elderly > child > young_adult > adult
        (more specific markers should win over generic ones).
        """
        votes: Counter[str] = Counter()
        votes["child"] += len(_CHILD_MARKERS.findall(text))
        votes["young_adult"] += len(_YOUNG_ADULT_MARKERS.findall(text))
        votes["adult"] += len(_ADULT_MARKERS.findall(text))
        votes["elderly"] += len(_ELDERLY_MARKERS.findall(text))

        total = sum(votes.values())
        if total == 0:
            return "unknown", 0.0

        # Resolve ties with the priority order above (more specific beats generic)
        _priority = ["elderly", "child", "young_adult", "adult"]
        max_count = votes.most_common(1)[0][1]
        # Among all labels tied at max_count, pick highest priority
        winner = next(label for label in _priority if votes[label] == max_count)
        confidence = round(max_count / total, 4)

        return winner, confidence
