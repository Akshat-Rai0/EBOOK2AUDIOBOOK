"""Tests for CharacterProfiler (Item 5).

All tests are pure-regex, offline — no network, no LLM.
"""

from __future__ import annotations

import pytest

from ebook2audiobook.attribution.profiler import CharacterProfiler


@pytest.fixture()
def profiler() -> CharacterProfiler:
    return CharacterProfiler()


# ---------------------------------------------------------------------------
# Gender inference
# ---------------------------------------------------------------------------


class TestGenderInference:
    def test_clear_female_pronouns(self, profiler: CharacterProfiler) -> None:
        """'she said', 'her voice' → gender=female with high confidence."""
        snippets = [
            "She walked into the room.",
            "Everyone admired her courage.",
            '"Stop," she cried, her hands trembling.',
        ]
        profile = profiler.infer(snippets)
        assert profile.gender == "female"
        assert profile.gender_confidence > 0.7

    def test_clear_male_pronouns(self, profiler: CharacterProfiler) -> None:
        """Predominantly 'he/him/his' → gender=male."""
        snippets = [
            "He raised his sword.",
            "The king looked at him.",
            "His laughter echoed through the hall.",
        ]
        profile = profiler.infer(snippets)
        assert profile.gender == "male"
        assert profile.gender_confidence > 0.7

    def test_nonbinary_pronouns(self, profiler: CharacterProfiler) -> None:
        """Predominantly 'they/them/their' → gender=nonbinary."""
        snippets = [
            "They arrived late.",
            "Everyone greeted them warmly.",
            "Their bag was still on the chair.",
        ]
        profile = profiler.infer(snippets)
        assert profile.gender == "nonbinary"
        assert profile.gender_confidence > 0.5

    def test_no_gender_signals(self, profiler: CharacterProfiler) -> None:
        """Snippets with no pronouns → gender=unknown, confidence=0.0."""
        snippets = [
            "The door opened.",
            "A sound echoed down the corridor.",
        ]
        profile = profiler.infer(snippets)
        assert profile.gender == "unknown"
        assert profile.gender_confidence == 0.0

    def test_conflicting_evidence_lowers_confidence(self, profiler: CharacterProfiler) -> None:
        """Mixed he/she signals → confidence lower than a clear signal."""
        # 2 female hits vs 2 male hits → tie → confidence ≤ 0.5
        snippets = [
            "She entered.  He left.",
            "Her coat.  His hat.",
        ]
        profile = profiler.infer(snippets)
        # Tied — exact winner depends on Counter.most_common ordering, but confidence must be low
        assert profile.gender_confidence <= 0.5

    def test_empty_snippets(self, profiler: CharacterProfiler) -> None:
        """Empty input → all unknowns at 0.0."""
        profile = profiler.infer([])
        assert profile.gender == "unknown"
        assert profile.gender_confidence == 0.0
        assert profile.age_bracket == "unknown"
        assert profile.age_confidence == 0.0


# ---------------------------------------------------------------------------
# Age bracket inference
# ---------------------------------------------------------------------------


class TestAgeInference:
    def test_elderly_marker(self, profiler: CharacterProfiler) -> None:
        """'old man' → age_bracket=elderly."""
        snippets = ["The old man sat by the fire.", "He was elderly and frail."]
        profile = profiler.infer(snippets)
        assert profile.age_bracket == "elderly"
        assert profile.age_confidence > 0.0

    def test_child_marker(self, profiler: CharacterProfiler) -> None:
        """'girl' and 'child' → age_bracket=child."""
        snippets = ["The girl ran ahead.", "She was just a child of eight."]
        profile = profiler.infer(snippets)
        assert profile.age_bracket == "child"

    def test_young_adult_marker(self, profiler: CharacterProfiler) -> None:
        """'young man' → age_bracket=young_adult."""
        snippets = ["A young man stepped forward.", "The teen looked nervous."]
        profile = profiler.infer(snippets)
        assert profile.age_bracket == "young_adult"

    def test_no_age_signals(self, profiler: CharacterProfiler) -> None:
        """No age markers → age_bracket=unknown, confidence=0.0."""
        snippets = ["She walked to the window.", "He said nothing."]
        profile = profiler.infer(snippets)
        assert profile.age_bracket == "unknown"
        assert profile.age_confidence == 0.0

    def test_elderly_beats_adult_on_tie(self, profiler: CharacterProfiler) -> None:
        """When elderly and adult each have 1 hit, elderly wins (priority)."""
        # Use disjoint markers: 'elderly' is only elderly, 'lady' is only adult.
        # ('old woman' is *not* a 1–1 tie — adult also matches 'woman'.)
        snippets = ["She was elderly, a real lady."]
        profile = profiler.infer(snippets)
        assert profile.age_bracket == "elderly"

    def test_grandmother_keyword(self, profiler: CharacterProfiler) -> None:
        """'grandmother' maps to elderly."""
        profile = profiler.infer(["Her grandmother smiled."])
        assert profile.age_bracket == "elderly"


# ---------------------------------------------------------------------------
# Combined / integration
# ---------------------------------------------------------------------------


class TestCombined:
    def test_female_elderly_combination(self, profiler: CharacterProfiler) -> None:
        """'she' + 'grandmother' → female elderly."""
        snippets = [
            "Her grandmother sat in the chair.",
            "She had seen many winters.",
        ]
        profile = profiler.infer(snippets)
        assert profile.gender == "female"
        assert profile.age_bracket == "elderly"

    def test_male_child_combination(self, profiler: CharacterProfiler) -> None:
        """'he' + 'boy' → male child."""
        snippets = ["The boy ran after him.", "He tripped and fell."]
        profile = profiler.infer(snippets)
        assert profile.gender == "male"
        assert profile.age_bracket == "child"

    def test_confidence_scores_between_zero_and_one(self, profiler: CharacterProfiler) -> None:
        """All confidence fields must be in [0.0, 1.0]."""
        snippets = ["She said. He replied. They laughed. The old man coughed."]
        profile = profiler.infer(snippets)
        assert 0.0 <= profile.gender_confidence <= 1.0
        assert 0.0 <= profile.age_confidence <= 1.0
