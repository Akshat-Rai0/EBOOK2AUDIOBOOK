"""
Casting engine: assigns distinct voices to characters from VoiceCatalogue.

Rules enforced:
1. Frequency Ranking: Characters sorted by line_count descending (narrator always first).
2. Protagonist / 'I' Policy: Character 'I' defaults to Narrator voice.
3. Minor Character Threshold (MINOR_CHARACTER_THRESHOLD = 3): Characters with <= 3 lines
   draw from a 2-voice extras pool ONLY if total_named_characters > available_distinct_voices.
   If voices are plentiful, every character receives a distinct voice.
4. Adjacency Co-occurrence Distinctness: Pairs of characters appearing within a 20-segment
   sliding window receive distinct voices penalized by acoustic proximity.
5. User Locked: Characters with user_locked=True are NEVER overwritten.
6. Engine Isolation: Voice assignments keyed by engine; switching engines keeps profiles
   and re-suggests for the new engine only.
7. Reuses CharacterProfiler from attribution/profiler.py for any profile inference.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Sequence

from ebook2audiobook.attribution.profiler import CharacterProfiler
from ebook2audiobook.audio.catalogue.catalogue import VoiceCatalogue, VoiceEntry
from ebook2audiobook.models.book import Book
from ebook2audiobook.models.cast import Cast, Character, VoiceRef
from ebook2audiobook.models.segment import Segment

logger = logging.getLogger(__name__)

MINOR_CHARACTER_THRESHOLD: int = 3
EXTRAS_POOL_SIZE: int = 2
ADJACENCY_WINDOW_SIZE: int = 20


class Caster:
    """Assigns voices to cast members based on frequency, adjacency, and profiles."""

    def __init__(
        self,
        catalogue: VoiceCatalogue | None = None,
        profiler: CharacterProfiler | None = None,
    ) -> None:
        self.catalogue = catalogue or VoiceCatalogue()
        self.profiler = profiler or CharacterProfiler()

    def build_adjacency_matrix(self, segments: Sequence[Segment]) -> dict[tuple[str, str], int]:
        """
        Count co-occurrences of speakers within a 20-segment sliding window.
        Returns a symmetric mapping (char_a, char_b) -> overlap_count.
        """
        adjacency: dict[tuple[str, str], int] = defaultdict(int)
        n = len(segments)
        for i in range(n):
            speaker_i = segments[i].speaker_id
            if not speaker_i or speaker_i in ("narrator", "unknown"):
                continue

            window_end = min(n, i + ADJACENCY_WINDOW_SIZE + 1)
            for j in range(i + 1, window_end):
                speaker_j = segments[j].speaker_id
                if not speaker_j or speaker_j in ("narrator", "unknown"):
                    continue
                if speaker_i != speaker_j:
                    pair = tuple(sorted((speaker_i, speaker_j)))
                    adjacency[pair] += 1
        return adjacency

    def profile_characters(self, cast: Cast, book: Book) -> None:
        """
        Infer demographic profiles for characters lacking verified profiles using
        the existing CharacterProfiler (pure-regex, advisory only).
        """
        # Gather text snippets where character is mentioned
        snippets_by_name: dict[str, list[str]] = defaultdict(list)
        for chapter in book.chapters:
            for para in chapter.paragraphs:
                for char in cast.characters:
                    if char.matches(para.text) or (char.name in para.text):
                        snippets_by_name[char.id].append(para.text)

        for char in cast.characters:
            if char.user_locked:
                continue
            snippets = snippets_by_name.get(char.id, [])
            if snippets:
                inferred = self.profiler.infer(snippets)
                # Only update if profile is currently unassigned
                if char.profile.gender == "unknown":
                    char.profile.gender = inferred.gender
                    char.profile.gender_confidence = inferred.gender_confidence
                if char.profile.age_bracket == "unknown":
                    char.profile.age_bracket = inferred.age_bracket
                    char.profile.age_confidence = inferred.age_confidence

    def cast_characters(
        self,
        cast: Cast,
        engine: str = "fake",
        segments: Sequence[Segment] | None = None,
    ) -> Cast:
        """
        Suggest voice assignments for an engine. Returns the mutated Cast.
        """
        engine = engine.lower()
        available_voices = self.catalogue.available_for_characters(engine)
        narrator_voice = self.catalogue.get_narrator_voice(engine)
        narrator_voice_id = narrator_voice.id if narrator_voice else "narrator"

        # 1. Narrator voice assignment
        if not cast.narrator.voice_assignments.get(engine):
            cast.narrator.voice_assignments[engine] = narrator_voice_id
        cast.narrator.voice = VoiceRef(
            engine=engine,
            voice_id=cast.narrator.voice_assignments[engine],
        )

        # 2. Compute adjacency co-occurrences if segments provided
        adjacency = self.build_adjacency_matrix(segments) if segments else {}

        # 3. Sort characters by line_count descending
        sorted_chars = sorted(cast.characters, key=lambda c: c.line_count, reverse=True)
        named_characters = [c for c in sorted_chars if c.id not in ("narrator", "unknown")]
        total_named = len(named_characters)

        # Decide if extras pool is active: only if total characters > available voices
        use_extras_pool = total_named > len(available_voices)

        # Separate main vs minor characters if pool is active
        extras_pool: list[VoiceEntry] = []
        if use_extras_pool:
            # Reserve last 2 voices for extras
            if len(available_voices) >= EXTRAS_POOL_SIZE:
                extras_pool = available_voices[-EXTRAS_POOL_SIZE:]
            else:
                extras_pool = list(available_voices)
            candidate_pool = [v for v in available_voices if v not in extras_pool]
        else:
            candidate_pool = list(available_voices)

        assigned_voice_ids: dict[str, str] = {}  # char_id -> voice_id
        assigned_voice_entries: dict[str, VoiceEntry] = {}  # char_id -> VoiceEntry
        extras_round_robin = 0

        # Preserve existing user-locked assignments
        for char in named_characters:
            existing = char.voice_assignments.get(engine)
            if existing and char.user_locked:
                char.voice = VoiceRef(engine=engine, voice_id=existing)
                assigned_voice_ids[char.id] = existing
                entry = self.catalogue.get_voice(engine, existing)
                if entry:
                    assigned_voice_entries[char.id] = entry
                    if entry in candidate_pool:
                        candidate_pool.remove(entry)

        # Assign voices to remaining characters
        for char in named_characters:
            if char.id in assigned_voice_ids:
                continue

            # Special case: Protagonist "I" defaults to Narrator voice
            if char.name.strip().upper() == "I" or char.id == "i":
                char.voice_assignments[engine] = narrator_voice_id
                char.voice = VoiceRef(engine=engine, voice_id=narrator_voice_id)
                assigned_voice_ids[char.id] = narrator_voice_id
                continue

            # Minor character check
            is_minor = char.line_count <= MINOR_CHARACTER_THRESHOLD
            if is_minor and use_extras_pool:
                if extras_pool:
                    extra_voice = extras_pool[extras_round_robin % len(extras_pool)]
                    extras_round_robin += 1
                    char.voice_assignments[engine] = extra_voice.id
                    char.voice = VoiceRef(engine=engine, voice_id=extra_voice.id)
                    assigned_voice_ids[char.id] = extra_voice.id
                    assigned_voice_entries[char.id] = extra_voice
                else:
                    char.voice_assignments[engine] = narrator_voice_id
                    char.voice = VoiceRef(engine=engine, voice_id=narrator_voice_id)
                continue

            # Select best voice from candidate_pool
            best_voice = self._select_best_voice(
                char=char,
                candidates=candidate_pool,
                engine=engine,
                assigned_voices=assigned_voice_entries,
                adjacency=adjacency,
                narrator_voice_id=narrator_voice_id,
            )

            if best_voice:
                char.voice_assignments[engine] = best_voice.id
                char.voice = VoiceRef(engine=engine, voice_id=best_voice.id)
                assigned_voice_ids[char.id] = best_voice.id
                assigned_voice_entries[char.id] = best_voice
                if best_voice in candidate_pool:
                    candidate_pool.remove(best_voice)
            else:
                # Pool exhausted: fallback to extras or narrator
                fallback_id = extras_pool[0].id if extras_pool else narrator_voice_id
                char.voice_assignments[engine] = fallback_id
                char.voice = VoiceRef(engine=engine, voice_id=fallback_id)
                assigned_voice_ids[char.id] = fallback_id

        return cast

    def _select_best_voice(
        self,
        char: Character,
        candidates: list[VoiceEntry],
        engine: str,
        assigned_voices: dict[str, VoiceEntry],
        adjacency: dict[tuple[str, str], int],
        narrator_voice_id: str,
    ) -> VoiceEntry | None:
        if not candidates:
            return None

        # Filter by verified profile tags if any exist
        verified_candidates = self.catalogue.filter_verified(
            engine=engine,
            gender=char.profile.gender,
            age_bracket=char.profile.age_bracket,
        )
        pool = [c for c in candidates if c in verified_candidates]
        if not pool:
            pool = list(candidates)

        # Score candidates: lower penalty is better
        def score(voice: VoiceEntry) -> float:
            penalty = 0.0
            for other_char_id, other_voice in assigned_voices.items():
                pair = tuple(sorted((char.id, other_char_id)))
                cooc_weight = adjacency.get(pair, 0)
                if cooc_weight > 0:
                    dist = voice.acoustic_distance_to(other_voice)
                    if dist < 1.0:
                        # Collision penalty inversely proportional to distance,
                        # weighted by co-occurrence
                        penalty += (1.0 - dist) * cooc_weight * 10.0
            return penalty

        return min(pool, key=score)
