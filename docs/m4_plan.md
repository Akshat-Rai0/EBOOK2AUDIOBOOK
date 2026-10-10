# M4 Architecture & Implementation Plan: Casting and Multi-Voice Synthesis

**Milestone:** M4 (`m4-casting-multivoice`)  
**Date:** 2026-10-01 (Updated 2026-10-11)  
**Authors:** Akshat Rai, Divyanshu Bhusan  
**Status:** IN PROGRESS (V2 Attribution, Character Registry, Dialogue Segmenter & Merged M2b TTS Implemented; Voice Catalogue & Review Gate implementation approved for Steps 1-3 with FakeTTS only; Step 4 real-engine multi-voice awaiting separate approval)

---

## 1. Stage Graph and Pipeline Flow

The multi-voice conversion pipeline extends M2's linear state machine with attribution, character casting, an interactive review gate, and selective re-synthesis:

```
[Ingest] ──> [Chunk / Segment] ──> [Attribute (LLM)] ──> [Cast (Voice Suggestion)]
                                                              │
                                                              ▼
                                                   ╔═════════════════════╗
                                                   ║     REVIEW GATE     ║
                                                   ║  (cast review/appr) ║
                                                   ╚═════════════════════╝
                                                              │
                                                      [Approve / --auto]
                                                              │
                                                              ▼
                                                   [Synthesize (Multi-Voice)]
                                                              │
                                                              ▼
                                                    [Assemble & Normalize]
                                                              │
                                                              ▼
                                                      [Export MP3 & M4B]
```

### Stage Graph Execution Rules
1. **Linear Progression:** Stages execute strictly in sequence: `ingest` → `chunk` → `attribute` → `cast` → `review_gate` → `synthesize` → `assemble` → `export`.
2. **Review Gate Pause:** Upon completing the `cast` stage, the pipeline pauses with status `PENDING_REVIEW` unless invoked with `--auto`. The user reviews unassigned or low-confidence lines (`castbook cast review`) and approves (`castbook cast approve`).
3. **Sequential Model Lifetime (Memory Guard):**
   - In `attribute`: The LLM (`llama3.2:3b` in Ollama) runs. Upon stage completion, the orchestrator explicitly unloads the model from Ollama memory (`keep_alive: 0`), enforced by `ModelManager`.
   - `ModelManager` verifies active RAM before loading any TTS engine. The LLM and TTS models are never held in memory simultaneously.
4. **Selective Re-Synthesis:** Re-running after manual edits invalidates only the segments whose synthesis cache key changed (see §7.1).

---

## 2. Schema Changes & Migrations

### 2.1 Segment Model (`ebook2audiobook/models/segment.py`) — already implemented

```python
class Segment(BaseModel):
    id: str  # c<CC>-p<PPP>-s<SS>
    chapter_index: int
    paragraph_id: str
    text: str
    speaker_id: str = "narrator"  # References Character.id
    kind: SegmentKind = SegmentKind.NARRATION
    confidence: float = 1.0  # LLM confidence or 1.0 for rule/user
    source: SegmentSource = SegmentSource.RULE
    evidence: str | None = None  # Short snippet/rationale from attribution
    audio_path: str | None = None
    voice_hash: str | None = None  # Synthesis cache key (see §7.1)
    status: SegmentStatus = SegmentStatus.PENDING
```

### 2.2 Character Model (`ebook2audiobook/models/cast.py`) — already implemented

```python
class CharacterProfile(BaseModel):
    gender: str = "unknown"  # "male", "female", "nonbinary", "unknown"
    gender_confidence: float = 0.0
    age_bracket: str = "unknown"  # "child", "young_adult", "adult", "elderly", "unknown"
    age_confidence: float = 0.0

class Character(BaseModel):
    id: str                          # Canonical slug
    display_name: str
    aliases: list[str]
    profile: CharacterProfile
    line_count: int = 0
    first_chapter: int = 0
    user_locked: bool = False        # Never re-cast or merged when True
    voice_assignments: dict[str, str] # {engine_name: voice_id}
```

### 2.3 SQLite Schema Migration (`projects/<name>/state.sqlite`) — already implemented
Columns `speaker_id`, `kind`, `confidence`, `source`, `evidence`, `voice_hash` added via `ALTER TABLE` migration in `init_schema()`.

---

## 3. Segmenter Update — already implemented

Paragraphs split into alternating dialogue / narration spans. Segment IDs are deterministic and stable (`c<CC>-p<PPP>-s<SS>`). Whitespace-only segments are filtered before registration.

---

## 4. Production Attribution Stage & Character Registry — already implemented

- V2 hybrid pipeline: rules → LLM (`llama3.2:3b`, ID-based `[Q1]`/`[Q2]`) → alternation heuristic.
- `CharacterRegistry`: alias merging, collision detection, merge history, undo, ambiguity flagging.
- `CharacterProfiler` in `attribution/profiler.py`: rule-based gender/age inference from pronouns & markers (no LLM, no network). Output always advisory (confidence < 1.0). **The caster reuses `CharacterProfiler` directly; no profiling logic is duplicated in `caster.py`.**

---

## 5. Voice Catalogue & Casting Algorithm

### 5.1 Voice Catalogue (`ebook2audiobook/audio/catalogue/`)

Files: `vits_voices.json`, `xtts_voices.json`, `fake_voices.json`  
Loaded by `catalogue.py:VoiceCatalogue`.

**CRITICAL — Unverified metadata rule:**  
VCTK speaker IDs (p225–p376) do not reliably match the `speaker-info.txt` demographics (Coqui issue #2258). XTTS reference speaker names carry no verified demographics either. Therefore **every per-voice tag (gender, age_bracket, accent) carries a `verified: false` flag by default.** The user sets `verified: true` by listening and running `castbook cast label` (a future command). The caster ONLY uses a tag when `verified: true`; otherwise it falls back to distinctness-only assignment.

Catalogue schema per voice entry:
```json
{
  "id": "p225",
  "name": "Speaker p225",
  "engine": "vits",
  "gender": "unknown",
  "gender_verified": false,
  "age_bracket": "unknown",
  "age_verified": false,
  "accent": "unknown",
  "accent_verified": false,
  "licence": "MIT",
  "narrator_reserved": false
}
```
One voice per engine is flagged `"narrator_reserved": true` and permanently excluded from character assignment.

### 5.2 Casting Algorithm (`ebook2audiobook/casting/caster.py`)

**Minor Character Threshold constant:**
```python
MINOR_CHARACTER_THRESHOLD: int = 3  # Segments; configurable via config.toml [cast]
EXTRAS_POOL_SIZE: int = 2           # Voices reserved for minor characters
```
**Important refinement:** Minor characters are only demoted to the extras pool if the total number of named speaking characters *exceeds* the number of available distinct catalogue voices (excluding narrator-reserved and already-assigned voices). When there are enough voices for everyone, every character gets a distinct one regardless of line count.

**Algorithm steps:**
1. **Frequency Rank:** Characters sorted by `line_count` descending; narrator always first.
2. **Profile Match (verified only):** Filter catalogue where `gender_verified=true AND gender == character.profile.gender`, same for `age_bracket`. If no verified match exists, widen to all unverified voices.
3. **Co-occurrence Distinctness (adjacency-weighted, NOT chapter-based):**
   - Build segment-level adjacency matrix: two characters are adjacent if they both appear within a 20-segment sliding window.
   - Co-occurrence weight = count of such overlapping windows.
   - During voice assignment, voices acoustically similar to an already-assigned voice are penalized for characters with high co-occurrence weight.
   - **Implementation reuses `attribution/profiler.py`** for any profiling lookups; caster.py does not contain its own profiling logic.
4. **Minor character extras pool:**
   - If `total_named_characters > available_distinct_voices`, characters with `line_count <= MINOR_CHARACTER_THRESHOLD` draw from a 2-voice pool (round-robin) instead of getting unique voices.
   - If the pool is not needed (enough voices for all), every character gets a distinct voice.
5. **Protagonist / "I" policy:** Speaker `"I"` resolves to the Narrator voice by default. `user_locked` manual assignment overrides this.
6. **Engine-keyed assignments:** `voice_assignments = {engine_name: voice_id}`. Switching `--engine` keeps all existing `(character, other_engine)` assignments intact, re-runs suggestions for the new engine, and never overrides user-locked choices.

### 5.3 `cast approve` — cast integrity hash
`castbook cast approve` computes `SHA256(cast.json content)` and stores it as `cast_hash` in `state.sqlite` (or a `cast_state` JSON sidecar). Any subsequent edit to `cast.json` invalidates the approval. Before synthesis, the pipeline verifies the stored hash matches the current `cast.json`. `--auto` mode accepts all, sets `unknown` → narrator voice, and auto-approves.

---

## 6. Review Commands (CLI Interface)

All commands wrap pure library functions from `ebook2audiobook.casting`:

| Command | Description |
|---|---|
| `castbook cast show -p <p>` | Table: character, line count, profile (with verified flags shown), assigned voice. |
| `castbook cast suggest -p <p> [--engine <vits\|xtts\|fake>]` | Run casting algorithm; writes `cast.json`; does NOT approve. |
| `castbook cast review -p <p>` | Step-through unknown & low-confidence (< 0.80) segments with 2-line preceding context. User picks speaker; sets `source=user`. |
| `castbook cast sample -p <p> <character>` | Synthesize 3–5 seconds of one of their real attributed lines in their assigned voice; plays or writes to `projects/<p>/samples/<character>.wav`. |
| `castbook cast voice -p <p> <character> <voice-id>` | Assign specific voice; marks `user_locked=True`. |
| `castbook cast rename -p <p> <character> <new-name>` | Rename display name; logs to merge history for undo. |
| `castbook cast merge -p <p> <source> <target>` | Merge two characters. Transactional: updates both `cast.json` and SQLite segments atomically. Records to merge history. Winner precedence: if `source` is `user_locked`, winner is `source`; if `target` is `user_locked`, winner is `target`; if both are `user_locked`, **abort with error** (ask user to manually resolve). Logs to `merge_history.json` for undo. |
| `castbook cast split -p <p> <character> <segment-id>` | Detach segment from a merged character; creates or restores separate entry. |
| `castbook cast approve -p <p> [--auto]` | Store SHA256 of approved cast. `--auto`: sets unknown → narrator, accepts all suggestions. |
| `castbook cast undo -p <p>` | Revert last merge or rename from `merge_history.json`. Works for both `cast.json` and SQLite segments transactionally. |

---

## 7. Multi-Voice Synthesis, Assembly, and QA

### 7.1 Synthesis Cache Key (extends M2 `voice_hash`)

**One hash, never two.** The existing `voice_hash` column in `segments` is extended to cover all cache-busting inputs:

```python
def compute_voice_hash(
    text: str,
    voice_id: str,
    engine: str,
    engine_version: str,   # e.g. "coqui-tts:0.22.0"
    normaliser_version: str,  # e.g. "1.0"
) -> str:
    """Delimiter-safe SHA-256 cache key for a synthesis call."""
    parts = [text, voice_id, engine, engine_version, normaliser_version]
    payload = "\x00".join(parts)  # NUL delimiter — safe for all text/version strings
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
```

Segment re-synthesis query: `SELECT * FROM segments WHERE status != 'done' OR voice_hash != :new_hash`.

### 7.2 Assembly & Pauses

Named constants (defined in `ebook2audiobook/audio/processor.py`):

```python
PAUSE_TAG_TRANSITION_MS: int = 150   # Dialogue ↔ narration tag within same sentence
PAUSE_SPEAKER_TURN_MS: int = 450     # Speaker changes between sentences
PAUSE_PARAGRAPH_MS: int = 700        # Already defined in M2 (unchanged)
PAUSE_CHAPTER_MS: int = 1500         # Already defined in M2 (unchanged)
PAUSE_SENTENCE_MS: int = 300         # Already defined in M2 (unchanged)
PAUSE_SPLIT_PIECE_MS: int = 150      # Already defined in M2 (unchanged)
```

**Pause precedence (highest wins when multiple rules apply):**
1. Chapter boundary: 1500 ms
2. Paragraph boundary: 700 ms
3. Sentence boundary: 300 ms
4. Speaker turn (cross-sentence): 450 ms
5. Tag transition within sentence: 150 ms
6. Split-piece join: 150 ms

### 7.3 Multi-Voice QA Checks

- QA-1: All segments have a valid voice assigned in catalogue.
- QA-2: Report count and % of `unknown` speaker lines.
- QA-3: No character has multiple discordant voices in the same output.
- QA-4: Utterances < 4 characters padded with 50 ms trailing silence.

---

## 8. Caveat Verification

| # | Caveat | Severity | Mitigation |
|---|---|---|---|
| 1 | Voice pool exhaustion | Medium | Extras pool only kicks in when `named_chars > distinct_voices`. Minor characters ($\le K=3$ lines) demoted only if pool is actually needed. |
| 2 | Engine-specific voices | High | Assignments keyed by `(character_id, engine)`. Switch engine: keep profile, re-suggest, never overwrite `user_locked`. |
| 3 | Gender/age misclassification | Medium | All catalogue tags `verified: false` by default. Caster uses tag only when `verified: true`. Advisory profile shown in `cast show`. |
| 4 | Short utterances | High | Pad segments < 15 chars with 50 ms trailing silence. |
| 5 | Choppy tag flow | Medium | 150 ms `PAUSE_TAG_TRANSITION_MS` named constant. |
| 6 | Alias collisions | High | Conservative matching, `user_locked` precedence in merge, abort if both sides user-locked. |
| 7 | First-person narratives ("I") | Medium | `"I"` → narrator voice by default; explicit `cast voice` overrides. |
| 8 | Thoughts & epistolary | Low | `thought` kind → narrator voice. |
| 9 | Cascade invalidation | High | Extended `voice_hash` includes engine, engine_version, normaliser_version with NUL delimiter. Only changed segments re-synthesized. |
| 10 | XTTS latent memory leaks | Medium | Latents bounded by cast size (≤ 30 voices ≈ 15 MB). |
| 11 | Licensing | High | All entries annotated with `licence` field; tracked in `docs/LICENSES.md`. |
| 12 | Long attribution runtime | Medium | Progress bars, per-chapter timing, SQLite checkpointing. |

---

## 9. Test Plan

All tests run offline with `FakeTTS` and deterministic `FakeAttributor`. Steps 1-3 are approved; Step 4 (real-engine multi-voice) awaits separate approval.

### Approved (Steps 1-3): Catalogue, Caster, CLI Review

1. **`test_voice_catalogue.py`**
   - Load catalogue for each engine (fake/vits/xtts from JSON).
   - All entries default to `*_verified=false`; `filter_verified()` returns empty unless tag is verified.
   - `narrator_reserved` voice excluded from `available_for_character()`.

2. **`test_caster.py`**
   - Frequency ranking: highest `line_count` character gets first pick.
   - Co-occurrence distinctness (20-segment window): two characters sharing 15+ adjacent segments get different voices.
   - Minor character extras pool ONLY triggers when `named_chars > available_voices`.
   - When pool is NOT needed, every character gets a distinct voice regardless of `line_count`.
   - `"I"` → narrator voice policy.
   - `user_locked=True` voice is never overwritten by re-suggest.
   - Engine switching re-suggests for new engine; existing engine assignments preserved.
   - Extras pool exhaustion: characters beyond pool size fall back to narrator.

3. **`test_cast_cli.py`**
   - `cast show` prints character table with verified flags.
   - `cast suggest` writes `cast.json`; does not approve.
   - `cast voice` sets voice, marks `user_locked`.
   - `cast merge` transactional: both `cast.json` and SQLite updated; merge logged.
   - `cast merge` with both sides `user_locked` → error, no mutation.
   - `cast undo` reverts last merge in both `cast.json` and SQLite.
   - `cast approve` stores hash; subsequent edit invalidates.
   - `cast approve --auto` sets unknown → narrator.
   - `cast rename` and `cast split` log to merge history for undo.

4. **`test_cast_review_interaction.py`**
   - `cast review` steps through unknown/low-confidence segments in order.
   - User input sets `speaker_id`, `source=user`.
   - Re-run does not overwrite user-set segments.

### Deferred (Step 4, pending approval): Multi-Voice Synthesis

5. `test_multivoice_pipeline.py`:
   - End-to-end multi-voice with FakeTTS: dialogue routed to character voice, narration to narrator.
   - Kill during synthesis (simulate crash at segment N); resume re-synthesizes only from N+1.
   - User correction surviving re-run: `source=user` segment never re-attributed.
   - Selective invalidation: voice change → only that character's segments become `PENDING`.
6. `test_voice_hash.py`: NUL-delimited hash stable across fields, changes when any field changes.
7. `test_cast_migration.py`: `cast.json` from schema `1.0` migrated to `1.1` without data loss; schema exported to `docs/schemas/cast.schema.json`.

---

## 10. Open Decisions — **RESOLVED**

| Decision | Choice |
|---|---|
| Minor character threshold | $K=3$ (named constant `MINOR_CHARACTER_THRESHOLD = 3`); extras pool only if `named_chars > available_voices` |
| First-person protagonist ("I") | Narrator voice by default; overridable via `cast voice` |
| Intra-sentence tag pause | 150 ms (`PAUSE_TAG_TRANSITION_MS`); 450 ms for speaker turns (`PAUSE_SPEAKER_TURN_MS`) |
| Catalogue metadata trust | All tags `verified: false`; caster uses only verified tags; distinctness-only fallback |
| Cache key | Single extended `voice_hash`: SHA-256(text ∥ `\x00` ∥ voice_id ∥ `\x00` ∥ engine ∥ `\x00` ∥ engine_version ∥ `\x00` ∥ normaliser_version) |
| `cast approve` integrity | SHA-256 of `cast.json` content stored; edit invalidates |
| Engine switch policy | Keep profile & user-locked assignments; re-suggest for new engine only |
| Co-occurrence scope | 20-segment sliding window (not chapter); reuses `attribution/profiler.py` |
| `cast merge` conflict | If both sides `user_locked` → abort with error |
