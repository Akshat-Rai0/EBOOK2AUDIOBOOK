# M4 Architecture & Implementation Plan: Casting and Multi-Voice Synthesis

**Milestone:** M4 (`m4-casting-multivoice`)  
**Date:** 2026-10-01 (Updated 2026-10-09)  
**Authors:** Akshat Rai, Divyanshu Bhusan  
**Status:** IN PROGRESS (V2 Attribution, Character Registry, Dialogue Segmenter & Merged M2b TTS Implemented; Voice Catalogue & Review Gate Pending)

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
1. **Linear Progression:** Stages execute strictly in sequence: `ingest` $\rightarrow$ `chunk` $\rightarrow$ `attribute` $\rightarrow$ `cast` $\rightarrow$ `review_gate` $\rightarrow$ `synthesize` $\rightarrow$ `assemble` $\rightarrow$ `export`.
2. **Review Gate Pause:** Upon completing the `cast` stage, the pipeline pauses with status `PENDING_REVIEW` unless invoked with `--auto`. The user reviews unassigned or low-confidence lines (`castbook cast review`) and approves (`castbook cast approve`).
3. **Sequential Model Lifetime (Memory Guard):**
   - In `attribute`: The LLM (`llama3.2:3b` in Ollama) runs. Upon stage completion, the orchestrator explicitly stops/unloads the model from Ollama memory (`ollama stop` / `keep_alive: 0`).
   - `ModelManager` verifies active RAM before loading any TTS engine (XTTS-v2 or VITS). The LLM and TTS models are never held in memory simultaneously.
4. **Selective Re-Synthesis:** Re-running after manual edits (`castbook cast voice`, `castbook cast merge`) invalidates only the segments whose text, speaker, or assigned voice changed.

---

## 2. Schema Changes & Migrations

### 2.1 Segment Model Update (`ebook2audiobook/models/book.py`)
```python
class SegmentSource(str, Enum):
    LLM = "llm"
    RULE = "rule"
    USER = "user"  # Authoritative: never overwritten by automatic re-runs


class SegmentKind(str, Enum):
    NARRATION = "narration"
    DIALOGUE = "dialogue"
    THOUGHT = "thought"


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
    voice_hash: str | None = None  # Hash of text + voice_id for cache validation
    status: SegmentStatus = SegmentStatus.PENDING
```

### 2.2 Character Model Update (`ebook2audiobook/models/cast.py`)
```python
class CharacterProfile(BaseModel):
    gender: str = "unknown"  # "male", "female", "nonbinary", "unknown"
    gender_confidence: float = 0.0
    age_bracket: str = "unknown"  # "child", "young_adult", "adult", "elderly", "unknown"
    age_confidence: float = 0.0


class Character(BaseModel):
    id: str  # Canonical identifier (slugified name)
    display_name: str  # User-facing canonical name
    aliases: list[str] = Field(default_factory=list)
    profile: CharacterProfile = Field(default_factory=CharacterProfile)
    line_count: int = 0
    first_chapter: int = 0
    user_locked: bool = False  # If True, re-runs will never re-cast or merge
    voice_assignments: dict[str, str] = Field(default_factory=dict)  # {engine_name: voice_id}
```

### 2.3 SQLite Schema Migration (`projects/<name>/state.sqlite`)
The `segments` table adds columns:
- `speaker_id TEXT DEFAULT 'narrator'`
- `kind TEXT DEFAULT 'narration'`
- `confidence REAL DEFAULT 1.0`
- `source TEXT DEFAULT 'rule'`
- `evidence TEXT`
- `voice_hash TEXT`

A migration runner executes `ALTER TABLE` statements conditionally if the columns are missing, ensuring backward compatibility with existing M2 SQLite databases. Re-export JSON schemas to `docs/schemas/`.

---

## 3. Segmenter Update (Quote / Tag Alternation)

### Alternating Span Splitting Rule
Paragraphs containing mixed dialogue and narration are parsed into alternating spans:
- `"You lied," she said, "and you know it."` $\rightarrow$
  1. `c01-p002-s01`: `"You lied,"` (kind: `dialogue`, speaker: character)
  2. `c01-p002-s02`: `she said,` (kind: `narration`, speaker: `narrator`)
  3. `c01-p002-s03`: `"and you know it."` (kind: `dialogue`, speaker: character)

### Stable Identifier Scheme
- Segment IDs preserve the M2 hierarchy: `c<CC>-p<PPP>-s<SS>`.
- `<SS>` is 1-indexed and orders spans consecutively within the paragraph.
- Idempotency: Segment IDs are deterministic. Re-segmenting generates identical IDs.

---

## 4. Production Attribution Stage & Character Registry

### 4.1 Production Attribution
- Uses `OllamaAttributor` calibrated in M3 with `llama3.2:3b`.
- Per-chunk processing with running canonical cast list and immediate previous-paragraph context.
- Constrained JSON output format enforced at API level.
- Maximum 2 retries on malformed output, then fallback to `speaker="unknown"` with confidence `0.0`.
- Stage persistence records `model_tag`, `prompt_version`, and `git_commit` in `stages` table.

### 4.2 Alias Merge & Character Registry
- `CharacterRegistry` resolves raw strings to canonical names.
- Token-intersection matches honorifics and surnames (`"Ms. Vane"` $\rightarrow$ `"Mira Vane"`).
- Ambiguous tokens (e.g. shared surnames "John Smith" vs "Jane Smith", titles "the Captain") are not merged; they are preserved as distinct characters and flagged for review.
- Merge operations record an audit log in `projects/<name>/merge_history.json` supporting `castbook cast undo`.

### 4.3 Profile Inference
- Rules extract stated descriptors (pronouns: he/she/they $\rightarrow$ gender; modifiers: "old", "young", "boy", "grandmother" $\rightarrow$ age bracket).
- Output is always advisory: `confidence < 1.0`, defaults to `"unknown"` when text is silent.

---

## 5. Voice Catalogue & Casting Algorithm

### 5.1 Voice Catalogue (`voices.json` per engine)
- Stored in `ebook2audiobook/audio/catalogue/`:
  - `xtts_voices.json`: Built-in XTTS-v2 reference speakers (male/female, accents, licensed for non-commercial research).
  - `vits_voices.json`: 109 VCTK speakers (MIT license, gender/accent tags).
- Reserved voice: One clear voice per engine is permanently reserved for `narrator` and excluded from character assignment.
- Commands: `castbook voices list --engine <xtts|vits>`, `castbook voices sample <voice-id>`.

### 5.2 Casting Algorithm
1. **Rank by Frequency:** Characters sorted by `line_count` descending.
2. **Profile Match:** Filter catalogue for matching `(gender, age_bracket)`. If unknown or pool exhausted, widen filter.
3. **Co-occurrence Distinctness:**
   - Compute chapter co-occurrence matrix $C_{ij}$ (number of chapters characters $i$ and $j$ both speak in).
   - Penalize voices with similar acoustic timbre/pitch on characters with high co-occurrence.
4. **Minor Character Extras Pool:**
   - Characters with fewer than $K$ lines (configurable, default: 3 lines) do not receive unique voices; they pull from a 2-voice "extras pool" or fall back to the narrator.
5. **Collective Speakers:** Phrases like `"they all cheered"` automatically assign to `narrator`.
6. **Engine Isolation:** Voice assignments are keyed by `(character_id, engine)`. Switching engines keeps character profiles and re-runs the recommendation.

---

## 6. Review Commands (CLI Interface)

All commands wrap pure library functions from `ebook2audiobook.casting`:
- `castbook cast show --project <p>`: Print table of characters, line counts, profiles, assigned voices, and unknown line count.
- `castbook cast review --project <p>`: Interactive terminal step-through of all `unknown` or low-confidence ($< 0.80$) segments with 2 lines of preceding context.
- `castbook cast merge <p> <alias> <target>`: Merge two characters, update segments, log to merge history.
- `castbook cast split <p> <character> <segment-id>`: Detach a segment from a merged character.
- `castbook cast rename <p> <character> <new-name>`: Rename character display name.
- `castbook cast voice <p> <character> <voice-id>`: Assign specific voice; marks character as `user_locked`.
- `castbook cast sample <p> <character>`: Synthesize a 3-second real sample using their assigned voice.
- `castbook cast approve --project <p>`: Mark cast approved, allowing synthesis to proceed.
- `castbook cast undo --project <p>`: Revert the last merge or rename from `merge_history.json`.

---

## 7. Multi-Voice Synthesis, Assembly, and QA

### 7.1 Multi-Voice Synthesis & Selective Invalidation
- Each segment computes `voice_hash = SHA256(text + voice_id)`.
- Re-synthesis query: `SELECT * FROM segments WHERE status != 'done' OR voice_hash != :new_hash`.
- User-edited segments (`source = 'user'`) are never re-attributed, but are re-synthesized if their assigned character's voice changes.
- Speaker conditioning latents (for XTTS-v2) are computed once per active voice and held in memory during the synthesis run.

### 7.2 Assembly & Pauses
- Seamless concatenations use tuned silence insertions:
  - Dialogue to tag (or tag to dialogue within same sentence): **150 ms**.
  - Speaker turn transition: **450 ms**.
  - Paragraph boundary: **750 ms**.
  - Chapter boundary: **1500 ms**.
- Loudness normalization: Peak-scaling with EBU R128 loudness match target (-18 dBFS) across alternating speakers to eliminate volume jumps.

### 7.3 Multi-Voice Quality Assurance (QA)
- Check 1: Verify 100% of segments have a valid voice assigned in catalogue.
- Check 2: Report count and percentage of `unknown` speaker lines.
- Check 3: Consistency assert: ensure no character has multiple discordant voices in the same output.
- Check 4: Utterance length guard: Utterances $< 4$ characters (e.g. "Oh.", "No.") padded with 50 ms trailing silence to prevent TTS truncation/glitches.

---

## 8. Verification of Caveats (Section 5)

| # | Caveat | Severity | Mitigation in M4 Design |
|---|---|---|---|
| 1 | **Voice pool exhaustion** (e.g. 20+ characters vs limited catalogue) | Medium | Minor characters (< 3 lines) routed to a rotating 2-voice extras pool or narrator; main characters prioritized by line count. |
| 2 | **Engine-specific voices** (XTTS voices $\ne$ VITS voices) | High | `Cast` stores assignments per engine: `dict[engine, voice_id]`. Switching engine re-suggests without erasing profiles. |
| 3 | **Gender/age misclassification** | Medium | Profile inference outputs confidence and accepts `"unknown"`. CLI provides single-command override; never presented as unquestionable fact. |
| 4 | **Short utterances synthesis degradation** ("No.", "Why?") | High | Pad short segments (< 15 chars) with micro-silence (50 ms); test on FakeTTS and engine adapters. |
| 5 | **Choppy sentence flow from tag splitting** | Medium | Calibrated 150 ms intra-sentence tag pause; cross-fade join to eliminate audio clicks. |
| 6 | **Alias collisions** (same surname, shared titles) | High | Conservative matching: never merge solely on common surname or title; flag for manual review. |
| 7 | **First-person narratives ("I")** | Medium | Explicit narrator policy: If narrator is identified as protagonist, narrator voice is assigned to character "I" with option to decouple. |
| 8 | **Thoughts & epistolary passages (letters/italics)** | Low | Default `thought` kind to narrator voice or character internal voice; plain text defaults to narrator. |
| 9 | **Cascade invalidation from cast changes** | High | Only segments whose `(text, voice_id)` hash altered are marked `PENDING`; all untouched WAVs remain cached. |
| 10 | **XTTS speaker latent memory leaks** | Medium | Latents cached in dict bounded by unique cast size ($\le 30$ active voices $\approx$ negligible ~15 MB RAM). |
| 11 | **Licensing restrictions** (XTTS non-commercial, VCTK attribution) | High | All catalogue voices annotated with license in `voices.json` and tracked in `docs/LICENSES.md`. |
| 12 | **Long CPU attribution runtime** | Medium | Progress bars, per-chapter timing, and SQLite segment checkpointing so crashes lose $< 1$ paragraph. |

---

## 9. Test Plan

All tests execute offline with `FakeTTS` and a deterministic `FakeAttributor`:
1. `test_multivoice_schema_migration`: Verifies database migration from M2 schema to M4 schema without data loss.
2. `test_segmenter_alternating_spans`: Verifies `"Quote," tag, "quote."` splits into 3 segments with accurate kind and speaker tags.
3. `test_character_registry_alias_merge_and_undo`: Verifies merge history and undo capability.
4. `test_casting_cooccurrence_distinctness`: Verifies two co-occurring characters receive different voices.
5. `test_selective_resynthesis_invalidation`: Verifies changing one character's voice only re-synthesizes that character's segments.
6. `test_user_locked_authoritative`: Verifies `source=user` and `user_locked=True` are never overwritten on re-run.
7. `test_memory_separation_assert`: Asserts LLM client is unloaded before TTS engine initializes.
8. `test_narrator_hardware_tier_refusal`: Verifies `< 8 GB` RAM system cleanly rejects multi-voice mode with helpful diagnostic.

---

## 10. Open Decisions / Questions

Before proceeding to code, please confirm your preference on these 3 design choices:

1. **Minor Character Threshold ($K$ lines):**
   - *(Recommended) Option A:* Set $K=3$. Characters with $\le 3$ lines share a generic 2-voice extras pool or narrator.
   - *Option B:* Set $K=1$. Every character who speaks even twice gets their own voice until catalogue is exhausted.
   - *Option C:* Prompt user interactively when catalogue voices run low.

2. **First-Person Protagonist Policy ("I"):**
   - *(Recommended) Option A:* By default, character "I" uses the Narrator voice unless explicitly assigned another voice by the user in `cast review`.
   - *Option B:* Always treat "I" as a distinct speaking character requiring a separate voice from the Narrator.

3. **Intra-Sentence Tag Pause Duration:**
   - *(Recommended) Option A:* 150 ms pause between dialogue and its narration tag (e.g., `"Stop," [150ms] she cried`).
   - *Option B:* 250 ms pause (more pronounced pause).
   - *Option C:* 0 ms (seamless join with 10 ms cross-fade).
