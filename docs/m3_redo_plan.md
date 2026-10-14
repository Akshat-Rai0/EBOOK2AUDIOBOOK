# M3 Redo Plan: Real Evaluation of Speaker Attribution

**Date:** 2026-10-14  
**Milestone:** M3 redo (real attribution evaluation)  
**Purpose:** Replace synthetic-gate accuracy figures with measurements on hand-labelled public-domain text

---

## 1. Dataset Proposal

### 1.1 Source Selection
Download 4–6 public-domain novels/stories from Project Gutenberg into `data/raw/` (gitignored). Selection criteria:
- **Variety of dialogue styles**:
  - Heavy dialogue tags in third person (e.g., 19th-century British novels)
  - First-person narrator (e.g., epistolary novels like *Dracula*)
  - Rapid two-person dialogue with few tags (e.g., fast-paced scenes)
  - Group scenes with 3+ speakers
- **Not famous classics only**: Mix of well-known and lesser-known works to test memorization
- **Chapter length**: 3,000–7,000 words per chapter (manageable labelling sessions)

### 1.2 Chapter Selection
- **Dev chapter**: 1 chapter for prompt tuning, few-shot examples, and variant selection
- **Test chapters**: 3 chapters for final evaluation (frozen configuration)
- **Name-swap chapter**: 1 of the test chapters, with all character names replaced consistently

### 1.3 Proposed Sources (for your approval)
I will download these from gutenberg.org after approval:
1. *The Strange Case of Dr. Jekyll and Mr. Hyde* (Robert Louis Stevenson) – 3rd person, moderate tags
2. *The Adventures of Sherlock Holmes* (Arthur Conan Doyle) – dialogue-heavy, 2-person scenes
3. *Dracula* (Bram Stoker) – epistolary, first-person, letters/diaries
4. *The Time Machine* (H.G. Wells) – 3rd person, some group scenes
5. *The War of the Worlds* (H.G. Wells) – 3rd person, dialogue with tags
6. *Short story collection* (e.g., Kipling or Poe) – rapid dialogue, varied styles

**Alternative approach**: Let me download a broader set (8–10 works) and propose specific chapters after inspecting their dialogue density.

---

## 2. Label Set and Policies

### 2.1 Label Categories
- **Character name**: Canonical speaker ID (e.g., "harry", "sherlock")
- **NARRATOR_I**: First-person narrator (the "I" character in epistolary text)
- **GROUP**: Multiple speakers ("they all shouted", "the crowd cheered")
- **UNKNOWN**: Genuinely undeterminable speaker (no clues, context insufficient)
- **NOT_SPEECH**: Scare quotes, titles, quoted words, thoughts, internal monologue

### 2.2 Labeling Policies
- **Unnamed speakers**: Use descriptive placeholder if unnamed ("the old man", "the stranger")
- **Speech inside quotes**: Label the actual speaker, not the narrator of the frame
- **Multi-paragraph quotes**: Same speaker across paragraphs (DialogueSegmenter already tracks this)
- **Letters and diaries**: Use NARRATOR_I for the author if first-person, or the character name if signed
- **Thoughts**: Mark as NOT_SPEECH (internal monologue), or use character name if explicitly thought-speech

### 2.3 Alias Clusters
- Each character entry includes: canonical name, aliases (e.g., "Sherlock Holmes" with "Holmes", "Sherlock")
- Tool supports adding aliases during labeling (press `n` for new character)
- Gold format stores alias clusters as a list per character

---

## 3. Labelling Tool Design

### 3.1 Terminal UI (macOS-compatible)
```
Previous paragraph:
[context before]

Quote [Q3]:
"Where were you last night?" he demanded.
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^

Next paragraph:
[context after]

Speaker options:
1. detective
2. suspect
3. narrator
4. NARRATOR_I
5. GROUP
6. UNKNOWN
7. NOT_SPEECH
n = new character with aliases
m = flag missed/wrong span
b = undo last label
s = skip (return later)
q = save and quit

> _
```

### 3.2 Features
- **Context window**: Shows previous paragraph, highlighted quote, next paragraph
- **Speaker list**: Numbered list with already-seen characters + special labels
- **Autosave**: After every label to `data/gold/<chapter>.json`
- **Resume**: Continues from last unlabeled quote on next run
- **Relabel mode**: `--relabel 0.15` presents random 15% of already-labeled quotes without showing prior labels
- **Quote alignment**: Uses production DialogueSegmenter to extract spans; tool reports misalignment
- **Progress summary**: Per-chapter statistics (total quotes, labeled, skipped, flagged)

### 3.3 Output Format
```json
{
  "chapter_id": "c03",
  "source_file": "dracula.txt",
  "source_hash": "sha256:...",
  "quotes": [
    {
      "quote_id": "Q1",
      "text": "...",
      "speaker": "jonathan_harker",
      "aliases": ["Jonathan", "Harker"],
      "label": "character",
      "flags": []
    }
  ],
  "metadata": {
    "labeled_at": "2026-10-14T...",
    "labeled_by": "human",
    "total_quotes": 42,
    "labeled": 40,
    "skipped": 2,
    "flagged": 0
  }
}
```

---

## 4. Quote Alignment

### 4.1 Alignment Method
- Run production DialogueSegmenter on the raw chapter text
- Extract quote spans with IDs [Q1], [Q2], etc.
- Compare gold labels to extractor spans
- Report:
  - **Precision**: How many extractor quotes match gold quotes
  - **Recall**: How many gold quotes the extractor found
  - **Missed quotes**: Gold quotes not in extractor output (flagged by user)
  - **Invented quotes**: Extractor quotes not in gold (flagged by user)

### 4.2 ID Mapping
- Gold labels use extractor IDs ([Q1], [Q2]) for direct comparison
- If extractor changes, re-run alignment and report differences
- Misalignment counts against the system (upstream error)

---

## 5. Metrics

### 5.1 Quote-Level Accuracy
- **Overall accuracy**: Correct / total quotes (UNKNOWN counted as wrong)
- **By style**: Accuracy per style category (tagged, pronoun-tagged, untagged, alternating, 3+ speakers)
- **GROUP/NOT_SPEECH**: Handled per your decision (see questions below), reported separately

### 5.2 Secondary Metrics
- **Unknown rate**: % of quotes labeled UNKNOWN by system
- **Fallback rate**: % of quotes where LLM failed and fell back to narrator
- **Confusion matrix**: Top 10 character pairs most often confused
- **Alias-merge quality**: Pairwise precision, recall, F1 over your alias clusters
- **JSON validity**: % of valid JSON before/after retries
- **Retries per 100 quotes**: Average retry count
- **Speed**: Seconds per 1,000 words, tokens per second (if available)
- **Memory**: Peak RAM of Ollama process, peak RAM of CastBook process
- **Disk size**: Model size on disk

### 5.3 Statistical Methods
- **Confidence intervals**: Bootstrap over paragraphs (quotes in one paragraph are correlated)
- **Per-chapter results**: Show each test chapter separately (don't hide weak chapters)
- **Your consistency**: Agreement % and Cohen's kappa on 15% relabel (caps measurable accuracy)

### 5.4 Unit Tests
- All metrics functions tested on toy data
- Bootstrap interval logic verified
- Confusion matrix correctness checked

---

## 6. Calibration

### 6.1 Accuracy by Confidence Bucket
- Bucket quotes by confidence: high (≥0.95), medium (0.70–0.95), low (<0.70)
- Report accuracy per bucket
- Identify if confidence scores are well-calibrated

### 6.2 Threshold Curve
- For thresholds from 0.40 to 0.95 in 0.05 steps:
  - Fraction of lines sent to review (confidence < threshold)
  - Precision of auto-accepted lines (confidence ≥ threshold)
- Plot or table showing trade-off

### 6.3 Proposed Threshold
- Choose threshold where auto-accepted lines reach target precision (e.g., 95%)
- Report: at threshold X, Y% of lines go to review, precision is Z%
- **Do NOT change config.py yet**—propose the value in the report

---

## 7. Evaluation Harness

### 7.1 Command Interface
**Option A**: New CLI command `castbook eval attribution`  
**Option B**: Standalone script `scripts/eval_attribution.py`

**Recommendation**: Option A (CLI command) for consistency with existing workflow.

### 7.2 Configuration
```json
{
  "run_id": "eval-2026-10-14-001",
  "model": "llama3.2:3b",
  "model_digest": "sha256:...",
  "quantisation": "Q4_K_M",
  "context_window": 4096,
  "previous_paragraphs": 3,
  "few_shot": true,
  "variant": "V3-production",
  "prompt_version": "2.0-m4-id-based",
  "temperature": 0.0,
  "seed": 42,
  "device": "cpu",
  "date": "2026-10-14T...",
  "commit_hash": "abc123"
}
```

### 7.3 Variants (Ablation Study)
- **V0**: Rules only (no LLM, no alternation)
- **V1**: LLM only (no rules, no alternation)
- **V2**: Rules + LLM (no alternation)
- **V3**: Production (rules + LLM + alternation)
- **V4**: Two-pass design (per-chapter cast discovery first, then attribution) — if feasible without changing production defaults

### 7.4 Context Sizes
- Test with 1, 3, and 6 previous paragraphs
- Measure impact on accuracy and speed

### 7.5 Few-Shot
- On/off comparison
- Examples from dev chapter only (never from test)

### 7.6 Output
- `results/<run-id>.jsonl` with one line per chapter
- Resumable (skip already-completed chapters)
- All metadata from hard rules recorded

---

## 8. Model Matrix

### 8.1 Baseline Models
- **llama3.2:1b** (Q4_K_M, ~1.3 GB)
- **llama3.2:3b** (Q4_K_M, ~2.0 GB)

### 8.2 Additional Models (to be proposed after checking Ollama library)
- Up to 3 current small-to-mid instruct models (3B–9B class at 4-bit)
- Must verify licences before pulling
- Check your existing models first

### 8.3 Per-Model Measurements
- JSON compliance rate
- Accuracy per variant
- Speed (sec/1000 words)
- Peak RAM
- Disk size

### 8.4 Model Selection Protocol
1. Check which models you already have installed
2. Query Ollama library for available 3B–9B instruct models
3. Verify licences (prefer Apache-2.0, MIT, BSD; avoid non-commercial if possible)
4. Propose final list with sizes and licences
5. Ask before pulling anything over 100 MB

---

## 9. Name-Swap Test

### 9.1 Method
- Take one test chapter
- Replace every character name and alias with fresh names consistently
  - e.g., "Harry" → "Xavier", "Hermione" → "Yara", "Ron" → "Zack"
- Update gold labels by the same mapping
- Run best configuration on both original and swapped versions

### 9.2 Interpretation
- **Large drop in accuracy**: Model was recalling the book, not reading it
- **Small/no drop**: Model is actually analyzing the text
- Report delta and interpretation

---

## 10. Run Protocol

### 10.1 Dev Phase
1. Label dev chapter
2. Run all variants and models on dev chapter
3. Choose best prompt, context size, few-shot strategy
4. Freeze one configuration per model and variant

### 10.2 Test Phase
1. Label 3 test chapters
2. Run each frozen configuration on each test chapter once
3. Record every run (including bad ones) in results/
4. Never edit a prompt after seeing test results (new prompt = new frozen version)

### 10.3 Determinism
- Temperature 0
- Fixed seed (42)
- Record Ollama digest, context window, keep_alive setting
- Record runtime version

---

## 11. Error Analysis

### 11.1 Selection
- Take 20 worst errors for top 2 configurations
- "Worst" = confident but wrong (high confidence, incorrect attribution)

### 11.2 Categories
- Wrong nearest name (attributed to closest character, not actual speaker)
- Pronoun chain (he/she/they confusion)
- Missing tag (no dialogue tag available)
- New character (first appearance without context)
- Alias collision (two characters with similar names)
- Quote inside a quote
- Two speakers in one paragraph
- Extractor error (span misalignment)
- My label doubtful (human error)
- Other (with explanation)

### 11.3 Reporting
- Short excerpts only (not full paragraphs)
- Count per category
- Show examples per category

---

## 12. Report Contents

### 12.1 Results Table
Per model and variant:
- Overall accuracy with 95% CI
- Accuracy by style category with 95% CI
- Unknown rate
- JSON validity (%)
- Retries per 100 quotes
- Speed (sec/1000 words)
- Peak RAM (MB)
- Disk size (GB)

### 12.2 Ablation
- What rules alone contribute
- What LLM adds
- What alternation adds
- Two-pass design impact (if implemented)

### 12.3 Calibration
- Accuracy per confidence bucket
- Threshold curve (review % vs precision)
- Proposed threshold with reasoning
- Share of lines that would go to review at proposed threshold

### 12.4 Additional Results
- Alias-merge results (precision, recall, F1)
- Name-swap test result (delta and interpretation)
- Extractor precision and recall
- Your labelling consistency (agreement %, Cohen's kappa)

### 12.5 Error Analysis
- Category breakdown with counts
- Examples per category
- Patterns identified

### 12.6 Recommendations
- **Go/no-go per hardware tier**:
  - Multi-voice tier (≥8 GB RAM): which model/variant?
  - Narrator-only tier (<8 GB RAM): is LLM feasible?
- **Recommended default**: Model, variant, context size, few-shot
- **Fallback option**: If recommended not available, what to use?
- **Measured RAM floor**: Replace 8 GB placeholder with actual measurement
- **Speed benchmark**: Minutes per 5,000-word chapter on your machine

### 12.7 Doc Changes Needed
- Update hardware floor in project docs
- Update success criteria (real accuracy target)
- Update review threshold constant in config.py
- Update risks list (add any new risks discovered)
- Update panel Q&A status table (M3 evaluation: DONE)

---

## 13. Pass/Fail Gates

### 13.1 Primary Gate
- Test-set accuracy ≥ 90% for recommended configuration (UNKNOWN counted as wrong)
- No style category below 80% (if below, report as known weakness, don't hide)

### 13.2 Secondary Gates
- JSON validity ≥ 99% after at most 2 retries
- Time per 5,000-word chapter within your time limit (see question below)
- Measured RAM floor supports claimed hardware tiers

### 13.3 Failure Mode
If primary gate missed:
- Report what each lever bought (bigger model, two-pass, more context, hybrid weighting, few-shot)
- Give 3 options:
  1. Accept lower accuracy with higher review load
  2. Change promise to "auto-attribute plus review screen" with measured review %
  3. Invest in additional improvements (specify which)

---

## 14. Questions for You

### Q1: Time Gate
**Maximum minutes to attribute one 5,000-word chapter on your machine** (Apple Silicon, 16 GB RAM)?

| Option | Time | Reasoning |
|--------|------|-----------|
| A | 5 minutes | Aggressive; tests if we can handle full novels in ~1 hour |
| B | 10 minutes | Moderate; balances speed with quality (current LLM is ~1.8s/para) |
| C | 15 minutes | Conservative; allows for larger models if needed |
| D | 20 minutes | Very conservative; prioritizes accuracy over speed |

**Recommendation**: B (10 minutes). A 5,000-word chapter has ~50–100 paragraphs. At 1.8s/para (llama3.2:3b), that's 90–180 seconds baseline. 10 minutes leaves headroom for context and larger models.

---

### Q2: Scoring GROUP, UNKNOWN, and NOT_SPEECH
**How should these special labels be scored in accuracy calculations?**

| Option | Method | Reasoning |
|--------|--------|-----------|
| A | Count all as wrong (like any incorrect attribution) | Strict; ensures system must get speakers right |
| B | Exclude GROUP and NOT_SPEECH from accuracy denominator | They're not "wrong speaker" cases; measure attribution on genuine speech only |
| C | Score GROUP as correct if any named speaker in group, NOT_SPEECH as always correct | More lenient; reflects that these are edge cases |
| D | Report overall accuracy (all wrong) AND separate "speech-only accuracy" (exclude GROUP/NOT_SPEECH) | Best of both; shows both strict and per-speech metrics |

**Recommendation**: D. Report both metrics: strict overall accuracy (for go/no-go gate) and speech-only accuracy (for understanding attribution quality on genuine dialogue). UNKNOWN always counts as wrong in both.

---

### Q3: First-Person "I" Policy
**How should we handle "I" in first-person text, letters, diaries, and thoughts?**

| Option | Policy | Reasoning |
|--------|--------|-----------|
| A | Always label "I" as NARRATOR_I (current default) | Simple; treats first-person narrator as distinct from characters |
| B | Label "I" as the protagonist character when it's clearly the protagonist speaking | More nuanced; distinguishes narrator voice from character voice in dialogue |
| C | Use NARRATOR_I for narration/letters/diaries, but use character name for "I" in dialogue | Hybrid; keeps epistolary narration separate but tracks character speech |
| D | Make this a per-book decision (configure in gold manifest) | Most flexible; different books have different conventions |

**Recommendation**: A for now (current default). First-person attribution is a separate hard problem (epistolary novels are structurally different). If evaluation shows this causes errors, we can revisit with option C as a refinement. Mark first-person chapters as their own style category in reporting.

---

## 15. Next Steps (After Approval)

1. Download public-domain texts to `data/raw/`
2. Build labelling guide (`docs/labelling_guide.md`)
3. Implement quote alignment verification
4. Build gold-label tool (terminal UI)
5. Implement automatic style tagging
6. Build metrics module with unit tests
7. Implement calibration logic
8. Build evaluation harness (CLI command)
9. Query Ollama for available models
10. You label dev chapter
11. Run dev-phase evaluations
12. Freeze configurations
13. You label test chapters (including 15% relabel)
14. Run test-phase evaluations
15. Perform name-swap test
16. Error analysis
17. Write final report (`docs/spikes/attribution_report_v2.md`)
18. Mark old report as superseded

---

## 16. Working Agreements

- Follow AGENTS.md (ask, don't assume; small commits; clear code; record decisions)
- Run `uv run ruff check . && uv run ruff format --check . && uv run pytest -q` before each commit
- Explain new concepts (bootstrap, Cohen's kappa, pairwise F1) in 2 lines with everyday analogy
- Log non-obvious choices in docs/DECISIONS.md with real date
- Public-domain text only; raw text in gitignored `data/raw/`, gold labels committed
- Deterministic runs (temp 0, seed 42, record all metadata)
- Ask before downloading >100 MB models
- Never let me label the gold set; I build the tool only
- Dev/test split strict; no test-set overfitting
- End session with: what changed, tests run, what's next, open questions
