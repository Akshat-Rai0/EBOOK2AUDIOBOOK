# Spike Report: Speaker Attribution with Local LLM (M3)

**Date:** 2026-10-01  
**Milestone:** M3 (`m3-attribution-spike`)  
**Authors:** Akshat Rai, Divyanshu Bhusan  

---

## 1. Executive Summary

This spike evaluates local LLM performance for literary speaker attribution to support multi-voice audiobook synthesis. We tested local models against the project criteria: $\ge 90\%$ attribution accuracy, structured output compliance (JSON mode), execution within memory bounds without concurrent TTS collision, and robust alias merging.

**Key Decisions & Calibration:**
- **Recommended Model:** `llama3.2:3b` (4-bit Q4_K_M quantization, ~2.0 GB on disk, ~2.8 GB peak RAM).
- **Prompt & Pipeline Variant:** Contextual dialogue span extraction with running cast list and single previous-paragraph context. Ollama native `format: "json"` constrained decoding.
- **Hardware Floor:**
  - **Multi-voice Tier ($\ge 8$ GB RAM):** GO. `llama3.2:3b` comfortably runs on CPU/MPS with ~1.5–2.5s per paragraph inference.
  - **Narrator-only Tier (< 8 GB RAM / 4 GB Floor):** NO-GO for local LLM attribution. Systems with < 8 GB RAM must bypass the LLM stage and use narrator-only synthesis or pre-attributed files.
- **Confidence & Review Threshold:**
  - Any span with confidence $< 0.80$, any fallback span, and any span labeled `unknown` is routed to the human review gate (`castbook cast review`).

---

## 2. Model Evaluation & Hardware Spike

### 2.1 Evaluated Candidates on Host Hardware (16 GB Unified RAM, Apple Silicon)

| Candidate | Size | Peak Memory | Avg Time / Para | JSON Compliance | Accuracy (Synthetic Gate) | Verdict |
|---|---|---|---|---|---|---|
| `gemma4:latest` (8B) | 9.6 GB | ~10.2 GB | ~9.0s | Partial (emitted `"span"`, `"narration"` as speaker) | 75% (bleed into dialogue) | **Rejected** (heavy memory footprint leaves insufficient RAM for TTS; schema drift) |
| `llama3.2:3b` | 2.0 GB | ~2.8 GB | ~1.8s | Strict (exact schema match with `format: "json"`) | **93.5%** | **Recommended** (fast, strictly adheres to schema, leaves 13+ GB free for OS and TTS) |
| `llama3.2:1b` | 1.3 GB | ~1.9 GB | ~0.9s | Good | 84.0% (struggled with multi-speaker banter) | **Fallback** (under 90% accuracy target) |

### 2.2 Memory Safety Guard
Per architectural invariant 7.4 ("Never hold the LLM and a TTS model in memory together"):
- The LLM runs in its own stage.
- Upon completion of `attribute`, the LLM model is explicitly evicted/unloaded from Ollama (`POST /api/generate` with `keep_alive: 0` or CLI stop) before any TTS model (XTTS-v2 or VITS) is initialized.
- `ModelManager` verifies active memory allocation prior to loading speech synthesis checkpoints.

---

## 3. Pipeline & Prompt Recipe

### 3.1 Prompt Template
The system prompt enforces strict span decomposition:
- `kind`: one of `"dialogue"`, `"narration"`, `"thought"`.
- `speaker`: canonical character name from cast, `"narrator"`, or `"unknown"`.
- Exact verbatim span slicing (no rewriting or paraphrasing).

### 3.2 Constrained JSON Decoding
Ollama's `format: "json"` ensures token-level grammar enforcement. The parser validates keys (`text`, `speaker`, `kind`) and retries up to twice with exponential backoff on failure before falling back to `speaker="unknown"` / `narrator`.

### 3.3 Alias Merging & Character Registry
- Resolved aliases merge into the most specific (longest) canonical name.
- Token-intersection prevents merging distinct names while catching common prefixes (`"Ms. Vane"` $\rightarrow$ `"Mira Vane"`).
- Ambiguous tokens and distinct surnames remain unmerged and are flagged for human review.

---

## 4. Calibration & Gate Criteria for M4

1. **Model:** `llama3.2:3b` via Ollama.
2. **Review Gate Trigger:** Spans with confidence $< 0.80$, unknown speakers, or unmerged name collisions pause pipeline execution for user approval (`castbook cast review` / `castbook cast approve`).
3. **Execution Guard:** Verify host RAM $\ge 8$ GB before launching multi-voice attribution. Refuse multi-voice on narrator-only hardware tier with a clear diagnostic message.
