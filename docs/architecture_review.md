# Architecture Review — CastBook (Ebook-to-Audiobook)
*Prepared: 2026-09-30 | Status: Pre-approval, no code written*

---

## 1. Scope Restatement (≤ 10 lines)

Build a **local, offline** Python application called **CastBook** that converts EPUB / TXT / DOCX / PDF (text layer) ebooks into chapter-wise MP3 files plus one M4B audiobook.  
The narrator and every named character get a **separate, consistent voice** chosen by a local LLM performing speaker attribution.  
Phase A (Project-I, this session: M0 + M1 only): English; XTTS-v2 and VITS TTS engines; FastAPI + React UI; CLI; SQLite job state; pydantic data contracts; FakeTTS adapter for CI.  
**Out of scope now:** OCR, MOBI/AZW3/FB2, non-English TTS, voice cloning, cloud services, LangGraph (evaluated Week 5), SML tags.

---

## 2. Files Found in /docs

| File | Status |
|---|---|
| `docs/` directory | **Missing — directory does not exist yet** |
| `Ebook_to_Audiobook_Project_Document.docx` | Not present; content provided inline in this session |
| `project_difination.html` | Not present |
| `how_it_should_work.html` | Not present |
| `system-wiring.html` | Not present |
| `UI.html` | Not present |
| `roadmap.html` | Not present |

> **Note:** The project directory at `/Users/bipinkumarrai/Desktop/EVERYTHING/CODING PROJECTS/EBOOK2AUDIOBOOK` is currently **empty**. All six source documents must be placed in `docs/` before M0 begins. The `.docx` is authoritative over the five HTML files where they conflict.

---

## 3. Architecture Review

### 3.1 Proposed Repository Tree

```
castbook/                          ← project root
│
├── castbook/                      ← core library (importable package)
│   ├── __init__.py
│   ├── models/                    ← pydantic data models + JSON schemas
│   │   ├── book.py                # Book, Chapter, Paragraph
│   │   ├── segment.py             # Segment, SegmentKind, SegmentStatus
│   │   ├── cast.py                # Character, Cast, VoiceRef
│   │   └── job.py                 # Job, StageStatus
│   │
│   ├── ingest/                    ← Stage 2–4: Extraction + NLP preprocessing
│   │   ├── __init__.py
│   │   ├── extractor.py           # Abstract Extractor interface
│   │   ├── txt.py                 # TXT extractor
│   │   ├── docx.py                # DOCX extractor (heading styles)
│   │   ├── epub.py                # EPUB extractor (spine + nav)
│   │   ├── pdf.py                 # PDF extractor (text layer; rejects scans)
│   │   └── cleaning.py            # Unicode norm, quote norm, de-hyphenation, etc.
│   │
│   ├── chunker/                   ← Stage 5: Structure analysis
│   │   ├── __init__.py
│   │   ├── chapter_detector.py    # Heuristics + confidence score; override hook
│   │   └── sentence_splitter.py   # spaCy sentence/paragraph split
│   │
│   ├── attribution/               ← Stage 6–7: Language + Speaker/Dialogue
│   │   ├── __init__.py
│   │   ├── attributor.py          # Abstract Attributor interface
│   │   ├── llm_attributor.py      # llama.cpp wrapper, JSON grammar, retry, fallback
│   │   └── language_detector.py   # langdetect / fasttext
│   │
│   ├── casting/                   ← Stage 7: Voice selection
│   │   ├── __init__.py
│   │   └── caster.py              # Character registry, alias merge, voice assignment
│   │
│   ├── tts/                       ← Stage 8: TTS synthesis
│   │   ├── __init__.py
│   │   ├── engine.py              # Abstract TTSEngine interface
│   │   ├── fake_tts.py            # FakeTTS (silence/tones) for CI
│   │   ├── xtts_adapter.py        # XTTS-v2 adapter (stub for now)
│   │   └── vits_adapter.py        # VITS multi-speaker adapter (stub for now)
│   │
│   ├── audio/                     ← Stage 9–11: Processing + QA + output
│   │   ├── __init__.py
│   │   ├── processor.py           # Join, pause insertion, loudness normalisation
│   │   ├── quality.py             # QA: silence detection, missing segments
│   │   └── exporter.py            # Write chapter MP3s, build M4B with FFmpeg
│   │
│   ├── orchestrator/              ← Agent coordination (plain state machine)
│   │   ├── __init__.py
│   │   ├── state_machine.py       # Stage graph, transitions, resume logic
│   │   └── job_runner.py          # Runs one job end-to-end
│   │
│   ├── store/                     ← SQLite persistence
│   │   ├── __init__.py
│   │   └── db.py                  # Job, stage status, cast read/write
│   │
│   └── models_manager/            ← ModelManager stub (load/unload one model at a time)
│       ├── __init__.py
│       └── manager.py
│
├── cli/                           ← Thin Click wrapper over castbook/
│   ├── __init__.py
│   └── main.py                    # castbook --help, castbook doctor, castbook ingest
│
├── apps/
│   ├── api/                       ← FastAPI backend (empty scaffold, M5)
│   │   └── __init__.py
│   └── web/                       ← React frontend (empty folder, M5)
│       └── .gitkeep
│
├── tests/                         ← pytest suite
│   ├── conftest.py                # Shared fixtures, synthetic file builders
│   ├── test_models.py
│   ├── test_ingest_txt.py
│   ├── test_ingest_docx.py
│   ├── test_ingest_epub.py
│   ├── test_ingest_pdf.py
│   ├── test_cleaning.py
│   ├── test_chapter_detector.py
│   ├── test_fake_tts.py
│   └── test_doctor.py
│
├── docs/                          ← Source docs + generated docs
│   ├── Ebook_to_Audiobook_Project_Document.docx   ← authoritative (needs adding)
│   ├── project_difination.html
│   ├── how_it_should_work.html
│   ├── system-wiring.html
│   ├── UI.html
│   ├── roadmap.html
│   ├── architecture_review.md     ← this file
│   ├── DECISIONS.md
│   ├── LICENSES.md
│   ├── PARKING_LOT.md
│   └── schemas/                   ← JSON schemas auto-exported from pydantic models
│       ├── book.schema.json
│       ├── segment.schema.json
│       ├── cast.schema.json
│       └── job.schema.json
│
├── projects/                      ← Runtime: one sub-folder per book job (gitignored)
│   └── .gitkeep
│
├── AGENTS.md                      ← Working agreements (both team and AI)
├── CLAUDE.md                      ← imports AGENTS.md (for Claude Code)
├── README.md
├── pyproject.toml                 ← uv-managed; ruff, pytest config
├── uv.lock
└── .github/
    └── workflows/
        └── ci.yml                 # lint + tests, no network
```

---

### 3.2 Architecture Caveats — Findings & Severities

| # | Caveat | Finding | Severity | Mitigation |
|---|---|---|---|---|
| 1 | **Multi-speaker VITS** | Standard VITS is single-speaker. `VITS-vctk` (VCTK 109-speaker checkpoint, MIT licence, ~350 MB) is the best permissive option. Coqui TTS ships a ready adapter. Must verify CPU real-time factor on target hardware in Week 6. | **High** | Use `coqui-tts` VCTK/VITS model; document speaker mapping in cast.json; benchmark in Week 6. |
| 2 | **XTTS-v2 text limit** | Coqui XTTS-v2 truncates silently beyond ~250 chars. Must split at sentence boundaries using `max_chars` from the adapter before calling `synthesize()`. Speaker conditioning (`get_conditioning_latents`) is expensive; cache per character in a latents dict keyed by `character.name`. | **High** | Implement `split_to_chunks(text, max_chars)` in the adapter; cache latents in `ModelManager`. |
| 3 | **LLM JSON reliability** | Small 3B models hallucinate outside JSON schemas. `llama.cpp` supports grammar-constrained generation (`-gbnf`); Ollama supports JSON mode. Plan: GBNF grammar for the attribution schema → retry up to 2× → narrator fallback. Log every fallback. | **High** | GBNF grammar file in `attribution/grammar.gbnf`; retry wrapper; fallback documented. |
| 4 | **Dialogue detection / quote styles** | Standard `" "` and `' '` are well-handled by spaCy. Guillemets (`« »`) and em-dash dialogue (common in Eastern European / translated fiction) require separate regex heuristics. Nested quotes need stack-based parsing. | **Medium** | Handle `" " ' '` and `« »` in Phase A; document em-dash as known limitation in PARKING_LOT.md. |
| 5 | **Chapter detection by format** | EPUB: reliable (nav + spine). DOCX: reliable (Heading 1/2 styles). TXT: heuristic (ALL CAPS lines, "Chapter N" regex, blank-line ratio). PDF: heuristic + font-size metadata from PyMuPDF. All output a `confidence` float (0–1); `chapters_override.json` overrides low-confidence results. | **Medium** | Per-format strategy in `chapter_detector.py`; override hook in M1. |
| 6 | **PDF text extraction & licence** | `PyMuPDF` (MuPDF) is **AGPL-3.0**. `pdfminer.six` is MIT but has no font-size access. `pypdf` is BSD-3 but text extraction is weaker on complex layouts. **Recommendation: use `pdfplumber` (MIT, wraps pdfminer) for text + `pypdf` for metadata; avoid PyMuPDF unless AGPL is acceptable.** Scanned PDFs (no text layer): detect via character-count threshold and reject with a clear error message. | **High** | Use `pdfplumber` (MIT); record decision in DECISIONS.md; scanned-PDF rejection in M1. |
| 7 | **M4B chapter markers** | FFmpeg `ffmetadata` format supports chapter markers; tested as working in VLC, Apple Books and Overcast. Must use `TLEN` and `CHAPTER` metadata. A post-M4B smoke test is advisable. | **Low** | Implement in `exporter.py`; add a test that parses the generated M4B metadata with `mutagen`. |
| 8 | **Novel-length audio & checkpointing** | At CPU synthesis speeds (~10× real-time for VITS, ~30–60× for XTTS-v2), a 100 000-word novel ≈ 10–20 h of processing. Each segment WAV is cached under `projects/<name>/audio/`. `state.sqlite` stores `status` per segment. On resume: skip `status == done` segments. Show ETA = `(remaining_segments × avg_seconds_per_segment)`. | **High** | Implement in `state_machine.py` from M2 onward; disk usage warning if `<10 GB free`. |
| 9 | **Python version + Coqui fork** | The original `coqui-ai/TTS` repo is archived. The maintained fork is **`idiap/coqui-ai-TTS`** (active as of Sep 2026). It requires Python ≥ 3.9, < 3.13 (PyTorch constraint). **Recommend Python 3.11** — stable, wide wheel support, supported by the fork. Pin in `pyproject.toml` with `requires-python = ">=3.11,<3.13"`. | **High** | Pin Python 3.11; use `uv` lockfile; test in CI on 3.11. |
| 10 | **OS differences** | Paths: use `pathlib.Path` everywhere, never string concatenation. FFmpeg: detect at startup in `castbook doctor`; on macOS install via Homebrew, on Linux via apt. GPU: detect CUDA (`torch.cuda`), MPS (`torch.backends.mps`), then CPU fallback; surface in `castbook doctor`. | **Medium** | Utility `castbook/utils/env.py`; tested in `test_doctor.py` with mocked `shutil.which`. |

---

### 3.3 Dependency Table (Phase A)

| Name | Purpose | Licence | Approx Size | Alternatives / Notes |
|---|---|---|---|---|
| **uv** | Package / venv management | Apache-2.0 | ~10 MB binary | pip + venv (slower) |
| **Python 3.11** | Runtime | PSF | — | 3.10 (older), 3.12 (some wheel gaps) |
| **pydantic v2** | Data models, JSON schemas | MIT | ~2 MB | attrs (less ergonomic) |
| **click** | CLI framework | BSD-3 | ~0.5 MB | typer (wraps click, fine too) |
| **fastapi** | HTTP API (M5 scaffold only) | MIT | ~1 MB | Flask (less modern) |
| **uvicorn** | ASGI server for FastAPI | BSD-3 | ~0.3 MB | hypercorn |
| **spaCy + en_core_web_sm** | Sentence/paragraph segmentation | MIT | ~15 MB model | NLTK (less accurate) |
| **ebooklib** | EPUB parsing | **AGPL-3.0** ⚠️ | ~0.1 MB | `epub-meta` (MIT, read-only); must verify if AGPL is acceptable |
| **python-docx** | DOCX parsing | MIT | ~0.5 MB | docx2txt (simpler, less structural) |
| **pdfplumber** | PDF text extraction (MIT) | MIT | ~0.5 MB | PyMuPDF (AGPL ⚠️), pypdf (BSD-3, weaker) |
| **pypdf** | PDF metadata | BSD-3 | ~0.5 MB | — |
| **langdetect** | Language identification | Apache-2.0 | ~2 MB | fasttext LangID (faster, MIT) |
| **llama-cpp-python** | Local LLM inference (attribution) | MIT | ~5 MB + model | Ollama (easier setup, less portable) |
| **idiap/coqui-ai-TTS** | XTTS-v2 + VITS-VCTK TTS | MPL-2.0 (lib) / Coqui PML (XTTS-v2 model weights) ⚠️ | ~50 MB lib + models separately | kokoro-tts (MIT, newer but smaller ecosystem) |
| **torch** | ML runtime | BSD-3 | ~800 MB (CPU) / ~2.5 GB (CUDA) | — |
| **FFmpeg** (system) | Audio joining, normalisation, M4B | LGPL-2.1 | system pkg | — |
| **pydub** | Python FFmpeg wrapper | MIT | ~0.1 MB | soundfile (lower-level) |
| **mutagen** | M4B/MP3 metadata read/write | GPL-2.0 ⚠️ | ~0.5 MB | `eyed3` (GPL ⚠️); investigate `mediafile` (MIT) |
| **pytest** | Testing | MIT | ~1 MB | — |
| **ruff** | Linting + formatting | MIT | ~5 MB | flake8 + black (two tools) |
| **SQLite** (stdlib) | Job / stage state | Public domain | stdlib | — |

> ⚠️ = licence flag (AGPL, GPL, or non-commercial). Each flagged item needs a decision in DECISIONS.md and LICENSES.md.

**Key licence decisions needed before M0:**
- `ebooklib` AGPL-3.0 → acceptable for a student project distributed as source? Or switch to `epub-meta` + manual spine parsing?
- `mutagen` GPL-2.0 → replace with `mediafile` (MIT) for M4B tag writing?
- XTTS-v2 model weights (Coqui PML, non-commercial) → acceptable for this academic project?

---

## 4. Questions for Approval (max 3)

### Q1 — AGPL/GPL dependencies

`ebooklib` (EPUB parsing) is AGPL-3.0. For a student academic project **not distributed as a binary**, AGPL source-availability requirements are usually met by keeping the repo open. However the project document says to prefer permissive alternatives.

| Option | Trade-off |
|---|---|
| **A. Keep ebooklib (AGPL-3.0)** | Zero extra work; AGPL is fine for open academic use; note it in LICENSES.md |
| **B. Replace with epub-meta + manual spine parsing** | More code (~150 lines), fully MIT; epub-meta is read-only and less maintained |
| **C. Replace with html.parser + zipfile (stdlib only)** | EPUB is a ZIP of XHTML; a small custom extractor keeps zero extra deps and is MIT-equivalent |

**→ Recommendation: A** (keep `ebooklib`) for now; flag in LICENSES.md; revisit if the project is ever packaged for distribution.

---

### Q2 — Local LLM runner for attribution

The attribution stage needs a local LLM. Two credible options:

| Option | Trade-off |
|---|---|
| **A. llama-cpp-python** | Single pip install; GBNF grammar for constrained JSON; no server process; works fully offline; harder to swap models |
| **B. Ollama (external process)** | Easier model management (`ollama pull`); JSON mode built-in; requires Ollama daemon running; harder to control in CI |
| **C. Defer: stub Attributor, decide in Week 4 after benchmark** | Unblocks M0/M1 now; attribution is not needed until M3 |

**→ Recommendation: C** — implement the `Attributor` abstract interface in M0, leave the concrete implementation as a stub, and decide in Week 4 (as the project plan already says). Both A and B stay viable.

---

### Q3 — Python package name

The CLI command in the spec is `castbook`. The Python package name should match.

| Option | Trade-off |
|---|---|
| **A. `castbook`** (matches CLI) | Consistent; `import castbook`; easy to remember |
| **B. `ebook2audiobook`** (matches existing GitHub repo name) | Familiar if comparing to the reference tool; longer to type |
| **C. `cb`** | Short; may clash with other packages |

**→ Recommendation: A** — `castbook` everywhere; the project folder on disk stays `EBOOK2AUDIOBOOK` as-is.

---

*Waiting for your approval before writing any code.*
