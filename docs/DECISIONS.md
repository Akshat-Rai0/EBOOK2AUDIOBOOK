# DECISIONS.md — Architecture and dependency decisions

| Date | Decision | Alternatives considered | Reason |
|---|---|---|---|
| 2026-10-01 | **Package name: `ebook2audiobook`** (import); CLI command: `castbook` | `castbook` for both; `cb` | User preference; CLI name matches project spec; package name matches repo |
| 2026-10-01 | **Keep `ebooklib` (AGPL-3.0)** for EPUB parsing | `epub-meta` (MIT, read-only); stdlib zipfile + html.parser | Zero extra work; AGPL acceptable for open academic use; most feature-complete |
| 2026-10-01 | **Ollama** as LLM runner for speaker attribution (M3) | `llama-cpp-python` (GBNF grammar); Transformers | JSON mode built-in; easier model management; good student DX |
| 2026-10-01 | **`pdfplumber` (MIT)** for PDF text extraction | `PyMuPDF` (AGPL-3.0); `pypdf` (BSD-3, weaker extraction) | Permissive licence; good text + table extraction; avoids AGPL chain |
| 2026-10-01 | **Python 3.11** pinned | 3.10 (older); 3.12 (some TTS wheel gaps at time of writing) | Wide wheel support; stable; supported by `idiap/coqui-ai-TTS` maintained fork |
| 2026-10-01 | **Plain state machine** for orchestration (not LangGraph) | LangGraph | LangGraph evaluation deferred to Week 5 per project plan; avoid premature dependency |
| 2026-10-01 | **`mediafile` (MIT)** for M4B tag writing | `mutagen` (GPL-2.0); `eyed3` (GPL-2.0) | Permissive licence; avoids GPL in the dependency chain |
| 2026-10-01 | **`FakeTTS`** as the only TTS engine in tests/CI | Real model stubs | No model download in CI; fast; deterministic; validates full WAV contract |
| 2026-10-01 | **`idiap/coqui-ai-TTS`** (maintained fork) as TTS library | Original `coqui-ai/TTS` (archived); `kokoro-tts` | Active maintenance; XTTS-v2 support; VITS-VCTK multi-speaker available |
| 2026-10-01 | **VITS-VCTK** (109-speaker, MIT) as low-resource multi-speaker fallback | Single-speaker VITS (cannot cast characters) | Multi-speaker required for character casting even in narrator-only+ mode |
| 2026-10-01 | **`chapters_override.json` escape hatch** for chapter detection | Manual editing of book.json | User-facing; survives re-runs; per-project; JSON is student-readable |
| 2026-10-01 | **`langdetect`** for language identification | `fasttext` LangID (MIT, faster) | Simpler install; bundles model; offline; fasttext is a future upgrade path |
| 2026-10-01 | **`httpx`** for local Ollama HTTP client | `requests`; stdlib `urllib` | Modern sync+async API, excellent timeout handling, connection pooling |
| 2026-10-01 | **Constrained JSON decoding (`format: "json"`)** for attribution | Raw prompt parsing; regex extraction | Ollama guarantees syntactically valid JSON at token level; eliminates JSON parsing errors |
| 2026-10-01 | **`llama3.2:3b`** default model for attribution | `llama3.1:8b`; `mistral:7b` | Small memory footprint (~2.2 GB), fast CPU/MPS inference, fits in 8 GB RAM tier |
| 2026-10-01 | **Significant token intersection** for alias merging in `CharacterRegistry` | Embedding similarity; edit distance | Zero-dependency, deterministic, fast, conservative (avoids false-positive merges) |
| 2026-10-03 | **VITS speaker metadata: p225-style IDs only** | Ship VCTK speaker-info.txt with warning; hard-code gender/accent | Coqui issue #2258: pNNN IDs don't reliably match VCTK speaker-info.txt. Gender/accent deferred to M4 listen-labelling. Narrator maps to p225 via config.toml (temporary). |
| 2026-10-03 | **Model weights in Coqui cache + manifest** | Project-local models/ directory; Env var only | Coqui's default cache (~/.local/share/tts on Linux, ~/Library/Application Support/tts on macOS). Manifest at ~/.castbook/models.json. CASTBOOK_MODELS_DIR env var sets TTS_HOME (Coqui honours this). ~/.castbook/config.toml for persistent settings. |
| 2026-10-03 | **Sentence splitting at punctuation nearest middle** | Fail with error; Let Coqui truncate; Hard split at end | Split at punctuation nearest middle (`, ; : — ( ) [ ]`), then spaces, never mid-word. Measured after normalisation. Stable sub-ids (s00a/s00b) for resume. 150ms pause between pieces. VITS max_chars=400, XTTS max_chars=255 (from config). |
| 2026-10-03 | **TTS dependency fixes** | Use latest transformers | Coqui requires transformers<5.0 for compatibility. PyTorch 2.9+ requires torchcodec for audio I/O. espeak-ng required for phonemization. |
| 2026-10-03 | **XTTS MPS slower than CPU** | Use MPS by default on Apple Silicon | Benchmark on Darwin arm64 shows XTTS MPS (RTF 1.52x, 4.4 GB RSS) is slower than CPU (RTF 0.70x, 1.1 GB RSS). XTTS defaults to CPU on Darwin. VITS CPU is very fast (RTF 0.11x, 42 MB RSS). |
| 2026-10-09 | **V2 hybrid attribution pipeline** (rules → LLM → alternation) | V1 single-step LLM per paragraph; rules-only | Rules cover obvious dialogue-tag cases at high precision (≈0.98); LLM only sees unmarked quotes, reducing token usage and hallucination risk; alternation corrects remaining gaps without an extra LLM call. |
| 2026-10-09 | **ID-based attribution format** (`[Q1]`, `[Q2]` markers; `{"attributions": [...]}` JSON response) | Positional indexing (V1 `{"spans": [...]}`); named-entity extraction | ID-based format avoids positional mismatch bug (P3) when segment and response counts differ; closed-set speaker list in system prompt prevents invented names (P6); quote markers remain stable even if the LLM rephrases prose in its reasoning. |
