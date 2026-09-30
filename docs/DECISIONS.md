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
