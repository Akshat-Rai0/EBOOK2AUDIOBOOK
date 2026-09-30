# CastBook — Ebook-to-Audiobook Converter

> Local · Offline · Agentic · Multi-voice

CastBook converts an ebook (EPUB, TXT, DOCX, PDF) into a chapter-organised audiobook where the **narrator and every named character speak in their own consistent voice** — the way a radio play sounds.

Built as the Phase-A prototype for **PROJECT-I (AI24300)**, B.Tech.-V CSE (AIML), Birla Institute of Technology Mesra, Patna Campus (Monsoon 2026).

---

## What it does

1. **Extracts** structured text from EPUB, TXT, DOCX, or PDF (text layer).
2. **Detects** chapters and dialogue spans automatically.
3. **Attributes** each line to a speaker using a local LLM (Ollama, M3+).
4. **Casts** a voice for the narrator and each character.
5. **Synthesises** audio with XTTS-v2 or VITS (M2+).
6. **Exports** chapter-wise MP3 files and a single M4B with chapter markers.

All processing is **offline after first model download**.

---

## Quick start (M0/M1)

```bash
# 1. Clone and enter the project
git clone <repo-url>
cd EBOOK2AUDIOBOOK

# 2. Set up the virtual environment (requires uv)
uv sync

# 3. Check your system
uv run castbook doctor

# 4. Ingest a book
uv run castbook ingest mybook.epub --project dracula
# Output: projects/dracula/book.json + projects/dracula/chapters.txt
```

### Running tests

```bash
uv run pytest
```

### Linting

```bash
uv run ruff check .
uv run ruff format --check .
```

---

## Project layout

```
ebook2audiobook/      ← Core library
  models/             ← Pydantic data contracts (Book, Segment, Cast, Job)
  ingest/             ← Extractors for TXT, DOCX, EPUB, PDF
  chunker/            ← Chapter detection, sentence splitting
  attribution/        ← Speaker attribution (Ollama stub, M3)
  tts/                ← TTS engines (FakeTTS + adapters stubs)
  audio/              ← Audio processing and export (M2)
  orchestrator/       ← State-machine pipeline (M2)
  store/              ← SQLite job/stage persistence (M2)
  models_manager/     ← Single-model-at-a-time loader (M2)
  utils/              ← System environment detection
  cli/                ← Click CLI (`castbook`)
apps/
  api/                ← FastAPI backend (M5 scaffold)
  web/                ← React frontend (M5 scaffold)
tests/                ← pytest suite (synthetic files, no network)
docs/                 ← Project documentation and schemas
projects/             ← Runtime job folders (gitignored)
```

---

## Hardware tiers

| Tier | RAM | GPU | Engine |
|---|---|---|---|
| Narrator-only | ≥ 4 GiB | Any | VITS |
| Multi-voice | ≥ 8 GiB | Optional (CUDA/MPS speeds things up) | Ollama LLM + XTTS-v2 |

Thresholds are targets, not benchmarked — Week 4 and Week 6 tests will confirm.

---

## Scope

**In (Phase A):** EPUB · TXT · DOCX · PDF text layer · Chapter detection · Speaker attribution · Character registry · Voice casting · XTTS-v2 + VITS · English · Chapter MP3s + M4B · Resume · CLI + FastAPI + React.

**Out (future phases):** OCR · MOBI/AZW3 · Non-English TTS · Voice cloning · Cloud services.

See `docs/PARKING_LOT.md` for deferred ideas.

---

## Authorship

Akshat Rai (BTECH/15096/24) and Divyanshu Bhusan (BTECH/15081/24)  
Under guidance of Dr. Naiyar Iqbal  
Department of CSE, BIT Mesra, Patna Campus
