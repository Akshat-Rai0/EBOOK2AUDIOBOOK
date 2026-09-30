# PARKING_LOT.md — Deferred features and out-of-scope ideas

Items here are **not** built in Phase A (Project-I).

## Phase B (formats)

| Item | Raised | Notes |
|---|---|---|
| OCR for scanned PDFs | 2026-10-01 | Tesseract; detect threshold already in PdfExtractor |
| MOBI / AZW3 / FB2 | 2026-10-01 | Calibre `ebook-convert` subprocess bridge |
| Image-only EPUB | 2026-10-01 | Likely out of scope entirely |

## Phase C (languages)

| Item | Raised | Notes |
|---|---|---|
| Hindi TTS | 2026-10-01 | XTTS-v2 supports it; test Week 7 |
| Bengali / Marathi / Telugu / Tamil TTS | 2026-10-01 | Fairseq MMS-TTS; MahaTTS |
| Per-paragraph language detection | 2026-10-01 | `langdetect` already in Phase A |
| Optional translation | 2026-10-01 | Helsinki-NLP/opus-mt offline models |

## Phase D (extras)

| Item | Raised | Notes |
|---|---|---|
| Voice cloning from user reference | 2026-10-01 | XTTS-v2 supports; legal review needed |
| Bark / Tortoise TTS engines | 2026-10-01 | High quality but slow |
| SML-style pause/voice tags | 2026-10-01 | SSML-like markup |
| Additional output formats (OGG, FLAC) | 2026-10-01 | Low effort with FFmpeg |
| Em-dash dialogue splitting | 2026-10-01 | Architecture caveat #4; flagged but not split in Phase A |
| LangGraph orchestration | 2026-10-01 | Evaluate Week 5; compare to plain state machine |
| Cloud TTS fallback | 2026-10-01 | Out of scope for offline-first design |
| Native desktop app (Electron/Tauri) | 2026-10-01 | Phase D or later |
