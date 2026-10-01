# Benchmark Report: M2 Narrator-Only Pipeline

**Date:** 2026-10-01  
**Milestone:** M2 (`m2-narrator-pipeline`)  
**Hardware:** Apple M-series (16 GB Unified RAM, CPU/MPS backend)

---

## 1. Overview
Evaluates the narrator-only end-to-end pipeline: text chunking, SQLite-backed segment state tracking, WAV synthesis, concatenation, loudness normalization (-18 dBFS peak target), and chapter-wise MP3/M4B export with FFmpeg metadata.

## 2. Key Metrics
- **Job Store:** SQLite WAL mode with atomic segment registration (`INSERT OR IGNORE`).
- **Synthesis:** Clean segment synthesis via TTS adapter with duration proportional to word count.
- **Audio Processing:** Pure-Python stdlib `wave` assembly for zero-dependency chapter concatenation.
- **Loudness Normalization:** Fast peak-scaling normalization to -18 dBFS with gain limiting.
- **Resume & Crash Resilience:** Verified idempotent re-runs; pre-existing `DONE` segments are never re-synthesized.
- **Export:** Verified chapter MP3 creation and single multi-chapter M4B container with embedded chapter marks.
