# AGENTS.md — Working agreements for human and AI contributors

This file governs how work is done on the CastBook project.  Both human
contributors and AI coding assistants must follow these rules.

> If you are an AI assistant (Claude Code, Antigravity, Copilot, etc.), also
> read **CLAUDE.md** (same content, included for Claude Code compatibility).

---

## Core principles

1. **Ask, do not assume.**  When a decision has real trade-offs, present options
   with your recommendation marked.  Do not make irreversible changes silently.

2. **Small commits, conventional messages.**  One milestone per branch.
   Tests and lint pass before every commit.  Never skip failing tests or bypass hooks.

3. **Clear code over clever code.**  The two authors must be able to explain
   every module in a presentation.  Use docstrings on all public functions;
   use inline comments to explain *why*, not *what*.

4. **No unexplained downloads.**  Never download a file > 100 MB without asking.
   Provide the command and state the size.

5. **Record decisions.**  Every non-obvious choice goes in `docs/DECISIONS.md`:
   date (use real calendar dates, not "Week N"), choice, alternatives, reason.

6. **Explain new concepts.**  When you introduce something a student may not know
   (constrained decoding, speaker latents, loudness normalisation, etc.), explain
   it in two lines with an everyday analogy — in your message, not in code comments.

---

## Scope guard

For every task, check `docs/PARKING_LOT.md`.  If the task touches anything listed
there (OCR, MOBI, non-English, voice cloning, cloud, LangGraph), stop and ask.

Do not scaffold or stub anything out-of-scope without explicit approval.

---

## Code style

- Python 3.11+, typed (type hints everywhere).
- `ruff` for lint and format.  Line length 100.
- `pydantic v2` for all data contracts.
- `pathlib.Path` everywhere — no string path concatenation.
- No global mutable state outside of explicit singletons (e.g. `ModelManager`).

---

## Testing rules

- Tests never touch the network or download models.
- Tests generate their own synthetic EPUB/DOCX/PDF/TXT files.
- No real book content in the repo (only tiny public-domain excerpts if needed).
- `FakeTTS` is the only TTS engine used in tests.
- `uv run pytest` must pass on a fresh clone after `uv sync`.

---

## Milestones (branch naming)

| Branch | Content |
|---|---|
| `main` | Stable, passing M0 and M1 |
| `m2-narrator-pipeline` | Audio generation, resume |
| `m3-attribution-spike` | Ollama integration, 90% accuracy test |
| `m4-casting-multivoice` | Full casting + multi-voice |
| `m5-api-react` | FastAPI + React UI |
| `m6-eval-polish` | Evaluation + polish |

---

## Session summary format

End every AI session with:
1. What changed (files created/modified).
2. Tests run and results (`pytest` output summary).
3. What is next (next milestone task).
4. Open questions (if any).
