# LICENSES.md — Dependency and model licence tracking

> Verify each entry before distributing. This is a record, not legal advice.
> ⚠️ = flagged licence that may affect distribution.

## Runtime dependencies

| Package | Licence | Notes |
|---|---|---|
| pydantic ≥ 2.7 | MIT | ✓ permissive |
| click ≥ 8.1 | BSD-3 | ✓ permissive |
| fastapi ≥ 0.111 | MIT | ✓ permissive |
| uvicorn ≥ 0.29 | BSD-3 | ✓ permissive |
| spacy ≥ 3.7 | MIT | ✓ permissive |
| **ebooklib ≥ 0.18** | **AGPL-3.0** ⚠️ | Acceptable for open academic source distribution. Binary distribution requires source offer. |
| python-docx ≥ 1.1 | MIT | ✓ permissive |
| pdfplumber ≥ 0.11 | MIT | ✓ permissive; chosen over PyMuPDF (AGPL) |
| pypdf ≥ 4.2 | BSD-3 | ✓ permissive |
| langdetect ≥ 1.0.9 | Apache-2.0 | ✓ permissive |
| httpx ≥ 0.27 | BSD-3 | ✓ permissive |
| pydub ≥ 0.25 | MIT | ✓ permissive |
| mediafile ≥ 0.12 | MIT | ✓ permissive; chosen over mutagen (GPL) |

## Dev-only dependencies

| Package | Licence | Notes |
|---|---|---|
| pytest | MIT | dev only |
| pytest-cov | MIT | dev only |
| ruff | MIT | dev only |
| reportlab | BSD | dev only; synthetic PDF generation in tests only |

## System dependencies

| Tool | Licence | Notes |
|---|---|---|
| FFmpeg | LGPL-2.1+ (core) | Users install separately. Some codec plugins have separate licences. |
| Ollama | MIT | External daemon. Model weights licensed separately. |

## AI model weights

| Model | Licence | Notes |
|---|---|---|
| XTTS-v2 weights | **Coqui Public Model License** ⚠️ | **Non-commercial use only.** Attribution required. Not for commercial products. |
| VITS-VCTK weights | MIT | 109-speaker English model. |
| en_core_web_sm (spaCy) | MIT | ~15 MB. `python -m spacy download en_core_web_sm` |
| Ollama LLM (TBD, Week 4) | Varies | Meta Llama 3: Llama 3 Community License. Mistral: Apache-2.0. Decision in Week 4. |

## Project source licence

Source code: **MIT License**.  
ebooklib (AGPL) in the dependency chain means binary distribution requires source availability.
