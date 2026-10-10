# M2b plan: real TTS engines, ModelManager, benchmark, loudness

**Milestone:** M2b (real synthesis on top of the M2 narrator pipeline)
**Branch:** `m2b-real-tts` (merged into `m4-casting-multivoice`)
**Status:** COMPLETE (Core synthesis & ModelManager wired into pipeline; benchmark/loudness evaluation pending approval)
**Machine (target):** Apple Silicon, 16 GB unified memory, CPU or MPS

This plan is the Week 6 TTS evaluation from `docs/project_document.md`, implemented against the existing M2 pipeline. It does **not** implement M4 casting, multi-voice synthesis, or the web UI.

---

## 0. Confirmed current state (read from code, 2026-10-03)

- `ebook2audiobook/tts/` has only `engine.py` (`TTSEngine` ABC + `TTSError`) and `fake_tts.py`. There is no `tts/__init__.py`.
- `castbook convert --engine` accepts `fake` and `vits` only. Non-fake prints *“VITS engine integration benchmark scheduled for Week 6”* and uses `FakeTTS`. There is no `xtts` choice yet.
- `PipelineOrchestrator` already chunks with `split_to_chunks(sent, max_chars=self.tts.max_chars)`, synthesises `VoiceRef(engine=self.tts.engine_name, voice_id="narrator")`, assembles chapters, peak-normalises, exports MP3 (`libmp3lame`) and M4B (`aac`).
- `split_to_chunks` logs a warning and **includes oversized sentences as-is**. `synthesize` then raises `ValueError` if `len(text) > max_chars`. That is a real failure mode for XTTS (~255 chars).
- `models_manager/manager.py` is a stub (name tracking only; no load).
- `AudioProcessor.normalize_loudness` scales the **peak** to −18 dBFS in pure Python. The docstring says LUFS; the code is not LUFS.
- `OllamaAttributor.close()` only closes the HTTP client. `AttributionStage` calls `close()` in `finally`. That does **not** unload weights from the Ollama daemon.
- `castbook doctor` reports FFmpeg path/version, not encoder availability (`libmp3lame`, `aac`).
- Joins are silence insertion (300 ms sentence / 700 ms paragraph). There is no cross-fade.
- CI (`.github/workflows/ci.yml`) runs `uv sync --frozen` with no extra groups, then pytest with `no_proxy: "*"`.
- `docs/benchmarks/m2_narrator_benchmark.md` has no measured numbers.

---

## 1. Adapters

Both adapters implement the existing `TTSEngine` contract. They do not change `synthesize(text, voice) -> WAV bytes`. They raise `TTSError` for unknown voices and empty text, and `ValueError` when `len(text) > max_chars`.

Lazy-import `TTS` from `coqui-tts` so `uv sync` without the extra still imports the rest of the package.

### 1.1 VITS (`ebook2audiobook/tts/vits.py`)

- Coqui model id: `tts_models/en/vctk/vits` (already chosen in `DECISIONS.md`).
- Speakers: 109 ids of the form `p225` … `p376` (p315 has no transcript in the corpus; the checkpoint still exposes 109 speakers).
- `engine_name`: `"vits"`.
- `sample_rate`: read from the loaded config (expected **22050** Hz). Declare whatever the checkpoint actually uses; do not hard-code if it differs.
- `max_chars`: VITS has no silent 255-char truncate. Use a conservative practical cap (proposed **400**, matching `DialogueSegmenter` default) so a single call cannot run away. Confirm against the loaded config at implementation time.
- `list_voices()`: one dict per speaker: `{id, name, gender, accent}` where gender/accent are filled only if we have a **verified** mapping. See question 1 — Coqui issue #2258 reports that `pNNN` ids in this checkpoint do **not** reliably match `speaker-info.txt`.
- `synthesize`: `tts.tts(text=..., speaker=voice.voice_id)` → numpy → 16-bit mono WAV bytes.
- Narrator default: a reserved speaker id chosen after a short listen (not in this plan’s code until you pick it). Until then, map `voice_id="narrator"` to a documented default (proposed `p225`) so the existing orchestrator call works without an interface change.
- Licence: Coqui lists the VITS-VCTK **weights** as MIT/Apache in model metadata; the **VCTK corpus** is ODC-By 1.0 (attribution required). Record both in `LICENSES.md` and flag attribution in the README. This is not legal advice.

### 1.2 XTTS-v2 (`ebook2audiobook/tts/xtts.py`)

- Coqui model id: `tts_models/multilingual/multi-dataset/xtts_v2`.
- Hugging Face files (do not download in this planning step): `model.pth` ≈ **1.87 GB**, `dvae.pth` ≈ **211 MB**, `speakers_xtts.pth` ≈ **7.75 MB**, plus config/vocab. Total cache **> 2 GB**.
- Built-in speakers: named entries in `speakers_xtts.pth` (docs list many, e.g. `Ana Florence`). Exact count will be read from the loaded `speaker_manager` at first download — **not guessed in catalogue files until then**. Voice cloning from a user WAV is out of scope (parking lot).
- `engine_name`: `"xtts"`.
- `sample_rate`: **24000** Hz (XTTS output). Different from VITS — see caveat 6.
- `max_chars`: Coqui `XttsArgs.num_chars` default is **255**. English tokenizer guidance is ~250 characters. Adapter will read `num_chars` from config and expose that as `max_chars` (fallback 250).
- Speaker conditioning: `get_conditioning_latents` (gpt latent + speaker embedding) is computed **once per `voice_id`**, cached on the adapter instance for the process lifetime. Analogy: like saving a character’s “voice print” after the first line so later lines skip the costume fitting. Cache key: `(engine, voice_id)`. Bounded by unique voices used in the run. This is for M4, but we do it now so M4 does not rework the adapter.
- Device: default **CPU** on Darwin. If `--device mps` is requested, try MPS once; on missing ops (`aten::_fft_r2c` and similar, Coqui issue #3649) catch, log, print a clear “falling back to CPU” line, and continue on CPU. Benchmarks will **measure** CPU vs MPS rather than assume.
- `synthesize`: English `language="en"` only. Missing model → `TTSError` naming `castbook models download xtts`.
- Licence: Coqui Public Model License — **non-commercial**. Flag in README and `LICENSES.md`.

### 1.3 Engine factory

`ebook2audiobook/tts/factory.py` (new): `get_engine(name, *, device, model_manager) -> TTSEngine`.

- `fake` → `FakeTTS` (no ModelManager).
- `vits` / `xtts` → require optional extra; if import fails, tell the user to `uv sync --extra tts`.
- If weights are absent, fail with `castbook models download <name>` and the byte size.

`castbook convert --engine fake|vits|xtts` uses this factory. The Week 6 FakeTTS fallback is removed.

---

## 2. Dependency changes

Keep CI and default `uv sync` light.

```toml
[project.optional-dependencies]
tts = [
  "coqui-tts>=0.27.4",   # idiap fork; PyPI name is coqui-tts, not TTS
  "torch",
  "torchaudio",
]
```

Notes from the fork (not yet installed here):

- Package: **`coqui-tts`** (MPL-2.0 library). From 0.27.4 **PyTorch is not bundled**; we add `torch`/`torchaudio` explicitly.
- Fork claims Python ≥ 3.10, < 3.15; we stay on pinned **3.11** (`requires-python = ">=3.11,<3.13"`).
- Default extra should **not** pull Japanese/Chinese G2P (`[ja]`, `[zh]`) — those pull extra spaCy stacks we do not need for English.
- `pyloudnorm` (MIT): add only to the loudness harness extra **or** to `tts` after you approve. Proposed: `tts` extra so the comparison harness and later option C share one install. Size is small (numpy/scipy already likely present via coqui).
- Do **not** add TTS deps to the default `[project] dependencies`.
- CI stays `uv sync --frozen` (no `--extra tts`). Default tests never import `TTS`.

**Install verification (after approval, before locking):** `uv sync --extra tts` on this Mac. Report: whether it resolves against current pydantic/spaCy pins, wheel vs source, and **approximate venv size delta**. No model download in that step. If the resolve wants a download > 100 MB (torch wheels often do), **stop and ask** with the size.

---

## 3. Where model files are stored

Coqui’s default cache is under the user data dir (typically `~/Library/Application Support/tts` on macOS, via `TTS.utils.generic_utils.get_user_data_dir("tts")`).

**Proposal (question 2):** keep Coqui’s cache, and treat `castbook models download` as a wrapper that:

1. Prints licence + size.
2. Asks for confirmation (Click confirm; non-interactive `--yes` for scripts).
3. Calls Coqui’s download into that cache.
4. Records a small manifest at `~/.castbook/models.json` (path, size, licence, downloaded_at) so `castbook models list` works without loading weights.

We will not copy multi-gigabyte checkpoints into the git repo or into `projects/`.

---

## 4. ModelManager design

**Role:** process-wide gate so **at most one heavy in-process model** is loaded, plus an explicit Ollama eviction so the LLM in the **other process** is not still resident.

```text
load("vits")  → ok
load("xtts")  while vits loaded → refuse (TTSError / ModelManagerError)
unload()      → del model, gc.collect(), torch cache empty
load_ollama() → refuse if TTS loaded; after attribution, unload_ollama()
```

### 4.1 In-process (TTS)

- Singleton (`ModelManager.instance()`), matching AGENTS.md.
- `load(name: str, loader: Callable) -> Any`: if another name is loaded, **refuse** (do not auto-swap). Caller must `unload()` first. This matches “refuse to load a second heavy model while one is loaded”.
- `unload()`: drop reference, `gc.collect()`, `torch.cuda.empty_cache()` if CUDA, `torch.mps.empty_cache()` if MPS exists. Log RSS before/after via `resource.getrusage` / `psutil` if present, and peak if available (`resource.RUSAGE_SELF.ru_maxrss` on macOS is bytes).
- Adapters obtain the Coqui object only through the manager.

### 4.2 Ollama (other process)

Current `close()` does not free daemon memory. Unify by giving ModelManager:

- `unload_ollama(model_tag, base_url)` → `POST /api/generate` with `keep_alive: 0` (and empty prompt), then optional `POST /api/stop` if the running Ollama version supports it.
- `AttributionStage.finally` calls **this**, not only HTTP `close()`.
- `is_ollama_loaded()`: `GET /api/ps` (running models). Used by the “LLM and TTS never together” test with a fake HTTP layer — no real daemon in CI.

We **reuse** the attribution-stage eviction point and **route it through ModelManager** so convert/attribute share one policy. We do not invent a second eviction path.

### 4.3 CLI

```text
castbook models list
castbook models download <vits|xtts|llama3.2:3b>
```

`download` prints size and licence, then confirms. `llama3.2:3b` is a thin wrapper around `ollama pull` only if `ollama` is on PATH; if not, print the command. Still ask before any pull > 100 MB.

---

## 5. Caveats (verified) — severity and mitigation

| # | Caveat | Finding (2026-10-03) | Severity | Mitigation |
|---|---|---|---|---|
| 1 | XTTS-v2 and MPS | Coqui #3649: `aten::_fft_r2c` unimplemented / hangs on MPS. Community reports CPU-only on Mac. | **High** | Default CPU on Darwin. Optional MPS attempt with fallback. `castbook bench --device cpu\|mps` records both if MPS runs. |
| 2 | Fork Python / pins | `coqui-tts` 0.27.x supports 3.10–3.14; pydantic v2 and spaCy 3.x are compatible on paper. Torch is a separate install from 0.27.4. **Not yet installed here.** | **High** | Optional extra only. First build commit is `uv sync --extra tts` + lock update. If resolve conflicts, stop and report. |
| 3 | Licences | XTTS-v2: CPML non-commercial. VCTK corpus: ODC-By 1.0 (attribution). VITS-VCTK weights: MIT/Apache in Coqui metadata. Built-in XTTS speakers inherit CPML. | **High** | `LICENSES.md` + README warning. `models download` prints the licence before confirm. |
| 4 | Download size | XTTS-v2 cache **> 2 GB** (`model.pth` 1.87 GB + DVAE ~211 MB). VITS-VCTK historically ~350 MB (architecture_review). Torch wheels also often > 100 MB. | **High** | Always print size and confirm. Never download > 100 MB in an agent session without asking. |
| 5 | Per-call char limits | XTTS `num_chars` **255**; English often cited as **250**. VITS is looser. `split_to_chunks` currently **passes through** oversized sentences; synthesize then raises. | **High** | Engine `max_chars` from config. Chunker must split inside a sentence when needed (question 3). |
| 6 | Sample rates | VITS ~22.05 kHz; XTTS **24 kHz**; FakeTTS 22.05 kHz. One convert job uses one engine, so chapter concat is uniform **if** we honour `engine.sample_rate`. Mixing engines in one chapter is out of scope. | **Medium** | `AudioProcessor` uses the engine rate. If a WAV header disagrees, resample **once** at assemble time (scipy or ffmpeg) to the engine rate, and test it. Export lets ffmpeg encode. |
| 7 | Clicks at joins | Current assembly is hard concat + silence. No fade. | **Medium** | After first real audio, listen. If clicks: 10 ms equal-power cross-fade at non-silence joins (processor change + test). Do not pre-emptively change pause timings. |
| 8 | Memory growth | Unmeasured. Coqui/XTTS can leak CUDA/MPS caches; speaker latents are small. | **Medium** | Bench + full-book script log RSS every N segments. If RSS climbs > 500 MiB over a 20k-word run, reload the engine every N segments (N proposed 200). Decision after numbers. |
| 9 | Ollama unload | `close()` is HTTP-only. Weights stay in the Ollama process until `keep_alive: 0` / `ollama stop`. | **High** | ModelManager `unload_ollama` via API; stage `finally` uses it. Test with mocked `/api/ps`. Real free-memory check is for the full-book / doctor notes, not CI. |
| 10 | FFmpeg builds | MP3 needs `libmp3lame`; M4B needs `aac`. Homebrew builds usually have both; some distro builds do not. | **Medium** | `castbook doctor` runs `ffmpeg -encoders` and reports lame/aac yes/no. Convert fails with that same hint. |
| 11 | Short segments | “No.” / “Why?” can glitch. M4 already parks padding. | **Low (M2b)** | Log duration of outputs < 400 ms during bench and the short test book. No padding in M2b unless you ask. |

---

## 6. TTS text normaliser

**Exists today:** ingest `clean_text` (unicode, hyphens, quotes, whitespace). That runs at extract time and **must stay** as stored book text.

**Missing:** synthesis-only expansion of numbers and abbreviations.

New module `ebook2audiobook/tts/normalise.py`, applied **only** inside `synthesize` (or immediately before the engine call in the adapter). Stored `Segment.text` unchanged.

Rules (minimal, unit-tested):

| Rule | Example in → out |
|---|---|
| Ellipses | `Wait...` → `Wait…` spoken as “Wait,” with a pause word `...` → `…` or `… ` → `, ` |
| Em/en dashes | `yes—no` → `yes - no` |
| Titles | `Mr.` `Mrs.` `Ms.` `Dr.` `St.` → `Mister` `Missus` `Miss` `Doctor` `Saint` (word-boundary, keep `St.` as Saint not Street — document the bias) |
| Integers | `42` → `forty-two` (`num2words`, already a coqui dependency; implement a tiny fallback for tests without the extra) |
| Years | `1999` → `nineteen ninety-nine` when the token is 4 digits 1100–2099 |
| All-caps words length ≥ 3 | `NASA` left as-is (acronym); `WAIT` → `Wait` if it is a common English word — **too clever**. Safer rule: if the whole token is A–Z and length ≥ 4 and not a known acronym list of ~20, convert to title case so XTTS does not spell-letter. |
| Stray symbols | drop leftover `* _ # ~` not part of numbers |

Empty-after-normalise still raises `TTSError`.

---

## 7. Engine-aware chunking

Today: sentence split, then `split_to_chunks`. Oversized **sentences** are emitted whole.

Change `split_to_chunks` (or a thin wrapper used only by the pipeline) so that if `len(sentence) > max_chars`, split further at `, ; :` then at spaces, **never** mid-word. Log when a mid-sentence split happens. Record the rule in `DECISIONS.md`.

`DialogueSegmenter(max_chars=400)` is M4; narrator pipeline uses the engine’s `max_chars` already. After adapters exist, XTTS jobs will chunk at 250/255 automatically **if** we fix the oversized-sentence path.

---

## 8. Benchmark method

### Passage

Store ~500 words of **public-domain** English in-repo, e.g. the opening of *Pride and Prejudice* (Gutenberg), as `docs/benchmarks/fixtures/pride_500.txt`. Not a copyrighted modern book.

### Command

```text
castbook bench --engine vits|xtts [--device cpu|mps] [--runs 3]
```

Requires `--extra tts` and a downloaded model.

### Metrics (mean of 3 warm runs; cold separate)

| Metric | How |
|---|---|
| Cold-load time | First `get_engine` wall time |
| Warm real-time factor | compute_seconds / audio_seconds. RTF > 1 means slower than playback (a 10-minute chapter that takes 20 minutes to synthesise has RTF 2). |
| Peak RSS | `resource.getrusage` max RSS |
| MPS / VRAM | `torch.mps.current_allocated_memory()` if MPS; else “n/a (unified / CPU)” |
| Audio duration | WAV frames / rate |
| Mean of 3 | After one discarded cold synth of a 1-sentence warmup |

Write:

- `docs/benchmarks/<machine-slug>.md` — hostname/chip/RAM from `castbook doctor` plus the table.
- Replace `docs/benchmarks/m2_narrator_benchmark.md` body with the real numbers **and** a link to the machine file.

The agent will not run a 2 GB XTTS download unless you confirm.

---

## 9. Loudness comparison harness

**Problem:** peak dBFS is the height of the loudest spike, like the tallest wave in a harbour. Perceived loudness (LUFS) is more like how full the harbour sounds over time. Two voices can share a −18 dBFS peak and still feel mismatched.

**True peak:** the reconstructed analog peak can sit *between* digital samples — like a ball thrown between two photographs of it. A true-peak limiter catches those inter-sample spikes so export does not clip on DAC.

Harness: `scripts/loudness_compare.py` (dev-only, not imported by CI).

Inputs: the same real clips from VITS and XTTS, several speakers, narration paragraph + short dialogue-like lines.

Three treatments:

| Id | Method |
|---|---|
| A | Current peak scale to −18 dBFS |
| B | FFmpeg two-pass `loudnorm` per concatenated chapter-like clip, I=−18, TP=−1.5, LRA=11 |
| C | `pyloudnorm` (MIT, ITU-R BS.1770-4) integrated loudness **per segment**, gain to −18 LUFS, then a true-peak limit per joined chapter (ffmpeg `alimiter` or `loudnorm` TP only) |

**Short clips:** BS.1770 uses ~400 ms gating blocks. For duration < 0.4 s: skip integrated LUFS, apply a **fixed** gain equal to the median gain of that voice’s longer clips (or 0 dB if none). Report how many clips hit this path.

Outputs:

- `docs/benchmarks/loudness_samples/` WAV files named `{engine}_{voice}_{A|B|C}.wav`
- A markdown table: integrated LUFS per voice, spread (max−min LUFS across voices), sample peak dBFS, clip count (samples at ±32767), processing time.

**Stop here for your listen / decision.** Recommendation will be written after numbers exist. Prior: **C** if the spread between voices is the pain (multi-voice later); **B** if chapter-level consistency is enough and FFmpeg is already required; **A** only if B/C colour the timbre badly.

Default stays A until you choose.

---

## 10. Wiring convert, full-book script, doctor

- `convert --engine fake|vits|xtts` via factory. Missing extra / missing weights: non-zero exit, named download command.
- FakeTTS remains default.
- `castbook doctor`: add lame/aac encoder flags.
- Full-book script `scripts/full_book_vits.sh` (you run it): ingest → convert vits → print wall time, RTF, peak RSS → `ffprobe` on the M4B → `kill -9` mid-run on a copy project → convert again → assert DONE segment wav mtimes unchanged.

We will not run the 20k-word book in the agent.

---

## 11. Test plan (no network)

| Area | What | Marker |
|---|---|---|
| Adapter contract | Parametrised over FakeTTS always; over VITS/XTTS only if models present | `models` skip by default |
| Contract assertions | RIFF/WAV header, `nchannels=1`, rate == `engine.sample_rate`, `len(text)==max_chars` succeeds, empty text → `TTSError`, unknown voice → `TTSError` | |
| ModelManager | Fake heavy model: sequence, refuse double-load, released after unload (ref gone) | default |
| LLM vs TTS | Fake Ollama `/api/ps` + fake TTS load: never both “loaded” | default |
| Normaliser | Each rule above | default |
| Chunking | Sentence longer than `max_chars` splits at comma/space; each piece ≤ cap | default |
| Loudness (after you choose) | Option A/B/C implementation tests with synthetic WAV | default |
| CI | No `--extra tts`; `models` tests skipped | |

AGENTS.md currently says FakeTTS is the only engine in tests. We keep that for **default** pytest. The `models` marker is an opt-in exception, recorded in `DECISIONS.md`.

---

## 12. Docs and follow-up commits (after approval, listed order)

Matches the requested commit sequence:

- [x] extras → pyproject.toml [tts] optional dependency group
- [x] ModelManager → singleton, load/unload, Ollama eviction, tests
- [x] VITS → adapter with list_voices(), narrator mapping, tests
- [x] XTTS → adapter with latent cache, MPS fallback, tests
- [x] convert wiring → factory integration, CLI --engine option
- [x] normaliser/chunking → synthesis normaliser, enhanced split_to_chunks
- [x] Contract tests → FakeTTS always, VITS/XTTS with @pytest.mark.models
- [x] ModelManager integration → factory check, AttributionStage unload_ollama
- [x] full-book script → end-to-end test with resume verification (scripts/full_book_vits.sh)
- [ ] benchmark → script created, ran once (awaiting approval for final version)
- [ ] loudness harness → stop for user decision on A/B/C
- [ ] chosen loudness → implementation after decision
- [ ] doctor updates → lame/aac encoder flags
- [x] docs → DECISIONS.md, LICENSES.md updates
- [x] datetime → separate datetime.utcnow() → timezone-aware datetime.now(UTC)

README still says “Quick start (M0/M1)” and describes real XTTS/VITS synthesis as if it existed. Update after wiring.

`docs/m4_plan.md` status: M2b supplies real engines and latent caching; M4 catalogue/casting still not done.

---

## 13. Out of scope (to add to `docs/PARKING_LOT.md` after approval)

| Item | Notes |
|---|---|
| M3 evaluation redo | 90% accuracy re-measure on labelled chapters |
| M4 voice catalogue, casting, `castbook cast *`, multi-voice synthesis | Stay on `m4-casting-multivoice`; do not start here |
| Web UI | M5 |
| Voice cloning from user WAV | Phase D; XTTS can do it; legal review |
| Other engines / languages | Bark, Tortoise, Hindi, … |
| OCR, MOBI, LangGraph | Already parked |

---

## 14. Implementation order (one commit per item, after approval)

1. Optional `tts` extra + lockfile; report install size; **no model download**.
2. Real ModelManager + `castbook models` + Ollama eviction + `LICENSES.md` model rows.
3. VITS adapter + speaker list.
4. XTTS adapter + latent cache + MPS fallback.
5. Wire `convert`; remove Week 6 FakeTTS fallback.
6. Synthesis normaliser + in-sentence chunking.
7. `castbook bench` + fixture + benchmark docs (after you approve model download).
8. Loudness harness → **stop**.
9. Chosen loudness + `DECISIONS.md`.
10. Full-book script (you run).
11. Remaining tests / markers.
12. README / m4_plan / DECISIONS / LICENSES; then utcnow commit.

No public interface change is planned except adding CLI commands and the `xtts` engine choice (additive). If `split_to_chunks` behaviour for oversized sentences must change, that is a documented behaviour change in `DECISIONS.md`.
