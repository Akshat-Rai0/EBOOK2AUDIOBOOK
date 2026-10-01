"""
Speaker attribution via a local Ollama server (M3 implementation).

**How it works at a high level:**
    1.  We build a prompt that shows the LLM a paragraph and asks it to split
        the text into spans — each span is one unbroken chunk of narration,
        dialogue, or internal thought — and label who speaks.
    2.  We call ``POST /api/chat`` with ``format: "json"`` so Ollama forces
        the model to emit valid JSON (constrained decoding: like autocomplete
        that only suggests valid JSON tokens rather than free prose).
    3.  We validate the shape of the JSON, retry up to ``max_retries`` times
        on parse errors or schema violations, then fall back to narrator if all
        retries are exhausted.

**Ollama primer (everyday analogy):**
    Ollama is a local "model-as-a-service" daemon — think of it as a personal
    translator that lives on your laptop.  You send it text and it sends back
    structured answers, all without any internet.

Decision recorded: 2026-10-05 (see docs/DECISIONS.md — model choice: llama3.2).
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import httpx

from ebook2audiobook.attribution.attributor import AttributionError, Attributor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HEALTH_ENDPOINT = "/api/tags"
_CHAT_ENDPOINT = "/api/chat"

# Number of seconds to wait between retries (doubles each attempt).
_RETRY_BASE_SLEEP = 0.5

# A valid span dict — used for type documentation and fallback construction.
_SpanDict = dict[str, str]

_NARRATOR_FALLBACK_KIND = "narration"

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a literary annotation assistant. Your only task is to
split a paragraph into spans and label each span with a speaker and a kind.

Rules:
- "kind" must be exactly one of: "dialogue", "narration", "thought"
- "speaker" must be exactly one of the names in the cast list, "narrator", or "unknown"
- Use "narrator" for prose narration that no character speaks aloud
- Use "unknown" ONLY when a speaker says something but you genuinely cannot tell who
- Keep the "text" field identical to the original text (no paraphrasing)
- Every character of the input paragraph must appear in exactly one span
- Return ONLY a JSON object with a single key "spans" whose value is a list

Example output:
{"spans": [
  {"text": "She walked to the window.", "speaker": "narrator", "kind": "narration"},
  {"text": "\\"I can't sleep,\\" she said.", "speaker": "Mira", "kind": "dialogue"}
]}"""

_USER_TEMPLATE = """Cast so far: {cast}

Previous paragraph (context):
{context}

Paragraph to annotate:
{paragraph}

Return JSON only."""


# ---------------------------------------------------------------------------
# OllamaAttributor
# ---------------------------------------------------------------------------


class OllamaAttributor(Attributor):
    """
    Speaker attribution using a local Ollama LLM server.

    The attributor is stateless between calls — no conversation history is kept
    because paragraph-level attribution is independent from turn to turn.

    Parameters
    ----------
    model:
        Ollama model tag, e.g. ``"llama3.2:3b"`` or ``"mistral:7b"``.
        3B-class 4-bit models are ~2–3 GB on disk and run comfortably with 8 GB RAM.
    base_url:
        Base URL of the Ollama daemon (default ``http://localhost:11434``).
    max_retries:
        How many times to retry a malformed response before falling back to
        the narrator.  Each retry doubles the sleep time (exponential back-off).
    timeout:
        Per-request timeout in seconds.  Large paragraphs on slow CPUs may need
        more than the default.
    """

    def __init__(
        self,
        model: str = "llama3.2:3b",
        base_url: str = "http://localhost:11434",
        max_retries: int = 2,
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.timeout = timeout
        self._client = httpx.Client(timeout=self.timeout)
        logger.debug("OllamaAttributor ready: model=%s base_url=%s", model, base_url)
        self._check_daemon_running()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def attribute(
        self,
        paragraph: str,
        cast_so_far: list[str],
        context: str = "",
    ) -> list[_SpanDict]:
        """
        Attribute each span of *paragraph* to a speaker.

        On persistent LLM failure the paragraph is returned as a single
        narrator span — never silently dropped or guessed.

        Parameters
        ----------
        paragraph:
            The raw paragraph text (may contain dialogue and narration mixed).
        cast_so_far:
            List of canonical character names seen so far.
        context:
            Optional preceding paragraph for continuity.

        Returns
        -------
        list[dict]
            Each dict has ``text``, ``speaker``, and ``kind`` keys.
        """
        if not paragraph.strip():
            return []

        cast_str = ", ".join(cast_so_far) if cast_so_far else "(none yet)"
        user_msg = _USER_TEMPLATE.format(
            cast=cast_str,
            context=context.strip() or "(none)",
            paragraph=paragraph.strip(),
        )

        last_error: str = ""
        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                sleep_sec = _RETRY_BASE_SLEEP * (2 ** (attempt - 1))
                logger.debug(
                    "Attribution retry %d/%d (sleep %.1fs)", attempt, self.max_retries, sleep_sec
                )
                time.sleep(sleep_sec)

            try:
                raw = self._call_ollama(user_msg)
                spans = self._parse_response(raw, paragraph)
                return spans
            except _ParseError as exc:
                last_error = str(exc)
                logger.warning("Attribution attempt %d failed: %s", attempt + 1, exc)

        # All retries exhausted — fall back to narrator, log prominently.
        logger.warning(
            "Attribution failed after %d attempts for paragraph %.60r…  "
            "Falling back to narrator.  Last error: %s",
            self.max_retries + 1,
            paragraph,
            last_error,
        )
        return self._narrator_fallback(paragraph)

    def close(self) -> None:
        """Close the underlying HTTP client.  Safe to call multiple times."""
        self._client.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_daemon_running(self) -> None:
        """Raise AttributionError with a clear message if Ollama is not reachable."""
        try:
            resp = self._client.get(self.base_url + _HEALTH_ENDPOINT)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise AttributionError(
                f"Ollama is not running at {self.base_url}. Start it with: ollama serve"
            ) from exc

    def _call_ollama(self, user_message: str) -> dict[str, Any]:
        """
        POST to the Ollama /api/chat endpoint and return the parsed JSON body.

        ``format: "json"`` activates Ollama's constrained decoding — the model
        is forced to emit only tokens that keep the output syntactically valid
        JSON (like a grammar checker that blocks every illegal character).
        """
        payload = {
            "model": self.model,
            "format": "json",  # constrained decoding — guarantees valid JSON output
            "stream": False,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        }
        try:
            resp = self._client.post(self.base_url + _CHAT_ENDPOINT, json=payload)
            resp.raise_for_status()
            return resp.json()  # type: ignore[return-value]
        except httpx.HTTPStatusError as exc:
            raise _ParseError(f"HTTP {exc.response.status_code} from Ollama") from exc
        except httpx.HTTPError as exc:
            raise _ParseError(f"HTTP error contacting Ollama: {exc}") from exc

    def _parse_response(self, body: dict[str, Any], paragraph: str) -> list[_SpanDict]:
        """
        Validate and extract the spans list from the Ollama response body.

        Raises ``_ParseError`` (internal) on any structural problem so the
        caller can retry without crashing.
        """
        try:
            content_str: str = body["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise _ParseError(f"Unexpected Ollama response shape: {exc}") from exc

        try:
            data = json.loads(content_str)
        except json.JSONDecodeError as exc:
            raise _ParseError(f"Model returned invalid JSON: {exc}") from exc

        if not isinstance(data, dict) or "spans" not in data:
            keys = list(data.keys()) if isinstance(data, dict) else type(data)
            raise _ParseError(f"JSON missing 'spans' key: keys={keys}")

        raw_spans = data["spans"]
        if not isinstance(raw_spans, list):
            raise _ParseError(f"'spans' is not a list: {type(raw_spans)}")

        valid_kinds = {"dialogue", "narration", "thought"}
        spans: list[_SpanDict] = []
        for i, span in enumerate(raw_spans):
            if not isinstance(span, dict):
                raise _ParseError(f"Span {i} is not a dict: {type(span)}")
            missing = {"text", "speaker", "kind"} - span.keys()
            if missing:
                raise _ParseError(f"Span {i} missing keys: {missing}")
            if span["kind"] not in valid_kinds:
                raise _ParseError(f"Span {i} has invalid kind: {span['kind']!r}")
            # Coerce speaker to str; trim whitespace.
            spans.append(
                {
                    "text": str(span["text"]).strip(),
                    "speaker": str(span["speaker"]).strip(),
                    "kind": str(span["kind"]),
                }
            )

        if not spans:
            raise _ParseError("Model returned an empty spans list")

        # Sanity check: the concatenated text should approximately cover the paragraph.
        # We don't abort on mismatch — just warn — because LLMs may lightly paraphrase.
        reconstructed = "".join(s["text"] for s in spans)
        if len(reconstructed) < len(paragraph.strip()) * 0.5:
            logger.warning(
                "Reconstructed text is much shorter than the input paragraph "
                "(got %d chars, expected ≥%d).  Possible truncation.",
                len(reconstructed),
                len(paragraph.strip()) // 2,
            )

        return spans

    @staticmethod
    def _narrator_fallback(paragraph: str) -> list[_SpanDict]:
        """Return the full paragraph attributed to narrator (fallback path)."""
        return [{"text": paragraph, "speaker": "narrator", "kind": _NARRATOR_FALLBACK_KIND}]


# ---------------------------------------------------------------------------
# Internal exception — not part of the public API
# ---------------------------------------------------------------------------


class _ParseError(Exception):
    """Raised internally when the LLM response cannot be parsed or validated."""
