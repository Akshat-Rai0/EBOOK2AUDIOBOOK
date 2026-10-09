"""
Speaker attribution via a local Ollama server (M4 V2 implementation).

**How it works at a high level (V2):**
    1.  Pre-segmented text with quote markers [Q1], [Q2] is sent to the LLM.
    2.  The LLM assigns each quote to a speaker ID from a closed set.
    3.  Confidence buckets: high (0.95), medium (0.70), low (0.40).
    4.  ID-based attribution avoids positional index issues (fixes P3).

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

from ebook2audiobook.attribution.attributor import (
    AttributionError,
    Attributor,
    QuoteAttribution,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HEALTH_ENDPOINT = "/api/tags"
_CHAT_ENDPOINT = "/api/chat"
_GENERATE_ENDPOINT = "/api/generate"

# Number of seconds to wait between retries (doubles each attempt).
_RETRY_BASE_SLEEP = 0.5

# Confidence buckets
_CONFIDENCE_HIGH = 0.95
_CONFIDENCE_MEDIUM = 0.70
_CONFIDENCE_LOW = 0.40

# ---------------------------------------------------------------------------
# Prompt template (V1 legacy)
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
# Prompt template (V2)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_V2 = """You are a literary annotation assistant. Your only task is to
identify which character speaks each quoted line in a paragraph.

Rules:
- Each quote is marked with a unique ID like [Q1], [Q2], etc.
- Assign each quote ID to exactly one speaker from the allowed list.
- Allowed speakers: {speaker_list}
- Return ONLY a JSON object with a single key "attributions" whose value is a list
- Each attribution must have: "quote_id" (e.g., "Q1"), "speaker_id", and "confidence"
- Confidence levels: "high" (very certain), "medium" (fairly certain), "low" (uncertain)
- Use "unknown" if you genuinely cannot determine the speaker, or if the quote is unspoken thoughts

Example output:
{{
  "attributions": [
    {{"quote_id": "Q1", "speaker_id": "harry", "confidence": "high"}},
    {{"quote_id": "Q2", "speaker_id": "ron", "confidence": "medium"}}
  ]
}}"""

_USER_TEMPLATE_V2 = """Paragraph with quote markers:
{marked_paragraph}

Return JSON only."""


# ---------------------------------------------------------------------------
# OllamaAttributor V2
# ---------------------------------------------------------------------------


class OllamaAttributor(Attributor):
    """
    Speaker attribution using a local Ollama LLM server (V2: ID-based).

    The attributor is stateless between calls — no conversation history is kept.

    Parameters
    ----------
    model:
        Ollama model tag, e.g. ``"llama3.2:3b"`` or ``"mistral:7b"``.
    base_url:
        Base URL of the Ollama daemon (default ``http://localhost:11434``).
    max_retries:
        How many times to retry a malformed response before falling back.
    timeout:
        Per-request timeout in seconds.
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
        logger.debug("OllamaAttributor V2 ready: model=%s base_url=%s", model, base_url)
        self._check_daemon_running()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def attribute(
        self,
        paragraph: str,
        cast_so_far: list[str],
        context: str = "",
    ) -> list[dict]:
        """
        Legacy V1 method (deprecated - use attribute_quotes instead).

        Kept for backwards compatibility and contract tests.
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
                time.sleep(sleep_sec)

            try:
                raw = self._call_ollama(_SYSTEM_PROMPT, user_msg)
                spans = self._parse_response(raw, paragraph)
                return spans
            except _ParseError as exc:
                last_error = str(exc)
                logger.warning("Attribution attempt %d failed: %s", attempt + 1, exc)

        logger.warning(
            "Attribution failed after %d attempts. Falling back to narrator. Last error: %s",
            self.max_retries + 1,
            last_error,
        )
        return [{"text": paragraph, "speaker": "narrator", "kind": "narration"}]

    def attribute_quotes(
        self,
        marked_paragraph: str,
        allowed_speakers: list[str],
    ) -> list[QuoteAttribution]:
        """
        Attribute each marked quote in the paragraph to a speaker.

        Parameters
        ----------
        marked_paragraph:
            Paragraph text with quote markers like [Q1], [Q2].
        allowed_speakers:
            List of allowed speaker IDs (closed set).

        Returns
        -------
        list[QuoteAttribution]
            Attribution for each quote.
        """
        if not marked_paragraph.strip():
            return []

        speaker_list = ", ".join(allowed_speakers)
        system_prompt = _SYSTEM_PROMPT_V2.format(speaker_list=speaker_list)
        user_msg = _USER_TEMPLATE_V2.format(marked_paragraph=marked_paragraph.strip())

        last_error: str = ""
        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                sleep_sec = _RETRY_BASE_SLEEP * (2 ** (attempt - 1))
                logger.debug(
                    "Attribution retry %d/%d (sleep %.1fs)", attempt, self.max_retries, sleep_sec
                )
                time.sleep(sleep_sec)

            try:
                raw = self._call_ollama(system_prompt, user_msg)
                attributions = self._parse_response_v2(raw)
                return attributions
            except _ParseError as exc:
                last_error = str(exc)
                logger.warning("Attribution attempt %d failed: %s", attempt + 1, exc)

        # All retries exhausted — return empty list (fallback to rules/narrator)
        logger.warning(
            "Attribution failed after %d attempts. Last error: %s",
            self.max_retries + 1,
            last_error,
        )
        return []

    def close(self) -> None:
        """
        Close the underlying HTTP client and unload the model from Ollama.

        Safe to call multiple times.
        """
        self._client.close()
        # Evict model weights to free memory
        try:
            # Use a new client for the unload call since we just closed the main one
            unload_client = httpx.Client(timeout=10.0)
            payload = {
                "model": self.model,
                "keep_alive": 0,  # Evict immediately
            }
            unload_client.post(self.base_url + _GENERATE_ENDPOINT, json=payload)
            unload_client.close()
            logger.debug("Unloaded Ollama model: %s", self.model)
        except Exception as exc:
            logger.warning("Failed to unload Ollama model: %s", exc)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_daemon_running(self) -> None:
        """Raise AttributionError if Ollama is not reachable."""
        try:
            resp = self._client.get(self.base_url + _HEALTH_ENDPOINT)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise AttributionError(
                f"Ollama is not running at {self.base_url}. Start it with: ollama serve"
            ) from exc

    def _call_ollama(self, system_prompt: str, user_message: str) -> dict[str, Any]:
        """
        POST to the Ollama /api/chat endpoint and return the parsed JSON body.
        """
        payload = {
            "model": self.model,
            "format": "json",  # constrained decoding
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
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

    def _parse_response(self, body: dict[str, Any], paragraph: str) -> list[dict[str, str]]:
        """
        Validate and extract the spans list from the Ollama response body (V1).
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
        spans: list[dict[str, str]] = []
        for i, span in enumerate(raw_spans):
            if not isinstance(span, dict):
                raise _ParseError(f"Span {i} is not a dict: {type(span)}")
            missing = {"text", "speaker", "kind"} - span.keys()
            if missing:
                raise _ParseError(f"Span {i} missing keys: {missing}")
            if span["kind"] not in valid_kinds:
                raise _ParseError(f"Span {i} has invalid kind: {span['kind']!r}")
            spans.append(
                {
                    "text": str(span["text"]).strip(),
                    "speaker": str(span["speaker"]).strip(),
                    "kind": str(span["kind"]),
                }
            )

        if not spans:
            raise _ParseError("Model returned an empty spans list")

        return spans

    def _parse_response_v2(self, body: dict[str, Any]) -> list[QuoteAttribution]:
        """
        Validate and extract attributions from the Ollama response body (V2).
        """
        try:
            content_str: str = body["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise _ParseError(f"Unexpected Ollama response shape: {exc}") from exc

        try:
            data = json.loads(content_str)
        except json.JSONDecodeError as exc:
            raise _ParseError(f"Model returned invalid JSON: {exc}") from exc

        if not isinstance(data, dict) or "attributions" not in data:
            keys = list(data.keys()) if isinstance(data, dict) else type(data)
            raise _ParseError(f"JSON missing 'attributions' key: keys={keys}")

        raw_attributions = data["attributions"]
        if not isinstance(raw_attributions, list):
            raise _ParseError(f"'attributions' is not a list: {type(raw_attributions)}")

        attributions: list[QuoteAttribution] = []
        for i, attr in enumerate(raw_attributions):
            if not isinstance(attr, dict):
                raise _ParseError(f"Attribution {i} is not a dict: {type(attr)}")
            missing = {"quote_id", "speaker_id", "confidence"} - attr.keys()
            if missing:
                raise _ParseError(f"Attribution {i} missing keys: {missing}")

            # Convert confidence string to float
            conf_str = str(attr["confidence"]).lower()
            if conf_str == "high":
                confidence = _CONFIDENCE_HIGH
            elif conf_str == "medium":
                confidence = _CONFIDENCE_MEDIUM
            elif conf_str == "low":
                confidence = _CONFIDENCE_LOW
            else:
                # Try to parse as float
                try:
                    confidence = float(conf_str)
                except ValueError:
                    confidence = _CONFIDENCE_LOW  # Default to low

            attributions.append(
                QuoteAttribution(
                    quote_id=str(attr["quote_id"]).strip(),
                    speaker_id=str(attr["speaker_id"]).strip(),
                    confidence=confidence,
                )
            )

        if not attributions:
            raise _ParseError("Model returned an empty attributions list")

        return attributions


# ---------------------------------------------------------------------------
# Internal exception — not part of the public API
# ---------------------------------------------------------------------------


class _ParseError(Exception):
    """Raised internally when the LLM response cannot be parsed or validated."""
