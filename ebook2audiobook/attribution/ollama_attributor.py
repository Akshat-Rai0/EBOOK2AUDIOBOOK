"""
Ollama attribution backend — stub for M0.

The concrete implementation is deferred to Week 4 (M3) after the benchmark
decides which model size achieves ≥ 90% speaker attribution accuracy on the
3-chapter hand-labelled test set.

**What Ollama does:**
Ollama is a local server that runs LLM models (like Llama, Mistral, etc.)
on your machine.  You start it with ``ollama serve`` and pull a model with
``ollama pull <model>``.  It exposes a simple HTTP API that returns JSON.
Analogy: think of Ollama as a translator agency you call on the phone —
you send text, it sends back structured answers, all without any internet.

**JSON mode:**
Ollama's ``format: "json"`` parameter forces the model to always emit valid JSON.
This is constrained decoding at the API level — no GBNF grammar file needed.

Decision recorded: 2026-10-01 (see docs/DECISIONS.md).
"""

from __future__ import annotations

import logging

from ebook2audiobook.attribution.attributor import Attributor

logger = logging.getLogger(__name__)

# Narrator-fallback template — returned when attribution fails after retries.
_NARRATOR_FALLBACK = [{"text": "", "speaker": "narrator", "kind": "narration"}]


class OllamaAttributor(Attributor):
    """
    Speaker attribution via a local Ollama server.

    **Status:** STUB — not implemented until M3 (Week 4).

    Raises ``NotImplementedError`` on any call so callers know to use the
    narrator-only path or wait for M3.
    """

    def __init__(
        self,
        model: str = "llama3.2:3b",
        base_url: str = "http://localhost:11434",
        max_retries: int = 2,
    ) -> None:
        """
        Parameters
        ----------
        model:
            Ollama model tag.  3B-class 4-bit quantised models are ~2–3 GB.
            Final choice decided in Week 4 after benchmarking.
        base_url:
            Ollama server URL.  Defaults to the local Ollama default.
        max_retries:
            How many times to retry before falling back to the narrator.
        """
        self.model = model
        self.base_url = base_url
        self.max_retries = max_retries
        logger.debug("OllamaAttributor initialised (STUB): model=%s", model)

    def attribute(
        self,
        paragraph: str,
        cast_so_far: list[str],
        context: str = "",
    ) -> list[dict]:
        """
        NOT YET IMPLEMENTED — milestone M3.

        The full implementation will:
        1. Build a prompt asking the LLM to split the paragraph into spans,
           each with a speaker and kind.
        2. Call the Ollama API with ``format: "json"`` (constrained decoding).
        3. Validate the response against a JSON schema.
        4. Retry up to ``max_retries`` times on invalid JSON.
        5. Fall back to narrator attribution on persistent failure.
        """
        raise NotImplementedError(
            "OllamaAttributor is a stub — implement in M3 (Week 4). "
            "Use a narrator-only pipeline for now."
        )
