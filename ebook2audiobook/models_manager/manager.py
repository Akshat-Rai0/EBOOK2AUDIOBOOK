"""
ModelManager — loads at most one heavy model at a time.

**Why this constraint exists:**
XTTS-v2 (TTS) and the attribution LLM are both large models that consume
several gigabytes of RAM/VRAM.  Holding them both in memory simultaneously
would exceed the hardware floor.  ModelManager enforces sequential loading:
load LLM → run attribution → unload LLM → load TTS → run synthesis.

Analogy: like a single recording booth that can only have one piece of
equipment set up at a time — the microphone is swapped out between takes.

**Status:** STUB — the actual model-loading code lives in TTS adapters and the
Ollama attributor.  ModelManager will orchestrate them in M2+.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Ensures at most one heavy model is loaded at any given time.

    Usage (future M2+)::

        manager = ModelManager()
        with manager.load("xtts", config) as tts:
            wav = tts.synthesize(text, voice)
        # model is unloaded here
    """

    def __init__(self) -> None:
        self._current_model_name: str | None = None
        self._current_model: Any = None

    @property
    def loaded_model_name(self) -> str | None:
        """Name of the currently loaded model, or None."""
        return self._current_model_name

    def unload(self) -> None:
        """Explicitly unload the current model and free memory."""
        if self._current_model is not None:
            logger.info("Unloading model: %s", self._current_model_name)
            # Future: call model.teardown() or del and gc.collect()
            self._current_model = None
            self._current_model_name = None

    def is_loaded(self, name: str) -> bool:
        """Return True if *name* is currently loaded."""
        return self._current_model_name == name

    # M2+ will add: load(name, config) -> ContextManager
