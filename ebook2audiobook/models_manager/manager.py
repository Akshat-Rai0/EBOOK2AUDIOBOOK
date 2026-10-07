"""
ModelManager — loads at most one heavy model at a time.

**Why this constraint exists:**
XTTS-v2 (TTS) and the attribution LLM are both large models that consume
several gigabytes of RAM/VRAM.  Holding them both in memory simultaneously
would exceed the hardware floor.  ModelManager enforces sequential loading:
load LLM → run attribution → unload LLM → load TTS → run synthesis.

Analogy: like a single recording booth that can only have one piece of
equipment set up at a time — the microphone is swapped out between takes.
"""

from __future__ import annotations

import gc
import logging
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Ensures at most one heavy model is loaded at any given time.

    Singleton pattern: use `ModelManager.instance()` to get the shared instance.

    Usage::

        manager = ModelManager.instance()
        with manager.load("xtts", loader_callable) as tts:
            wav = tts.synthesize(text, voice)
        # model is unloaded here
    """

    _instance: ModelManager | None = None

    def __init__(self) -> None:
        if ModelManager._instance is not None:
            raise RuntimeError("Use ModelManager.instance() to get the singleton")
        self._current_model_name: str | None = None
        self._current_model: Any = None

    @classmethod
    def instance(cls) -> ModelManager:
        """Return the singleton ModelManager instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def loaded_model_name(self) -> str | None:
        """Name of the currently loaded model, or None."""
        return self._current_model_name

    def is_loaded(self, name: str) -> bool:
        """Return True if *name* is currently loaded."""
        return self._current_model_name == name

    @contextmanager
    def load(self, name: str, loader: Callable[[], Any]):
        """
        Load a model via *loader* and yield it.

        Refuses to load if another model is already loaded (caller must unload first).

        Parameters
        ----------
        name:
            Model identifier (e.g., "vits", "xtts").
        loader:
            Callable that returns the loaded model object.

        Yields
        ------
        Any
            The loaded model object.

        Raises
        ------
        RuntimeError
            If a different model is already loaded.
        """
        if self._current_model is not None and self._current_model_name != name:
            raise RuntimeError(
                f"Cannot load {name}: {self._current_model_name} is already loaded. "
                f"Call unload() first."
            )

        if self._current_model is None:
            logger.info("Loading model: %s", name)
            self._current_model = loader()
            self._current_model_name = name

        try:
            yield self._current_model
        finally:
            # Unload only if we were the ones who loaded it
            # (allows nested loads of the same model)
            if self._current_model_name == name:
                self.unload()

    def unload(self) -> None:
        """Explicitly unload the current model and free memory."""
        if self._current_model is not None:
            logger.info("Unloading model: %s", self._current_model_name)
            self._current_model = None
            self._current_model_name = None
            gc.collect()

            # Clear PyTorch caches if available
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.debug("Cleared CUDA cache")
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                    logger.debug("Cleared MPS cache")
            except ImportError:
                pass

    def unload_ollama(self, model_tag: str, base_url: str = "http://localhost:11434") -> None:
        """
        Unload an Ollama model from the daemon.

        Uses the Ollama API to tell the daemon to unload the model weights.

        Parameters
        ----------
        model_tag:
            Ollama model tag (e.g., "llama3.2:3b").
        base_url:
            Base URL for the Ollama daemon.
        """
        try:
            with httpx.Client(timeout=10.0) as client:
                # Send a keep_alive=0 request to unload
                response = client.post(
                    f"{base_url}/api/generate",
                    json={
                        "model": model_tag,
                        "prompt": "",
                        "keep_alive": 0,
                    },
                )
                logger.info(f"Requested Ollama to unload {model_tag}: {response.status_code}")
        except Exception as exc:
            logger.warning(f"Failed to unload Ollama model {model_tag}: {exc}")

    def is_ollama_loaded(self, model_tag: str, base_url: str = "http://localhost:11434") -> bool:
        """
        Check if an Ollama model is currently loaded in the daemon.

        Parameters
        ----------
        model_tag:
            Ollama model tag (e.g., "llama3.2:3b").
        base_url:
            Base URL for the Ollama daemon.

        Returns
        -------
        bool
            True if the model is currently loaded.
        """
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{base_url}/api/ps")
                if response.status_code == 200:
                    data = response.json()
                    running_models = [m.get("name") for m in data.get("models", [])]
                    return model_tag in running_models
        except Exception as exc:
            logger.warning(f"Failed to check Ollama loaded models: {exc}")
        return False

