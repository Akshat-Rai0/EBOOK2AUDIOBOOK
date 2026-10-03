"""
TTS engine factory — creates TTS engine instances.

**Why a factory:**
Different engines have different dependencies (VITS/XTTS require coqui-tts,
FakeTTS has no dependencies). The factory handles lazy imports and provides
a unified interface for getting the right engine.

Analogy: like a casting director who chooses the right actor for a role based
on availability and requirements.
"""

from __future__ import annotations

import logging

from ebook2audiobook.models_manager.manager import ModelManager
from ebook2audiobook.tts.engine import TTSEngine, TTSError
from ebook2audiobook.tts.fake_tts import FakeTTS

logger = logging.getLogger(__name__)


def get_engine(
    name: str,
    *,
    device: str = "cpu",
    model_manager: ModelManager | None = None,
) -> TTSEngine:
    """
    Get a TTS engine instance by name.

    Parameters
    ----------
    name:
        Engine name: "fake", "vits", or "xtts".
    device:
        Device to run on: "cpu", "cuda", or "mps".
        Ignored for FakeTTS.
    model_manager:
        Optional ModelManager instance for coordinating model loading.
        If None, engines will load models directly (not recommended for production).

    Returns
    -------
    TTSEngine
        An instance of the requested engine.

    Raises
    ------
    TTSError
        If the engine is not available or model weights are missing.
    ValueError
        If the engine name is unknown.
    """
    if name == "fake":
        return FakeTTS()

    if name == "vits":
        try:
            from ebook2audiobook.tts.vits import VITS

            return VITS(device=device)
        except ImportError as exc:
            raise TTSError(
                "VITS engine requires coqui-tts. Run: uv sync --extra tts"
            ) from exc

    if name == "xtts":
        try:
            from ebook2audiobook.tts.xtts import XTTS

            return XTTS(device=device)
        except ImportError as exc:
            raise TTSError(
                "XTTS engine requires coqui-tts. Run: uv sync --extra tts"
            ) from exc

    raise ValueError(f"Unknown engine: {name}. Available: fake, vits, xtts")
