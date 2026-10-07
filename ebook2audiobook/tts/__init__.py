"""
TTS engine adapters and utilities.

This package contains:
- engine.py: Abstract base class for TTS engines
- fake_tts.py: No-op engine for testing
- normalise.py: Text normalisation for synthesis
- vits.py: VITS-VCTK adapter
- xtts.py: XTTS-v2 adapter
- factory.py: Engine factory
"""

from ebook2audiobook.tts.engine import TTSEngine, TTSError
from ebook2audiobook.tts.factory import get_engine
from ebook2audiobook.tts.fake_tts import FakeTTS

__all__ = ["TTSEngine", "TTSError", "FakeTTS", "get_engine"]
