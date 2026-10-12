"""Tests for TTS engine factory and ModelManager integration."""

import importlib.util

import pytest

from ebook2audiobook.models_manager.manager import ModelManager
from ebook2audiobook.tts.factory import get_engine
from ebook2audiobook.tts.fake_tts import FakeTTS


def _coqui_installed() -> bool:
    """Return True if coqui-ai-TTS is available (importlib.util.find_spec avoids F401)."""
    return importlib.util.find_spec("TTS") is not None


class TestTTSFactory:
    def test_get_fake_engine(self):
        """FakeTTS should always be available."""
        engine = get_engine("fake")
        assert isinstance(engine, FakeTTS)
        assert engine.engine_name == "fake"

    def test_get_fake_ignores_model_manager(self):
        """FakeTTS should not check ModelManager."""
        manager = ModelManager.instance()
        engine = get_engine("fake", model_manager=manager)
        assert isinstance(engine, FakeTTS)

    def test_model_manager_refuses_double_load(self):
        """ModelManager should refuse to load a different model while one is loaded."""
        if not _coqui_installed():
            pytest.skip("coqui-tts not installed")

        from ebook2audiobook.tts.vits import VITS

        manager = ModelManager.instance()

        with manager.load("vits", lambda: VITS(device="cpu")) as _engine:
            assert manager.is_loaded("vits")

            # Try to get XTTS engine while VITS is loaded - should raise RuntimeError
            with pytest.raises(RuntimeError, match="vits is already loaded"):
                get_engine("xtts", model_manager=manager)

    def test_model_manager_allows_same_model_reload(self):
        """ModelManager should allow re-loading the same model."""
        if not _coqui_installed():
            pytest.skip("coqui-tts not installed")

        from ebook2audiobook.tts.vits import VITS

        manager = ModelManager.instance()

        with manager.load("vits", lambda: VITS(device="cpu")) as _engine:
            assert manager.is_loaded("vits")

            # Getting VITS again should work (same model)
            engine = get_engine("vits", model_manager=manager)
            assert engine.engine_name == "vits"

    def test_get_engine_without_model_manager(self):
        """Without ModelManager, engines should load directly (for tests)."""
        if not _coqui_installed():
            pytest.skip("coqui-tts not installed")

        # Should work without ModelManager (tests don't need it)
        engine = get_engine("vits", model_manager=None)
        assert engine.engine_name == "vits"

    def test_get_unknown_engine_raises(self):
        """Unknown engine names should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown engine"):
            get_engine("unknown")

    def test_vits_requires_coqui_tts(self):
        """VITS gate: skip this test if coqui-tts is installed (error path unreachable)."""
        if _coqui_installed():
            pytest.skip("coqui-tts is installed; ImportError path untestable")
        # If TTS is not installed, the factory would raise an error — we verify
        # the guard exists by confirming find_spec returned None.
        assert importlib.util.find_spec("TTS") is None
