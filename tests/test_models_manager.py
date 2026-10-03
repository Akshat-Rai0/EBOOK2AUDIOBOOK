"""Tests for ModelManager singleton and load/unload behavior."""

import pytest

from ebook2audiobook.models_manager.manager import ModelManager


class TestModelManager:
    def test_singleton_pattern(self):
        """ModelManager should be a singleton."""
        manager1 = ModelManager.instance()
        manager2 = ModelManager.instance()
        assert manager1 is manager2

    def test_direct_instantiation_raises(self):
        """Direct instantiation should raise RuntimeError."""
        with pytest.raises(RuntimeError, match="Use ModelManager.instance"):
            ModelManager()

    def test_load_and_unload(self):
        """Test basic load/unload cycle."""
        manager = ModelManager.instance()

        def dummy_loader():
            return {"model": "data"}

        with manager.load("test", dummy_loader) as model:
            assert model == {"model": "data"}
            assert manager.loaded_model_name == "test"

        # After context manager, model should be unloaded
        assert manager.loaded_model_name is None

    def test_refuse_double_load_different_model(self):
        """Refuse to load a different model while one is loaded."""
        manager = ModelManager.instance()

        def loader1():
            return {"model": "data1"}

        def loader2():
            return {"model": "data2"}

        with manager.load("model1", loader1):
            assert manager.loaded_model_name == "model1"

            # Try to load a different model while one is loaded
            with pytest.raises(RuntimeError, match="model1 is already loaded"):
                with manager.load("model2", loader2):
                    pass

    def test_allow_re_load_same_model(self):
        """Allow re-loading the same model (nested loads)."""
        manager = ModelManager.instance()

        def loader():
            return {"model": "data"}

        with manager.load("test", loader) as model1:
            assert model1 == {"model": "data"}

            # Nested load of same model should work
            with manager.load("test", loader) as model2:
                assert model2 == {"model": "data"}

        # After both contexts, model should be unloaded
        assert manager.loaded_model_name is None

    def test_explicit_unload(self):
        """Test explicit unload() call."""
        manager = ModelManager.instance()

        def loader():
            return {"model": "data"}

        with manager.load("test", loader):
            assert manager.loaded_model_name == "test"
            manager.unload()
            assert manager.loaded_model_name is None

    def test_is_loaded(self):
        """Test is_loaded() method."""
        manager = ModelManager.instance()

        def loader():
            return {"model": "data"}

        assert not manager.is_loaded("test")

        with manager.load("test", loader):
            assert manager.is_loaded("test")
            assert not manager.is_loaded("other")

        assert not manager.is_loaded("test")

    def test_unload_when_nothing_loaded(self):
        """unload() should be safe when nothing is loaded."""
        manager = ModelManager.instance()
        manager.unload()  # Should not raise
        assert manager.loaded_model_name is None
