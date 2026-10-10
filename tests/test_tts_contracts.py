"""Contract tests for TTS engines (FakeTTS, VITS, XTTS).

These tests verify that each engine implements the TTSEngine contract correctly.
VITS and XTTS tests are marked with `models` and skipped by default (require downloaded models).
"""

import pytest

from ebook2audiobook.models.cast import VoiceRef
from ebook2audiobook.tts.engine import TTSError
from ebook2audiobook.tts.fake_tts import FakeTTS


class TestFakeTTSContract:
    """Contract tests for FakeTTS (always runs)."""

    def test_engine_name_property(self):
        engine = FakeTTS()
        assert engine.engine_name == "fake"

    def test_max_chars_property(self):
        engine = FakeTTS()
        assert engine.max_chars == 100_000

    def test_sample_rate_property(self):
        engine = FakeTTS()
        assert engine.sample_rate == 22050

    def test_list_voices_returns_dicts(self):
        engine = FakeTTS()
        voices = engine.list_voices()
        assert isinstance(voices, list)
        assert len(voices) > 0
        for voice in voices:
            assert "id" in voice
            assert "name" in voice

    def test_synthesize_returns_wav_bytes(self):
        engine = FakeTTS()
        wav = engine.synthesize("Hello", VoiceRef(engine="fake", voice_id="v1"))
        assert isinstance(wav, bytes)
        assert len(wav) > 44  # WAV header is 44 bytes

    def test_synthesize_with_string_voice_id(self):
        """Engine should accept string voice ID as well as VoiceRef."""
        engine = FakeTTS()
        wav = engine.synthesize("Hello", "v1")
        assert isinstance(wav, bytes)

    def test_exceeds_max_chars_raises(self):
        engine = FakeTTS()
        with pytest.raises(ValueError, match="exceeds"):
            engine.synthesize("x" * 100_001, VoiceRef(engine="fake", voice_id="v1"))

    def test_empty_text_allowed_for_faketts(self):
        """FakeTTS is lenient: empty text produces a short silence WAV."""
        engine = FakeTTS()
        wav = engine.synthesize("", VoiceRef(engine="fake", voice_id="v1"))
        assert isinstance(wav, bytes)
        assert len(wav) > 44  # Should still produce a WAV (short silence)


@pytest.mark.models
class TestVITSContract:
    """Contract tests for VITS (requires downloaded models, skipped by default)."""

    @pytest.fixture(autouse=True)
    def skip_if_no_coqui(self):
        import importlib.util

        if importlib.util.find_spec("TTS") is None:
            pytest.skip("coqui-tts not installed")

    def test_engine_name_property(self):
        from ebook2audiobook.tts.vits import VITS

        engine = VITS(device="cpu")
        assert engine.engine_name == "vits"

    def test_max_chars_property(self):
        from ebook2audiobook.tts.vits import VITS

        engine = VITS(device="cpu")
        assert engine.max_chars == 500  # decided in DECISIONS.md; update both if it changes

    def test_sample_rate_property(self):
        from ebook2audiobook.tts.vits import VITS

        engine = VITS(device="cpu")
        assert engine.sample_rate == 22050

    def test_list_voices_returns_dicts(self):
        from ebook2audiobook.tts.vits import VITS

        engine = VITS(device="cpu")
        voices = engine.list_voices()
        assert isinstance(voices, list)
        assert len(voices) > 0
        for voice in voices:
            assert "id" in voice
            assert "name" in voice

    def test_synthesize_returns_wav_bytes(self):
        from ebook2audiobook.tts.vits import VITS

        engine = VITS(device="cpu")
        voices = engine.list_voices()
        voice_id = voices[0]["id"]

        wav = engine.synthesize("Hello world", voice_id)
        assert isinstance(wav, bytes)
        assert len(wav) > 44

    def test_synthesize_with_string_voice_id(self):
        from ebook2audiobook.tts.vits import VITS

        engine = VITS(device="cpu")
        voices = engine.list_voices()
        voice_id = voices[0]["id"]

        wav = engine.synthesize("Hello world", voice_id)
        assert isinstance(wav, bytes)

    def test_exceeds_max_chars_raises(self):
        from ebook2audiobook.tts.vits import VITS

        engine = VITS(device="cpu")
        with pytest.raises(ValueError, match="exceeds"):
            engine.synthesize(
                "x" * (engine.max_chars + 1),
                VoiceRef(engine="vits", voice_id="p225"),
            )

    def test_empty_text_raises(self):
        from ebook2audiobook.tts.vits import VITS

        engine = VITS(device="cpu")
        with pytest.raises(TTSError, match="empty"):
            engine.synthesize("", VoiceRef(engine="vits", voice_id="p225"))

    def test_narrator_mapping(self):
        from ebook2audiobook.tts.vits import VITS

        engine = VITS(device="cpu")
        # "narrator" should map to configured default (p225)
        wav = engine.synthesize("Test", "narrator")
        assert isinstance(wav, bytes)


@pytest.mark.models
class TestXTTSContract:
    """Contract tests for XTTS (requires downloaded models, skipped by default)."""

    @pytest.fixture(autouse=True)
    def skip_if_no_coqui(self):
        import importlib.util

        if importlib.util.find_spec("TTS") is None:
            pytest.skip("coqui-tts not installed")

    def test_engine_name_property(self):
        from ebook2audiobook.tts.xtts import XTTS

        engine = XTTS(device="cpu")
        assert engine.engine_name == "xtts"

    def test_max_chars_property(self):
        from ebook2audiobook.tts.xtts import XTTS

        engine = XTTS(device="cpu")
        assert engine.max_chars == 255

    def test_sample_rate_property(self):
        from ebook2audiobook.tts.xtts import XTTS

        engine = XTTS(device="cpu")
        assert engine.sample_rate == 24000

    def test_list_voices_returns_dicts(self):
        from ebook2audiobook.tts.xtts import XTTS

        engine = XTTS(device="cpu")
        voices = engine.list_voices()
        assert isinstance(voices, list)
        assert len(voices) > 0
        for voice in voices:
            assert "id" in voice
            assert "name" in voice

    def test_synthesize_returns_wav_bytes(self):
        from ebook2audiobook.tts.xtts import XTTS

        engine = XTTS(device="cpu")
        voices = engine.list_voices()
        voice_id = voices[0]["id"]

        wav = engine.synthesize("Hello world", voice_id)
        assert isinstance(wav, bytes)
        assert len(wav) > 44

    def test_synthesize_with_string_voice_id(self):
        from ebook2audiobook.tts.xtts import XTTS

        engine = XTTS(device="cpu")
        voices = engine.list_voices()
        voice_id = voices[0]["id"]

        wav = engine.synthesize("Hello world", voice_id)
        assert isinstance(wav, bytes)

    def test_exceeds_max_chars_raises(self):
        from ebook2audiobook.tts.xtts import XTTS

        engine = XTTS(device="cpu")
        with pytest.raises(ValueError, match="exceeds"):
            engine.synthesize("x" * 256, VoiceRef(engine="xtts", voice_id="Claribel Dervla"))

    def test_empty_text_raises(self):
        from ebook2audiobook.tts.xtts import XTTS

        engine = XTTS(device="cpu")
        with pytest.raises(TTSError, match="empty"):
            engine.synthesize("", VoiceRef(engine="xtts", voice_id="Claribel Dervla"))
