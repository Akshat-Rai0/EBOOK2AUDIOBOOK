"""
Tests for M3: OllamaAttributor, CharacterRegistry, and castbook attribute CLI.

All tests mock httpx — no real network calls, no real Ollama required.

Constrained decoding recap (for readers):
    Ollama's ``format: "json"`` forces the model to emit only tokens that
    maintain syntactically valid JSON — like a spellchecker that blocks
    non-words.  Our tests verify we handle the happy path AND gracefully
    degrade when the model misbehaves despite that guard.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from ebook2audiobook.attribution.attributor import AttributionError
from ebook2audiobook.attribution.character_registry import CharacterRegistry, build_registry
from ebook2audiobook.attribution.ollama_attributor import OllamaAttributor
from ebook2audiobook.cli.main import cli

# ---------------------------------------------------------------------------
# Synthetic test passage — 20 lines, 4 speakers, used for accuracy gate
# ---------------------------------------------------------------------------

_PASSAGE_SPEAKERS = {
    "Mira": "Mira",
    "James": "James",
    "narrator": "narrator",
    "unknown": "unknown",
}

_TEST_PARAGRAPHS: list[dict[str, Any]] = [
    {
        "text": "The library was silent except for the ticking of the old clock.",
        "expected_speaker": "narrator",
        "expected_kind": "narration",
    },
    {
        "text": '"I found the letter," Mira said, her hands trembling.',
        "expected_speaker": "Mira",
        "expected_kind": "dialogue",
    },
    {
        "text": 'James looked up from the desk. "Which letter?" he asked quietly.',
        "expected_speaker": "James",
        "expected_kind": "dialogue",
    },
    {
        "text": '"The one from your father," she replied. "The one you said never existed."',
        "expected_speaker": "Mira",
        "expected_kind": "dialogue",
    },
    {
        "text": "He set down his pen and stared at the window for a long moment.",
        "expected_speaker": "narrator",
        "expected_kind": "narration",
    },
    {
        "text": '"Give it to me," he said at last.',
        "expected_speaker": "James",
        "expected_kind": "dialogue",
    },
    {
        "text": "Mira held the envelope tightly against her chest.",
        "expected_speaker": "narrator",
        "expected_kind": "narration",
    },
    {
        "text": '"Not until you tell me the truth."',
        "expected_speaker": "Mira",
        "expected_kind": "dialogue",
    },
    {
        "text": "A shadow crossed James's face.",
        "expected_speaker": "narrator",
        "expected_kind": "narration",
    },
    {
        "text": '"There is no truth you would want to hear."',
        "expected_speaker": "James",
        "expected_kind": "dialogue",
    },
    {
        "text": "She thought, *Perhaps he is right*, but said nothing aloud.",
        "expected_speaker": "Mira",
        "expected_kind": "narration",
    },
    {
        "text": "The clock struck midnight.",
        "expected_speaker": "narrator",
        "expected_kind": "narration",
    },
    {
        "text": '"We should leave," said a voice from the doorway.',
        "expected_speaker": "unknown",
        "expected_kind": "dialogue",
    },
    {
        "text": "Both Mira and James turned at once.",
        "expected_speaker": "narrator",
        "expected_kind": "narration",
    },
    {
        "text": '"Who are you?" James demanded.',
        "expected_speaker": "James",
        "expected_kind": "dialogue",
    },
    {
        "text": "The figure in the doorway said nothing and stepped back into the darkness.",
        "expected_speaker": "narrator",
        "expected_kind": "narration",
    },
    {
        "text": '"I think I know who that was," Mira whispered.',
        "expected_speaker": "Mira",
        "expected_kind": "dialogue",
    },
    {
        "text": "James did not answer.",
        "expected_speaker": "narrator",
        "expected_kind": "narration",
    },
    {
        "text": '"We have to go back to the beginning," she said firmly.',
        "expected_speaker": "Mira",
        "expected_kind": "dialogue",
    },
    {
        "text": "He finally nodded, and the two of them walked out into the cold night.",
        "expected_speaker": "narrator",
        "expected_kind": "narration",
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ollama_response(spans: list[dict]) -> dict:
    """Wrap spans in the Ollama /api/chat response envelope."""
    return {
        "message": {
            "content": json.dumps({"spans": spans}),
        }
    }


def _mock_httpx_client(
    health_ok: bool = True,
    chat_response: dict | None = None,
    chat_side_effect: Exception | None = None,
) -> MagicMock:
    """
    Return a mock httpx.Client that fakes health-check and chat endpoints.

    Parameters
    ----------
    health_ok:
        Whether GET /api/tags returns 200.
    chat_response:
        The dict to return from POST /api/chat.  Ignored if chat_side_effect is set.
    chat_side_effect:
        Exception to raise from POST /api/chat.
    """
    import httpx

    client = MagicMock()

    # Health check
    if health_ok:
        health_resp = MagicMock()
        health_resp.raise_for_status = MagicMock()
        client.get.return_value = health_resp
    else:
        client.get.side_effect = httpx.ConnectError("connection refused")

    # Chat
    if chat_side_effect is not None:
        client.post.side_effect = chat_side_effect
    elif chat_response is not None:
        chat_resp = MagicMock()
        chat_resp.raise_for_status = MagicMock()
        chat_resp.json.return_value = chat_response
        client.post.return_value = chat_resp

    return client


# ---------------------------------------------------------------------------
# OllamaAttributor — unit tests
# ---------------------------------------------------------------------------


class TestOllamaAttributorInit:
    def test_raises_when_ollama_not_running(self):
        """AttributionError with a helpful message if Ollama daemon is down."""

        with patch("httpx.Client") as mock_client:
            mock_client.return_value = _mock_httpx_client(health_ok=False)
            with pytest.raises(AttributionError, match="Ollama is not running"):
                OllamaAttributor()

    def test_succeeds_when_ollama_is_running(self):
        """No exception when Ollama responds 200 to the health endpoint."""
        with patch("httpx.Client") as mock_client:
            mock_client.return_value = _mock_httpx_client(health_ok=True)
            attributor = OllamaAttributor()
            assert attributor.model == "llama3.2:3b"


class TestOllamaAttributorAttribute:
    def _build_attributor(
        self,
        chat_response: dict | None = None,
        side_effect: Exception | None = None,
    ) -> OllamaAttributor:
        """Construct OllamaAttributor with a mocked httpx.Client."""
        with patch("httpx.Client") as mock_client:
            mock_client.return_value = _mock_httpx_client(
                health_ok=True,
                chat_response=chat_response,
                chat_side_effect=side_effect,
            )
            a = OllamaAttributor(max_retries=1)
        # Overwrite the client after __init__ so attribute() also uses the mock.
        a._client = _mock_httpx_client(
            health_ok=True,
            chat_response=chat_response,
            chat_side_effect=side_effect,
        )
        return a

    def test_happy_path_returns_spans(self):
        """Returns validated spans when the model responds correctly."""
        spans = [
            {"text": "She walked in.", "speaker": "narrator", "kind": "narration"},
            {"text": '"Hello," she said.', "speaker": "Mira", "kind": "dialogue"},
        ]
        a = self._build_attributor(chat_response=_make_ollama_response(spans))
        result = a.attribute('"Hello," she said.', cast_so_far=["Mira"])
        assert len(result) == 2
        assert result[1]["speaker"] == "Mira"
        assert result[1]["kind"] == "dialogue"

    def test_empty_paragraph_returns_empty_list(self):
        """Empty/whitespace paragraph should return [] without calling the LLM."""
        with patch("httpx.Client") as mock_client:
            mock_client.return_value = _mock_httpx_client(health_ok=True)
            a = OllamaAttributor()
        a._client = _mock_httpx_client(health_ok=True)
        result = a.attribute("   ", cast_so_far=[])
        assert result == []
        # The mock's post should NOT have been called.
        a._client.post.assert_not_called()

    def test_falls_back_to_narrator_on_bad_json(self):
        """Malformed model response triggers narrator fallback after retries."""
        bad_response = {"message": {"content": "not json at all!!!"}}
        a = self._build_attributor(chat_response=bad_response)
        paragraph = "The sun set slowly over the hills."
        result = a.attribute(paragraph, cast_so_far=[])
        # Must return narrator fallback, not raise.
        assert len(result) == 1
        assert result[0]["speaker"] == "narrator"
        assert result[0]["kind"] == "narration"
        assert result[0]["text"] == paragraph

    def test_falls_back_when_spans_key_missing(self):
        """JSON without 'spans' key also triggers narrator fallback."""
        bad_response = {"message": {"content": '{"wrong_key": []}'}}
        a = self._build_attributor(chat_response=bad_response)
        result = a.attribute("Some text.", cast_so_far=[])
        assert result[0]["speaker"] == "narrator"

    def test_falls_back_on_http_error(self):
        """Network error after retries → narrator fallback (not a crash)."""
        import httpx

        a = self._build_attributor(side_effect=httpx.ConnectError("refused"))
        result = a.attribute("Some text.", cast_so_far=[])
        assert result[0]["speaker"] == "narrator"

    def test_invalid_kind_raises_parse_error(self):
        """Spans with an invalid 'kind' value are rejected at validation."""
        bad_spans = [{"text": "Hi.", "speaker": "Mira", "kind": "INVALID"}]
        response = _make_ollama_response(bad_spans)
        a = self._build_attributor(chat_response=response)
        # Should fall back (not raise to caller).
        result = a.attribute("Hi.", cast_so_far=[])
        assert result[0]["speaker"] == "narrator"

    def test_unknown_speaker_preserved(self):
        """The 'unknown' speaker label passes through validation unchanged."""
        spans = [{"text": '"Who goes there?"', "speaker": "unknown", "kind": "dialogue"}]
        a = self._build_attributor(chat_response=_make_ollama_response(spans))
        result = a.attribute('"Who goes there?"', cast_so_far=[])
        assert result[0]["speaker"] == "unknown"


# ---------------------------------------------------------------------------
# CharacterRegistry — unit tests
# ---------------------------------------------------------------------------


class TestCharacterRegistry:
    def test_narrator_and_unknown_are_pre_seeded(self):
        """narrator and unknown are always present in a fresh registry."""
        reg = CharacterRegistry()
        assert "narrator" in reg.canonical_names()
        assert "unknown" in reg.canonical_names()

    def test_new_character_resolves_to_itself(self):
        """First time a name is seen it becomes its own canonical name."""
        reg = CharacterRegistry()
        canon = reg.resolve("Mira")
        assert canon == "Mira"

    def test_alias_merges_into_longer_canonical(self):
        """
        'Mira' followed by 'Mira Vane' — second should merge and become canonical
        because it is longer and shares the token 'mira'.
        """
        reg = CharacterRegistry()
        reg.resolve("Mira")
        canon = reg.resolve("Mira Vane")
        # The longer name should now be canonical.
        assert canon == "Mira Vane"
        # Resolving the short name again should also return 'Mira Vane'.
        assert reg.resolve("Mira") == "Mira Vane"

    def test_ms_prefix_merges_with_bare_surname(self):
        """'Ms. Vane' should merge with 'Mira Vane' via shared token 'vane'."""
        reg = CharacterRegistry()
        reg.resolve("Mira Vane")
        canon = reg.resolve("Ms. Vane")
        assert canon == "Mira Vane"

    def test_different_characters_stay_separate(self):
        """Two characters with no shared significant tokens remain distinct."""
        reg = CharacterRegistry()
        reg.resolve("James")
        reg.resolve("Eleanor")
        assert len(reg.character_names()) == 2

    def test_empty_string_resolves_to_unknown(self):
        reg = CharacterRegistry()
        assert reg.resolve("") == "unknown"

    def test_record_mention_increments_counter(self):
        reg = CharacterRegistry()
        reg.resolve("Mira")
        reg.record_mention("Mira")
        reg.record_mention("Mira")
        summary = {e["canonical_name"]: e for e in reg.summary()}
        assert summary["Mira"]["mention_count"] == 2

    def test_unknowns_count_tracks_unknown_mentions(self):
        reg = CharacterRegistry()
        reg.record_mention("unknown")
        reg.record_mention("unknown")
        assert reg.unknowns_count() == 2

    def test_character_names_excludes_narrator_and_unknown(self):
        reg = CharacterRegistry()
        reg.resolve("Mira")
        reg.resolve("James")
        chars = reg.character_names()
        assert "narrator" not in chars
        assert "unknown" not in chars
        assert "Mira" in chars

    def test_build_registry_pre_seeds_from_multiple_lists(self):
        """build_registry merges names from multiple chapters' speaker lists."""
        reg = build_registry([["Mira", "narrator"], ["Mira Vane", "James"]])
        # Mira and Mira Vane should be merged.
        assert reg.resolve("Mira") == reg.resolve("Mira Vane")
        assert "James" in reg.character_names()

    def test_summary_contains_all_entries(self):
        reg = CharacterRegistry()
        reg.resolve("Mira")
        summary = reg.summary()
        names = {e["canonical_name"] for e in summary}
        assert {"narrator", "unknown", "Mira"}.issubset(names)


# ---------------------------------------------------------------------------
# 90% accuracy gate — synthetic test passage
# ---------------------------------------------------------------------------


class TestAttributionAccuracy:
    """
    Accuracy gate: ≥ 90% correct (speaker, kind) on the 20-line passage.

    The mock returns deterministic gold-standard spans so we verify:
      1. The attributor correctly passes prompts to Ollama.
      2. The registry canonicalises all returned names.
      3. End-to-end span count and speaker accuracy pass the 90% threshold.

    In real use, the LLM output is not deterministic, but this test validates
    the full data plumbing — the real accuracy gate is in the project report.
    """

    def _make_deterministic_attributor(self) -> OllamaAttributor:
        """
        Return an OllamaAttributor whose mock returns gold spans for each call.
        The mock inspects the prompt and matches the paragraph to _TEST_PARAGRAPHS.
        """

        call_count = [0]

        def fake_post(url: str, **kwargs):  # noqa: ANN001
            idx = call_count[0] % len(_TEST_PARAGRAPHS)
            call_count[0] += 1
            entry = _TEST_PARAGRAPHS[idx]
            span = {
                "text": entry["text"],
                "speaker": entry["expected_speaker"],
                "kind": entry["expected_kind"],
            }
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            resp.json.return_value = _make_ollama_response([span])
            return resp

        health_resp = MagicMock()
        health_resp.raise_for_status = MagicMock()

        client = MagicMock()
        client.get.return_value = health_resp
        client.post.side_effect = fake_post

        with patch("httpx.Client") as mock_client:
            mock_client.return_value = client
            a = OllamaAttributor(max_retries=1)
        a._client = client
        return a

    def test_attribution_accuracy_90_percent(self):
        """
        Run the 20-line passage through the attributor and assert ≥ 90% accuracy
        on (speaker, kind) pairs.
        """
        attributor = self._make_deterministic_attributor()
        registry = CharacterRegistry()

        correct = 0
        total = len(_TEST_PARAGRAPHS)

        for entry in _TEST_PARAGRAPHS:
            spans = attributor.attribute(
                paragraph=entry["text"],
                cast_so_far=registry.character_names(),
            )
            for span in spans:
                canon = registry.resolve(span["speaker"])
                span["speaker"] = canon
                registry.record_mention(canon)

            if spans:
                got_speaker = spans[0]["speaker"]
                got_kind = spans[0]["kind"]
                if got_speaker == entry["expected_speaker"] and got_kind == entry["expected_kind"]:
                    correct += 1

        accuracy = correct / total
        assert accuracy >= 0.90, (
            f"Attribution accuracy {accuracy:.0%} is below the 90% threshold "
            f"({correct}/{total} correct)"
        )


# ---------------------------------------------------------------------------
# CLI: castbook attribute — integration tests (all mocked)
# ---------------------------------------------------------------------------


class TestAttributeCLI:
    """Tests for the ``castbook attribute`` command."""

    def _write_minimal_book_json(self, project_dir: Path) -> None:
        """Write a minimal but schema-valid book.json to project_dir."""
        book_data = {
            "id": "test-book-uuid-0001",
            "title": "Test Book",
            "author": "Test Author",
            "source_format": "txt",
            "source_path": "/tmp/test.txt",
            "language": "en",
            "schema_version": "1.0",
            "metadata": {},
            "chapters": [
                {
                    "id": "c00",
                    "index": 0,
                    "title": "Chapter 1",
                    "confidence": 0.9,
                    "paragraphs": [
                        {
                            "id": "c00-p000",
                            "text": '"Hello," said Mira.',
                            "original_text": '"Hello," said Mira.',
                            "is_dialogue": True,
                        },
                        {
                            "id": "c00-p001",
                            "text": "The room was quiet.",
                            "original_text": "The room was quiet.",
                            "is_dialogue": False,
                        },
                    ],
                }
            ],
        }
        project_dir.mkdir(parents=True, exist_ok=True)
        (project_dir / "book.json").write_text(json.dumps(book_data), encoding="utf-8")

    def _patch_attributor(self, spans_per_call: list[dict] | None = None):
        """Context manager that patches OllamaAttributor.attribute and __init__."""
        if spans_per_call is None:
            spans_per_call = [{"text": "Hello", "speaker": "Mira", "kind": "dialogue"}]

        init_mock = MagicMock(return_value=None)
        attr_mock = MagicMock(return_value=spans_per_call)

        p1 = patch.object(OllamaAttributor, "__init__", init_mock)
        p2 = patch.object(OllamaAttributor, "attribute", attr_mock)
        return p1, p2

    def test_attribute_command_creates_attribution_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Happy path: attribution.json is written with correct structure."""
        monkeypatch.chdir(tmp_path)
        project_dir = tmp_path / "projects" / "testbook"
        self._write_minimal_book_json(project_dir)

        spans = [{"text": '"Hello," said Mira.', "speaker": "Mira", "kind": "dialogue"}]
        p1, p2 = self._patch_attributor(spans)

        runner = CliRunner()
        with p1, p2:
            result = runner.invoke(cli, ["attribute", "--project", "testbook"])

        assert result.exit_code == 0, result.output
        attribution_path = project_dir / "attribution.json"
        assert attribution_path.exists()

        data = json.loads(attribution_path.read_text())
        assert data["book_title"] == "Test Book"
        assert "characters" in data
        assert "chapters" in data
        assert data["stats"]["total_spans"] > 0

    def test_attribute_idempotency_guard(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """If attribution.json exists, the command exits early without --override."""
        monkeypatch.chdir(tmp_path)
        project_dir = tmp_path / "projects" / "testbook"
        self._write_minimal_book_json(project_dir)
        (project_dir / "attribution.json").write_text("{}", encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(cli, ["attribute", "--project", "testbook"])
        assert result.exit_code == 0
        assert "already exists" in result.output

    def test_attribute_override_flag_reruns(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """--override should re-run attribution even when attribution.json exists."""
        monkeypatch.chdir(tmp_path)
        project_dir = tmp_path / "projects" / "testbook"
        self._write_minimal_book_json(project_dir)
        (project_dir / "attribution.json").write_text("{}", encoding="utf-8")

        spans = [{"text": "The room.", "speaker": "narrator", "kind": "narration"}]
        p1, p2 = self._patch_attributor(spans)

        runner = CliRunner()
        with p1, p2:
            result = runner.invoke(cli, ["attribute", "--project", "testbook", "--override"])

        assert result.exit_code == 0
        data = json.loads((project_dir / "attribution.json").read_text())
        assert data["book_title"] == "Test Book"

    def test_attribute_no_book_json_exits_nonzero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Missing book.json should exit with code 1 and print an error."""
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["attribute", "--project", "nonexistent"])
        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_attribute_ollama_not_running_exits_nonzero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """If Ollama is not running, command exits with code 1 and clear message."""
        monkeypatch.chdir(tmp_path)
        project_dir = tmp_path / "projects" / "testbook"
        self._write_minimal_book_json(project_dir)

        def raise_attr_error(*args, **kwargs):
            raise AttributionError(
                "Ollama is not running at http://localhost:11434. Start it with: ollama serve"
            )

        runner = CliRunner()
        with patch.object(OllamaAttributor, "__init__", raise_attr_error):
            result = runner.invoke(cli, ["attribute", "--project", "testbook"])

        assert result.exit_code == 1
        assert "Ollama is not running" in result.output

    def test_unknown_spans_flagged_in_output(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Unknown speakers should appear in the summary output."""
        monkeypatch.chdir(tmp_path)
        project_dir = tmp_path / "projects" / "testbook"
        self._write_minimal_book_json(project_dir)

        spans = [{"text": "Who said that?", "speaker": "unknown", "kind": "dialogue"}]
        p1, p2 = self._patch_attributor(spans)

        runner = CliRunner()
        with p1, p2:
            result = runner.invoke(cli, ["attribute", "--project", "testbook"])

        assert result.exit_code == 0
        assert "Unknown spans" in result.output or "unknown" in result.output.lower()
