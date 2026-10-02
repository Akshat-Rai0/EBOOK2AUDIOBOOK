"""Public exports for the attribution package."""

from ebook2audiobook.attribution.attributor import AttributionError, Attributor
from ebook2audiobook.attribution.character_registry import (
    CharacterEntry,
    CharacterRegistry,
    build_registry,
)
from ebook2audiobook.attribution.ollama_attributor import OllamaAttributor
from ebook2audiobook.attribution.profiler import CharacterProfiler
from ebook2audiobook.attribution.stage import AttributionStage

__all__ = [
    "Attributor",
    "AttributionError",
    "AttributionStage",
    "CharacterEntry",
    "CharacterProfiler",
    "CharacterRegistry",
    "OllamaAttributor",
    "build_registry",
]
