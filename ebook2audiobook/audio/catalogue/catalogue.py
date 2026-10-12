"""
Voice catalogue: discovery, acoustic proxy distance, and profile matching.

Concepts:
  VoiceEntry — Metadata for a single speaker voice in a specific TTS engine.
  VoiceCatalogue — Aggregation of all engine catalogues (Fake, VITS, XTTS)
                   with verified demographic filtering and acoustic proxy distance.

Verification Rule:
  Demographic tags (gender, age_bracket, accent) carry a `verified: false` flag
  by default.  The caster ONLY filters on demographics when verified=True,
  otherwise falling back to acoustic distinctness.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Standard normalization ranges for acoustic proxies
_F0_MEAN = 160.0
_F0_STD = 45.0
_SC_MEAN = 1900.0
_SC_STD = 400.0


class VoiceEntry(BaseModel):
    """Metadata for one speaker in a TTS engine catalogue."""

    id: str = Field(..., description="Engine voice ID (e.g. 'p225', 'fake-char-1').")
    name: str = Field(..., description="Display name of the speaker.")
    engine: str = Field(..., description="TTS engine: 'fake', 'vits', 'xtts'.")
    gender: str = Field(default="unknown", description="male | female | nonbinary | unknown")
    gender_verified: bool = Field(default=False, description="True if verified by listening.")
    age_bracket: str = Field(
        default="unknown",
        description="child | young_adult | adult | elderly | unknown",
    )
    age_verified: bool = Field(default=False, description="True if verified by listening.")
    accent: str = Field(default="unknown", description="Accent tag or unknown.")
    accent_verified: bool = Field(default=False, description="True if verified by listening.")
    proxy_median_f0_hz: float | None = Field(
        default=None,
        description="Median fundamental frequency (Hz) proxy. Null if unmeasured.",
    )
    proxy_spectral_centroid_hz: float | None = Field(
        default=None,
        description="Center of mass of spectrum (Hz) proxy. Null if unmeasured.",
    )
    acoustic_proxy_verified: bool = Field(
        default=False,
        description="True if proxy measured on ground truth.",
    )
    licence: str = Field(default="MIT", description="Licence covering this voice model.")
    narrator_reserved: bool = Field(
        default=False,
        description="True if reserved for default narrator.",
    )

    def acoustic_distance_to(self, other: VoiceEntry) -> float:
        """
        Compute Euclidean distance in normalized (F0, Spectral Centroid) space.

        A distance < 1.0 represents high acoustic similarity (potential collision).
        Returns infinity if either voice lacks proxy measurements.
        """
        if self.proxy_median_f0_hz is None or self.proxy_spectral_centroid_hz is None:
            return float("inf")
        if other.proxy_median_f0_hz is None or other.proxy_spectral_centroid_hz is None:
            return float("inf")

        z_f0_self = (self.proxy_median_f0_hz - _F0_MEAN) / _F0_STD
        z_f0_other = (other.proxy_median_f0_hz - _F0_MEAN) / _F0_STD
        z_sc_self = (self.proxy_spectral_centroid_hz - _SC_MEAN) / _SC_STD
        z_sc_other = (other.proxy_spectral_centroid_hz - _SC_MEAN) / _SC_STD

        delta_f0 = z_f0_self - z_f0_other
        delta_sc = z_sc_self - z_sc_other
        return math.sqrt(delta_f0 * delta_f0 + delta_sc * delta_sc)


class VoiceCatalogue:
    """Manages voice entries across engines."""

    def __init__(
        self,
        catalogue_dir: Path | None = None,
        engine_validators: dict[str, list[dict]] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        catalogue_dir:
            Directory containing {engine}_voices.json files.
        engine_validators:
            Optional mapping of engine name to list_voices() output for validation.
            If provided, catalogue IDs are validated against the real engine's voice list
            and unknown IDs are dropped with a warning.
        """
        self.catalogue_dir = catalogue_dir or Path(__file__).parent
        self._entries_by_engine: dict[str, list[VoiceEntry]] = {}
        self._load_all(engine_validators or {})

    def _load_all(self, engine_validators: dict[str, list[dict]]) -> None:
        for engine in ("fake", "vits", "xtts"):
            path = self.catalogue_dir / f"{engine}_voices.json"
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    entries = [VoiceEntry.model_validate(item) for item in data]

                    # Validate against engine's real voice list if validator provided
                    if engine in engine_validators:
                        valid_ids = {v["id"] for v in engine_validators[engine]}
                        filtered_entries = []
                        for entry in entries:
                            if entry.id in valid_ids:
                                filtered_entries.append(entry)
                            else:
                                logger.warning(
                                    "Dropping unknown voice ID %s from %s catalogue "
                                    "(not in engine.list_voices())",
                                    entry.id,
                                    engine,
                                )
                        entries = filtered_entries

                    self._entries_by_engine[engine] = entries
                except Exception as exc:
                    logger.warning("Failed to load catalogue %s: %s", path, exc)
                    self._entries_by_engine[engine] = []
            else:
                self._entries_by_engine[engine] = []

    def get_entries(self, engine: str) -> list[VoiceEntry]:
        """Return all voice entries for an engine."""
        return list(self._entries_by_engine.get(engine.lower(), []))

    def get_voice(self, engine: str, voice_id: str) -> VoiceEntry | None:
        """Find a specific voice by engine and ID."""
        for entry in self.get_entries(engine):
            if entry.id == voice_id:
                return entry
        return None

    def get_narrator_voice(self, engine: str) -> VoiceEntry | None:
        """Return the reserved narrator voice for an engine."""
        for entry in self.get_entries(engine):
            if entry.narrator_reserved:
                return entry
        entries = self.get_entries(engine)
        return entries[0] if entries else None

    def available_for_characters(self, engine: str) -> list[VoiceEntry]:
        """Return all voices for an engine excluding reserved narrator voices."""
        return [e for e in self.get_entries(engine) if not e.narrator_reserved]

    def filter_verified(
        self,
        engine: str,
        gender: str | None = None,
        age_bracket: str | None = None,
    ) -> list[VoiceEntry]:
        """
        Filter available character voices by VERIFIED demographic tags.

        If a voice has gender_verified=False or age_verified=False, it will NOT
        match a specific gender/age query (enforces unverified metadata rule).
        """
        candidates = self.available_for_characters(engine)
        if gender and gender != "unknown":
            candidates = [
                c
                for c in candidates
                if c.gender_verified and c.gender.lower() == gender.lower()
            ]
        if age_bracket and age_bracket != "unknown":
            candidates = [
                c
                for c in candidates
                if c.age_verified and c.age_bracket.lower() == age_bracket.lower()
            ]
        return candidates

    def validate_against_engine(self, engine: str, engine_voices: list[dict]) -> None:
        """
        Validate catalogue entries against a real engine's voice list.

        Drops unknown voice IDs with a warning. This should be called once
        after engine initialization to ensure catalogue only contains valid IDs.
        """
        valid_ids = {v["id"] for v in engine_voices}
        filtered_entries = []
        for entry in self._entries_by_engine.get(engine, []):
            if entry.id in valid_ids:
                filtered_entries.append(entry)
            else:
                logger.warning(
                    "Dropping unknown voice ID %s from %s catalogue "
                    "(not in engine.list_voices())",
                    entry.id,
                    engine,
                )
        self._entries_by_engine[engine] = filtered_entries
