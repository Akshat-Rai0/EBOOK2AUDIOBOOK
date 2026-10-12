"""
Build voice catalogue JSON files from real engine speaker lists.

Generates:
  ebook2audiobook/audio/catalogue/vits_voices.json (from VITS model speaker names)
  ebook2audiobook/audio/catalogue/xtts_voices.json (from XTTS speaker_manager names)
  ebook2audiobook/audio/catalogue/fake_voices.json (from FakeTTS voices)

Demographic tags and proxy acoustic metrics are initialized to null/unverified.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CATALOGUE_DIR = Path("ebook2audiobook/audio/catalogue")


def build_fake_catalogue() -> list[dict]:
    from ebook2audiobook.tts.fake_tts import FakeTTS

    tts = FakeTTS()
    voices = tts.list_voices()
    entries = []
    for v in voices:
        vid = v["id"]
        narrator = (vid == "fake-narrator")
        entries.append({
            "id": vid,
            "name": v["name"],
            "engine": "fake",
            "gender": "unknown",
            "gender_verified": False,
            "age_bracket": "unknown",
            "age_verified": False,
            "accent": "unknown",
            "accent_verified": False,
            "proxy_median_f0_hz": None,
            "proxy_spectral_centroid_hz": None,
            "acoustic_proxy_verified": False,
            "licence": "MIT",
            "narrator_reserved": narrator,
        })
    return entries


def build_vits_catalogue() -> list[dict]:
    from ebook2audiobook.tts.vits import VITS

    vits = VITS(device="cpu")
    voices = vits.list_voices()
    entries = []
    for v in voices:
        vid = v["id"].strip()
        if not vid:
            continue
        narrator = (vid == "p225")
        entries.append({
            "id": vid,
            "name": f"Speaker {vid}",
            "engine": "vits",
            "gender": "unknown",
            "gender_verified": False,
            "age_bracket": "unknown",
            "age_verified": False,
            "accent": "unknown",
            "accent_verified": False,
            "proxy_median_f0_hz": None,
            "proxy_spectral_centroid_hz": None,
            "acoustic_proxy_verified": False,
            "licence": "MIT / ODC-By",
            "narrator_reserved": narrator,
        })
    return entries


def build_xtts_catalogue() -> list[dict]:
    from ebook2audiobook.tts.xtts import XTTS

    xtts = XTTS(device="cpu")
    voices = xtts.list_voices()
    entries = []
    for v in voices:
        vid = v["id"].strip()
        if not vid:
            continue
        narrator = (vid == "Ana Florence")
        entries.append({
            "id": vid,
            "name": vid,
            "engine": "xtts",
            "gender": "unknown",
            "gender_verified": False,
            "age_bracket": "unknown",
            "age_verified": False,
            "accent": "unknown",
            "accent_verified": False,
            "proxy_median_f0_hz": None,
            "proxy_spectral_centroid_hz": None,
            "acoustic_proxy_verified": False,
            "licence": "Coqui Public Model License (CPML) non-commercial",
            "narrator_reserved": narrator,
        })
    return entries


def main() -> None:
    CATALOGUE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Fake voices
    fake_entries = build_fake_catalogue()
    fake_path = CATALOGUE_DIR / "fake_voices.json"
    fake_path.write_text(json.dumps(fake_entries, indent=2), encoding="utf-8")
    logger.info("Wrote %d fake voices to %s", len(fake_entries), fake_path)

    # 2. VITS voices
    try:
        vits_entries = build_vits_catalogue()
        vits_path = CATALOGUE_DIR / "vits_voices.json"
        vits_path.write_text(json.dumps(vits_entries, indent=2), encoding="utf-8")
        logger.info("Wrote %d VITS voices to %s", len(vits_entries), vits_path)
    except Exception as exc:
        logger.warning("Could not build VITS voices (model may not be downloaded): %s", exc)

    # 3. XTTS voices
    try:
        xtts_entries = build_xtts_catalogue()
        xtts_path = CATALOGUE_DIR / "xtts_voices.json"
        xtts_path.write_text(json.dumps(xtts_entries, indent=2), encoding="utf-8")
        logger.info("Wrote %d XTTS voices to %s", len(xtts_entries), xtts_path)
    except Exception as exc:
        logger.warning("Could not build XTTS voices (model may not be downloaded): %s", exc)


if __name__ == "__main__":
    main()
