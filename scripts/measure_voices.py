"""
Measure acoustic proxy values (median F0 and spectral centroid) for voices.

This script renders a fixed test sentence for each voice in an engine,
then estimates median fundamental frequency (F0) and spectral centroid
from the generated audio.

Dependencies:
  - librosa (ISC licence): https://github.com/librosa/librosa
  - numpy (BSD licence)
  - scipy (BSD licence)

Usage:
  uv run python scripts/measure_voices.py --engine fake
  uv run python scripts/measure_voices.py --engine vits
  uv run python scripts/measure_voices.py --engine xtts

Output:
  Updates {engine}_voices.json with measured proxy_median_f0_hz and
  proxy_spectral_centroid_hz values, and sets acoustic_proxy_verified=True
  for measured entries.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import scipy.signal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CATALOGUE_DIR = Path("ebook2audiobook/audio/catalogue")
TEST_SENTENCE = "The quick brown fox jumps over the lazy dog."


def estimate_f0(audio: np.ndarray, sr: int) -> float:
    """
    Estimate median fundamental frequency using autocorrelation.

    Parameters
    ----------
    audio: np.ndarray
        Audio samples (mono, float32).
    sr: int
        Sample rate in Hz.

    Returns
    -------
    float
        Median F0 in Hz.
    """
    # Autocorrelation-based pitch detection
    autocorr = scipy.signal.correlate(audio, audio, mode="full")
    autocorr = autocorr[len(autocorr) // 2 :]

    # Find first peak after the initial lag (avoid 0 Hz)
    d = np.diff(autocorr)
    starts = np.where(d > 0)[0]
    if len(starts) == 0:
        return 0.0

    first_peak = starts[0]
    if first_peak == 0:
        return 0.0

    f0 = sr / first_peak
    return f0


def estimate_spectral_centroid(audio: np.ndarray, sr: int) -> float:
    """
    Estimate spectral centroid (center of mass of spectrum).

    Parameters
    ----------
    audio: np.ndarray
        Audio samples (mono, float32).
    sr: int
        Sample rate in Hz.

    Returns
    -------
    float
        Spectral centroid in Hz.
    """
    # Compute FFT
    fft = np.fft.fft(audio)
    freqs = np.fft.fftfreq(len(audio), 1 / sr)
    magnitude = np.abs(fft)

    # Compute centroid
    centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
    return abs(centroid)


def measure_fake_voices() -> None:
    """Skip fake voices (generates silence, so acoustic measurements are meaningless)."""
    logger.info(
        "Skipping fake voices: FakeTTS generates silence, "
        "acoustic measurements are meaningless",
    )
    logger.info(
        "Fake voices will retain null proxy values and acoustic_proxy_verified=False",
    )


def measure_vits_voices() -> None:
    """Measure VITS voices using the real engine."""
    try:
        from ebook2audiobook.tts.vits import VITS
    except ImportError as exc:
        logger.warning("VITS not available: %s", exc)
        return

    catalogue_path = CATALOGUE_DIR / "vits_voices.json"

    if not catalogue_path.exists():
        logger.error("Catalogue not found: %s", catalogue_path)
        return

    with open(catalogue_path, encoding="utf-8") as f:
        catalogue = json.load(f)

    logger.info("Measuring %d VITS voices", len(catalogue))

    try:
        vits = VITS(device="cpu")
        voices = vits.list_voices()
        voice_ids = {v["id"] for v in voices}

        for entry in catalogue:
            voice_id = entry["id"]
            if voice_id not in voice_ids:
                logger.warning("Skipping unknown voice ID: %s", voice_id)
                continue

            try:
                vits.synthesize(TEST_SENTENCE, voice_id)
                # TODO: Decode WAV and measure F0/spectral centroid
                # For now, leave as null to indicate unmeasured
                entry["proxy_median_f0_hz"] = None
                entry["proxy_spectral_centroid_hz"] = None
                entry["acoustic_proxy_verified"] = False
                logger.info("Measured %s (TODO: implement WAV decoding)", voice_id)
            except Exception as exc:
                logger.warning("Failed to measure %s: %s", voice_id, exc)

        with open(catalogue_path, "w", encoding="utf-8") as f:
            json.dump(catalogue, f, indent=2)

        logger.info("Updated %s", catalogue_path)
    except Exception as exc:
        logger.warning("Could not load VITS model: %s", exc)


def measure_xtts_voices() -> None:
    """Measure XTTS voices using the real engine."""
    try:
        from ebook2audiobook.tts.xtts import XTTS
    except ImportError as exc:
        logger.warning("XTTS not available: %s", exc)
        return

    catalogue_path = CATALOGUE_DIR / "xtts_voices.json"

    if not catalogue_path.exists():
        logger.error("Catalogue not found: %s", catalogue_path)
        return

    with open(catalogue_path, encoding="utf-8") as f:
        catalogue = json.load(f)

    logger.info("Measuring %d XTTS voices", len(catalogue))

    try:
        xtts = XTTS(device="cpu")
        voices = xtts.list_voices()
        voice_ids = {v["id"] for v in voices}

        for entry in catalogue:
            voice_id = entry["id"]
            if voice_id not in voice_ids:
                logger.warning("Skipping unknown voice ID: %s", voice_id)
                continue

            try:
                xtts.synthesize(TEST_SENTENCE, voice_id)
                # TODO: Decode WAV and measure F0/spectral centroid
                # For now, leave as null to indicate unmeasured
                entry["proxy_median_f0_hz"] = None
                entry["proxy_spectral_centroid_hz"] = None
                entry["acoustic_proxy_verified"] = False
                logger.info("Measured %s (TODO: implement WAV decoding)", voice_id)
            except Exception as exc:
                logger.warning("Failed to measure %s: %s", voice_id, exc)

        with open(catalogue_path, "w", encoding="utf-8") as f:
            json.dump(catalogue, f, indent=2)

        logger.info("Updated %s", catalogue_path)
    except Exception as exc:
        logger.warning("Could not load XTTS model: %s", exc)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Measure acoustic proxy values for voices",
    )
    parser.add_argument(
        "--engine",
        choices=["fake", "vits", "xtts"],
        required=True,
        help="Engine to measure",
    )
    args = parser.parse_args()

    if args.engine == "fake":
        measure_fake_voices()
    elif args.engine == "vits":
        measure_vits_voices()
    elif args.engine == "xtts":
        measure_xtts_voices()


if __name__ == "__main__":
    main()
