"""
Configuration system for CastBook.

Handles:
- Default values (e.g., VITS narrator speaker)
- User config file (~/.castbook/config.toml)
- Environment variable overrides (CASTBOOK_MODELS_DIR)
- Model manifest path (~/.castbook/models.json)

Analogy: like a settings menu that remembers your preferences between sessions.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

try:
    import tomli
except ImportError:
    tomli = None

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default configuration values
# ---------------------------------------------------------------------------

DEFAULT_VITS_NARRATOR = "p225"  # Temporary default until M4 listen-labelling
DEFAULT_CONFIG_DIR = Path.home() / ".castbook"
DEFAULT_CONFIG_FILE = DEFAULT_CONFIG_DIR / "config.toml"
DEFAULT_MANIFEST_FILE = DEFAULT_CONFIG_DIR / "models.json"


# ---------------------------------------------------------------------------
# Model directory resolution
# ---------------------------------------------------------------------------


def get_models_dir() -> Path:
    """
    Get the directory where TTS models are stored.

    Resolution order:
    1. CASTBOOK_MODELS_DIR environment variable (user override)
    2. If set, sets TTS_HOME env var (Coqui honours this)
    3. Falls back to Coqui's default via TTS.utils.generic_utils.get_user_data_dir("tts")

    Returns
    -------
    Path
        Directory path for model storage.
    """
    castbook_models_dir = os.environ.get("CASTBOOK_MODELS_DIR")
    if castbook_models_dir:
        # Set TTS_HOME so Coqui uses our custom directory
        os.environ["TTS_HOME"] = castbook_models_dir
        logger.debug(f"Using CASTBOOK_MODELS_DIR: {castbook_models_dir}")
        return Path(castbook_models_dir)

    # Use Coqui's default
    try:
        from TTS.utils.generic_utils import get_user_data_dir

        coqui_dir = Path(get_user_data_dir("tts"))
        logger.debug(f"Using Coqui default cache: {coqui_dir}")
        return coqui_dir
    except ImportError:
        # Coqui not installed yet, use the default location
        import platform

        if platform.system() == "Darwin":
            default = Path.home() / "Library" / "Application Support" / "tts"
        elif platform.system() == "Windows":
            default = Path.home() / "AppData" / "Local" / "tts"
        else:  # Linux and others
            default = Path.home() / ".local" / "share" / "tts"

        logger.debug(f"Coqui not installed, using platform default: {default}")
        return default


def get_manifest_path() -> Path:
    """Return the path to the models manifest file (~/.castbook/models.json)."""
    return DEFAULT_MANIFEST_FILE


def get_config_path() -> Path:
    """Return the path to the user config file (~/.castbook/config.toml)."""
    return DEFAULT_CONFIG_FILE


# ---------------------------------------------------------------------------
# Config file handling
# ---------------------------------------------------------------------------


def read_config() -> dict:
    """
    Read the user config file (~/.castbook/config.toml).

    Returns an empty dict if the file doesn't exist or tomli is not available.
    """
    config_path = get_config_path()
    if not config_path.exists():
        return {}

    if tomli is None:
        logger.warning("tomli not installed, cannot read config.toml")
        return {}

    try:
        with open(config_path, "rb") as f:
            config = tomli.load(f)
        logger.debug(f"Loaded config from {config_path}")
        return config
    except Exception as exc:
        logger.warning(f"Failed to read config.toml: {exc}")
        return {}


def write_config(config: dict) -> None:
    """
    Write configuration to ~/.castbook/config.toml.

    Requires tomli_w for writing. If not available, log a warning.
    """
    config_path = get_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import tomli_w

        with open(config_path, "wb") as f:
            tomli_w.dump(config, f)
        logger.debug(f"Wrote config to {config_path}")
    except ImportError:
        logger.warning("tomli_w not installed, cannot write config.toml")
    except Exception as exc:
        logger.warning(f"Failed to write config.toml: {exc}")


# ---------------------------------------------------------------------------
# Configuration accessors
# ---------------------------------------------------------------------------


def get_vits_narrator() -> str:
    """
    Get the VITS narrator speaker ID.

    Resolution order:
    1. config.toml [vits] narrator setting
    2. DEFAULT_VITS_NARRATOR (p225)

    Returns
    -------
    str
        Speaker ID for the narrator (e.g., "p225").
    """
    config = read_config()
    vits_config = config.get("vits", {})
    return vits_config.get("narrator", DEFAULT_VITS_NARRATOR)


def set_vits_narrator(speaker_id: str) -> None:
    """
    Set the VITS narrator speaker ID in config.toml.

    Parameters
    ----------
    speaker_id:
        The speaker ID to use (e.g., "p225").
    """
    config = read_config()
    if "vits" not in config:
        config["vits"] = {}
    config["vits"]["narrator"] = speaker_id
    write_config(config)
    logger.info(f"Set VITS narrator to {speaker_id}")
