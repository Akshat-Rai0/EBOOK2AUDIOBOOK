"""
System environment detection — used by ``castbook doctor``.

Checks:
  - Python version and platform.
  - Available RAM and free disk space.
  - GPU presence (CUDA / MPS / CPU-only).
  - FFmpeg installation and version.
  - Tier verdict: narrator-only or multi-voice.

Hardware tiers (from the project specification):
  narrator_only — 4 GB RAM / 2 GB VRAM minimum; uses VITS only.
  multi_voice   — 8 GB RAM recommended; uses local LLM (Ollama) + XTTS-v2.

These thresholds are targets, not benchmarked limits — they will be revised
after the Week 4 and Week 6 hardware tests.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Tier thresholds (GiB)
# ---------------------------------------------------------------------------
_NARRATOR_ONLY_RAM_GIB = 4.0
_MULTI_VOICE_RAM_GIB = 8.0
_NARRATOR_ONLY_DISK_GIB = 5.0  # OS + venv + VITS model
_MULTI_VOICE_DISK_GIB = 20.0  # OS + venv + LLM (3B 4-bit ≈ 2 GB) + XTTS-v2 (≈ 2 GB)


def get_python_info() -> dict:
    return {
        "version": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
    }


def get_ram_gib() -> float | None:
    """Return total physical RAM in GiB, or None if detection fails."""
    try:
        import psutil

        return psutil.virtual_memory().total / (1024**3)
    except ImportError:
        pass
    # macOS fallback via sysctl.
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            return int(result.stdout.strip()) / (1024**3)
        except Exception:
            pass
    return None


def get_gpu_info() -> dict:
    """
    Detect available GPU acceleration.

    Returns a dict with keys:
      backend — "cuda", "mps", or "cpu"
      vram_gib — VRAM in GiB for CUDA, None otherwise
      device_name — human-readable device name
    """
    try:
        import torch

        if torch.cuda.is_available():
            device = torch.cuda.get_device_properties(0)
            return {
                "backend": "cuda",
                "vram_gib": device.total_memory / (1024**3),
                "device_name": device.name,
            }
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return {"backend": "mps", "vram_gib": None, "device_name": "Apple Silicon (MPS)"}
    except ImportError:
        pass
    return {"backend": "cpu", "vram_gib": None, "device_name": "CPU only"}


def get_ffmpeg_info() -> dict:
    """Return FFmpeg version string or indicate it is missing."""
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        return {"installed": False, "version": None, "path": None}
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        first_line = result.stdout.splitlines()[0] if result.stdout else "unknown"
        return {"installed": True, "version": first_line, "path": ffmpeg_path}
    except Exception:
        return {"installed": True, "version": "unknown", "path": ffmpeg_path}


def get_free_disk_gib(path: Path = Path(".")) -> float | None:
    """Return free disk space in GiB at *path*, or None if detection fails."""
    try:
        usage = shutil.disk_usage(path)
        return usage.free / (1024**3)
    except Exception:
        return None


def compute_tier(ram_gib: float | None, gpu: dict, disk_gib: float | None) -> str:
    """
    Decide the supported hardware tier.

    Returns ``"multi_voice"``, ``"narrator_only"``, or ``"unsupported"``.
    """
    if ram_gib is None:
        return "unknown"
    if ram_gib >= _MULTI_VOICE_RAM_GIB:
        return "multi_voice"
    if ram_gib >= _NARRATOR_ONLY_RAM_GIB:
        return "narrator_only"
    return "unsupported"


def run_doctor(project_dir: Path = Path(".")) -> dict:
    """
    Collect all environment information and return a structured report dict.

    Used by ``castbook doctor`` to print the system readiness report.
    """
    ram = get_ram_gib()
    gpu = get_gpu_info()
    disk = get_free_disk_gib(project_dir)
    ffmpeg = get_ffmpeg_info()
    tier = compute_tier(ram, gpu, disk)

    return {
        "python": get_python_info(),
        "ram_gib": ram,
        "gpu": gpu,
        "disk_free_gib": disk,
        "ffmpeg": ffmpeg,
        "tier": tier,
    }
