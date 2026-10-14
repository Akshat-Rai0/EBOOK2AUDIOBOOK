"""Evaluation harness for M3 attribution redo.

Analogy: like a test runner that grades multiple student papers using
different grading rubrics and records the results.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class RunMetadata:
    """Metadata for a single evaluation run."""

    run_id: str
    timestamp: str
    model_name: str
    model_digest: str | None
    quantization: str | None
    context_window: int
    prompt_version: str
    variant: str
    commit_hash: str
    device: str
    date: str


@dataclass
class RunResult:
    """Result of a single evaluation run."""

    metadata: RunMetadata
    accuracy: float
    speech_only_accuracy: float
    unknown_rate: float
    json_validity: float
    retries_per_100: float
    seconds_per_1000_words: float
    tokens_per_second: float | None
    peak_ram_mb: float
    model_disk_size_mb: float
    style_breakdown: dict[str, float]
    confidence_breakdown: dict[str, dict[str, float]]
    calibration_threshold: float | None


def generate_run_id() -> str:
    """Generate a unique run ID."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_run_result(result: RunResult, output_dir: Path) -> Path:
    """
    Save a run result to a JSONL file.

    Parameters
    ----------
    result:
        The run result to save.
    output_dir:
        Directory to save results in.

    Returns
    -------
    Path
        Path to the saved file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{result.metadata.run_id}.jsonl"

    with output_path.open("w") as f:
        json.dump(asdict(result), f, indent=2)

    return output_path


def load_run_result(run_id: str, output_dir: Path) -> RunResult | None:
    """
    Load a run result from a JSONL file.

    Parameters
    ----------
    run_id:
        The run ID to load.
    output_dir:
        Directory containing results.

    Returns
    -------
    RunResult | None
        The loaded result, or None if not found.
    """
    output_path = output_dir / f"{run_id}.jsonl"

    if not output_path.exists():
        return None

    with output_path.open("r") as f:
        data = json.load(f)

    # Reconstruct nested objects
    metadata = RunMetadata(**data["metadata"])
    return RunResult(metadata=metadata, **{k: v for k, v in data.items() if k != "metadata"})


def list_runs(output_dir: Path) -> list[RunMetadata]:
    """
    List all available run metadata.

    Parameters
    ----------
    output_dir:
        Directory containing results.

    Returns
    -------
    list[RunMetadata]
        List of run metadata.
    """
    runs: list[RunMetadata] = []

    for path in output_dir.glob("*.jsonl"):
        try:
            with path.open("r") as f:
                data = json.load(f)
            metadata = RunMetadata(**data["metadata"])
            runs.append(metadata)
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")

    runs.sort(key=lambda r: r.timestamp, reverse=True)
    return runs


@dataclass
class EvaluationConfig:
    """Configuration for an evaluation run."""

    variant: str  # V0, V1, V2, V3, V4
    context_size: int  # Previous paragraphs to include
    few_shot: bool  # Whether to use few-shot examples
    rules_enabled: bool
    alternation_enabled: bool
    two_pass: bool  # Whether to use two-pass design


VARIANTS = {
    "V0": EvaluationConfig(
        variant="V0",
        context_size=1,
        few_shot=False,
        rules_enabled=True,
        alternation_enabled=False,
        two_pass=False,
    ),
    "V1": EvaluationConfig(
        variant="V1",
        context_size=1,
        few_shot=False,
        rules_enabled=False,
        alternation_enabled=False,
        two_pass=False,
    ),
    "V2": EvaluationConfig(
        variant="V2",
        context_size=1,
        few_shot=False,
        rules_enabled=True,
        alternation_enabled=False,
        two_pass=False,
    ),
    "V3": EvaluationConfig(
        variant="V3",
        context_size=1,
        few_shot=False,
        rules_enabled=True,
        alternation_enabled=True,
        two_pass=False,
    ),
    "V4": EvaluationConfig(
        variant="V4",
        context_size=1,
        few_shot=False,
        rules_enabled=True,
        alternation_enabled=True,
        two_pass=True,
    ),
}


def get_variant_config(variant: str) -> EvaluationConfig:
    """
    Get a variant configuration by name.

    Parameters
    ----------
    variant:
        Variant name (V0, V1, V2, V3, V4).

    Returns
    -------
    EvaluationConfig
        The variant configuration.

    Raises
    ------
    ValueError
        If variant is not recognized.
    """
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}. Available: {list(VARIANTS.keys())}")
    return VARIANTS[variant]
