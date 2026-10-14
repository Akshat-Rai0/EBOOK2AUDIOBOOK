"""Evaluation module for M3 attribution redo.

This module contains tools for:
- Quote alignment between gold labels and production extractor
- Metrics calculation
- Calibration analysis
- Evaluation harness
"""

from ebook2audiobook.evaluation.alignment import (
    AlignmentResult,
    align_gold_with_extractor,
    compute_text_hash,
    extract_quotes_from_text,
    verify_source_integrity,
)
from ebook2audiobook.evaluation.metrics import (
    AliasMergeMetrics,
    AttributionMetrics,
    ConfusionPair,
    bootstrap_confidence_interval,
    compute_accuracy,
    compute_alias_merge_metrics,
    compute_confusion_matrix,
    compute_labeller_consistency,
)
from ebook2audiobook.evaluation.style import (
    QuoteStyle,
    StyleAnalysis,
    analyze_chapter_styles,
    apply_manual_overrides,
    detect_style,
)

__all__ = [
    "AlignmentResult",
    "align_gold_with_extractor",
    "compute_text_hash",
    "extract_quotes_from_text",
    "verify_source_integrity",
    "AliasMergeMetrics",
    "AttributionMetrics",
    "ConfusionPair",
    "bootstrap_confidence_interval",
    "compute_accuracy",
    "compute_alias_merge_metrics",
    "compute_confusion_matrix",
    "compute_labeller_consistency",
    "QuoteStyle",
    "StyleAnalysis",
    "analyze_chapter_styles",
    "apply_manual_overrides",
    "detect_style",
]
