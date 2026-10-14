"""Calibration analysis for attribution confidence.

Analogy: like adjusting a scale—calibration ensures the confidence score
means what we think it means (e.g., 0.90 confidence should actually be
correct 90% of the time).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceBucket:
    """Accuracy statistics for a confidence range."""

    min_conf: float
    max_conf: float
    total: int
    correct: int
    accuracy: float

    @property
    def precision(self) -> float:
        """Precision (correct / total)."""
        return self.accuracy


@dataclass
class CalibrationCurve:
    """Calibration data across thresholds."""

    thresholds: list[float]
    auto_accept_precision: list[float]
    review_fraction: list[float]
    bucket_stats: list[ConfidenceBucket]


def bucket_by_confidence(
    predictions: dict[str, tuple[str, float]],
    gold_labels: dict[str, str],
    bucket_boundaries: list[float] | None = None,
) -> list[ConfidenceBucket]:
    """
    Group predictions by confidence buckets and compute accuracy per bucket.

    Parameters
    ----------
    predictions:
        Mapping from quote_id to (predicted_speaker, confidence).
    gold_labels:
        Mapping from quote_id to gold speaker label.
    bucket_boundaries:
        Confidence boundaries (e.g., [0.0, 0.5, 0.7, 0.9, 1.0]).
        If None, uses [0.0, 0.4, 0.7, 0.95, 1.0].

    Returns
    -------
    list[ConfidenceBucket]
        One bucket per boundary interval, sorted by min_conf.
    """
    if bucket_boundaries is None:
        bucket_boundaries = [0.0, 0.4, 0.7, 0.95, 1.0]

    # Initialize buckets
    buckets: list[dict] = []
    for i in range(len(bucket_boundaries) - 1):
        buckets.append(
            {
                "min_conf": bucket_boundaries[i],
                "max_conf": bucket_boundaries[i + 1],
                "total": 0,
                "correct": 0,
            }
        )

    # Assign predictions to buckets
    for quote_id, (pred_speaker, conf) in predictions.items():
        gold_speaker = gold_labels.get(quote_id)
        if gold_speaker is None:
            continue

        # Find bucket
        for bucket in buckets:
            if bucket["min_conf"] <= conf < bucket["max_conf"]:
                bucket["total"] += 1
                if pred_speaker == gold_speaker:
                    bucket["correct"] += 1
                break

    # Compute accuracy
    result: list[ConfidenceBucket] = []
    for bucket in buckets:
        if bucket["total"] > 0:
            accuracy = bucket["correct"] / bucket["total"]
        else:
            accuracy = 0.0
        result.append(
            ConfidenceBucket(
                min_conf=bucket["min_conf"],
                max_conf=bucket["max_conf"],
                total=bucket["total"],
                correct=bucket["correct"],
                accuracy=accuracy,
            )
        )

    return result


def compute_calibration_curve(
    predictions: dict[str, tuple[str, float]],
    gold_labels: dict[str, str],
    min_threshold: float = 0.0,
    max_threshold: float = 1.0,
    num_points: int = 20,
) -> CalibrationCurve:
    """
    Compute precision vs review fraction across confidence thresholds.

    Parameters
    ----------
    predictions:
        Mapping from quote_id to (predicted_speaker, confidence).
    gold_labels:
        Mapping from quote_id to gold speaker label.
    min_threshold:
        Minimum confidence threshold to test.
    max_threshold:
        Maximum confidence threshold to test.
    num_points:
        Number of threshold points to evaluate.

    Returns
    -------
    CalibrationCurve
        Thresholds, auto-accept precision, and review fraction.
    """
    thresholds = [
        min_threshold + i * (max_threshold - min_threshold) / (num_points - 1)
        for i in range(num_points)
    ]

    auto_accept_precision: list[float] = []
    review_fraction: list[float] = []

    for threshold in thresholds:
        auto_accept_correct = 0
        auto_accept_total = 0
        review_total = 0

        for quote_id, (pred_speaker, conf) in predictions.items():
            gold_speaker = gold_labels.get(quote_id)
            if gold_speaker is None:
                continue

            if conf >= threshold:
                auto_accept_total += 1
                if pred_speaker == gold_speaker:
                    auto_accept_correct += 1
            else:
                review_total += 1

        total = auto_accept_total + review_total
        if total == 0:
            auto_accept_precision.append(0.0)
            review_fraction.append(0.0)
        else:
            precision = auto_accept_correct / auto_accept_total if auto_accept_total > 0 else 0.0
            auto_accept_precision.append(precision)
            review_fraction.append(review_total / total)

    # Bucket stats using default boundaries
    bucket_stats = bucket_by_confidence(predictions, gold_labels)

    return CalibrationCurve(
        thresholds=thresholds,
        auto_accept_precision=auto_accept_precision,
        review_fraction=review_fraction,
        bucket_stats=bucket_stats,
    )


def propose_threshold(
    calibration_curve: CalibrationCurve,
    target_precision: float = 0.95,
    max_review_fraction: float = 0.5,
) -> tuple[float, str]:
    """
    Propose a confidence threshold based on calibration data.

    Parameters
    ----------
    calibration_curve:
        Calibration curve from compute_calibration_curve.
    target_precision:
        Desired precision for auto-accepted lines.
    max_review_fraction:
        Maximum acceptable fraction of lines sent to review.

    Returns
    -------
    tuple
        (proposed_threshold, reasoning_string)
    """
    # Find lowest threshold that meets target precision
    for threshold, precision, review_frac in zip(
        calibration_curve.thresholds,
        calibration_curve.auto_accept_precision,
        calibration_curve.review_fraction,
        strict=False,
    ):
        if precision >= target_precision and review_frac <= max_review_fraction:
            reasoning = (
                f"Threshold {threshold:.2f} achieves {precision:.1%} precision "
                f"with {review_frac:.1%} review load (target: {target_precision:.1%} "
                f"precision, max {max_review_fraction:.1%} review)."
            )
            return threshold, reasoning

    # If no threshold meets both criteria, relax review fraction
    for threshold, precision, review_frac in zip(
        calibration_curve.thresholds,
        calibration_curve.auto_accept_precision,
        calibration_curve.review_fraction,
        strict=False,
    ):
        if precision >= target_precision:
            reasoning = (
                f"Threshold {threshold:.2f} achieves {precision:.1%} precision "
                f"but requires {review_frac:.1%} review (exceeds max {max_review_fraction:.1%}). "
                f"Recommend increasing max_review_fraction or accepting higher review load."
            )
            return threshold, reasoning

    # If no threshold meets precision, return best available
    best_idx = max(
        range(len(calibration_curve.auto_accept_precision)),
        key=lambda i: calibration_curve.auto_accept_precision[i],
    )
    best_threshold = calibration_curve.thresholds[best_idx]
    best_precision = calibration_curve.auto_accept_precision[best_idx]
    best_review = calibration_curve.review_fraction[best_idx]

    reasoning = (
        f"No threshold achieves {target_precision:.1%} precision. "
        f"Best is {best_threshold:.2f} with {best_precision:.1%} precision "
        f"and {best_review:.1%} review. Consider model/prompt improvements."
    )
    return best_threshold, reasoning
