"""Metrics module for attribution evaluation.

Analogy: like a scorecard that grades a student's test not just on
overall correctness, but on specific skill areas and effort (speed, retries).
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AttributionMetrics:
    """Metrics for a single attribution run."""

    # Accuracy metrics
    overall_accuracy: float  # Correct / total (UNKNOWN counted as wrong)
    speech_only_accuracy: float  # Correct / speech-only (exclude GROUP/NOT_SPEECH)
    accuracy_by_style: dict[str, float]  # Accuracy per style category

    # Error metrics
    unknown_rate: float  # % of quotes labeled UNKNOWN
    fallback_rate: float  # % of quotes where LLM failed

    # JSON validity
    json_validity_before_retry: float  # % valid JSON before any retry
    json_validity_after_retry: float  # % valid JSON after retries
    retries_per_100_quotes: float  # Average retry count per 100 quotes

    # Speed and memory
    seconds_per_1000_words: float
    tokens_per_second: float | None  # If available from Ollama
    peak_ram_mb: float
    peak_ram_llm_mb: float | None  # Ollama process memory

    # Model info
    model_size_gb: float


@dataclass
class ConfusionPair:
    """A pair of characters often confused."""

    predicted: str
    actual: str
    count: int


@dataclass
class AliasMergeMetrics:
    """Metrics for alias merging quality."""

    precision: float  # Correct merges / total merges
    recall: float  # Correct merges / total true pairs
    f1: float  # Harmonic mean


def compute_accuracy(
    predictions: dict[str, str],
    gold_labels: dict[str, str],
    style_tags: dict[str, str] | None = None,
    group_labels: set[str] | None = None,
    not_speech_labels: set[str] | None = None,
) -> tuple[float, float, dict[str, float]]:
    """
    Compute overall accuracy, speech-only accuracy, and accuracy by style.

    Parameters
    ----------
    predictions:
        Mapping from quote_id to predicted speaker.
    gold_labels:
        Mapping from quote_id to gold speaker.
    style_tags:
        Optional mapping from quote_id to style category.
    group_labels:
        Set of label values that count as GROUP.
    not_speech_labels:
        Set of label values that count as NOT_SPEECH.

    Returns
    -------
    tuple
        (overall_accuracy, speech_only_accuracy, accuracy_by_style)
    """
    if not predictions:
        return 0.0, 0.0, {}

    overall_correct = 0
    overall_total = 0
    speech_correct = 0
    speech_total = 0
    style_correct: dict[str, int] = {}
    style_total: dict[str, int] = {}

    for quote_id, predicted_speaker in predictions.items():
        gold_speaker = gold_labels.get(quote_id, "unknown")

        # Overall accuracy (UNKNOWN always wrong)
        overall_total += 1
        if predicted_speaker == gold_speaker:
            overall_correct += 1

        # Speech-only accuracy (exclude GROUP and NOT_SPEECH)
        is_group = group_labels and predicted_speaker in group_labels
        is_not_speech = not_speech_labels and predicted_speaker in not_speech_labels
        gold_is_group = group_labels and gold_speaker in group_labels
        gold_is_not_speech = not_speech_labels and gold_speaker in not_speech_labels

        if not (is_group or is_not_speech or gold_is_group or gold_is_not_speech):
            speech_total += 1
            if predicted_speaker == gold_speaker:
                speech_correct += 1

        # Accuracy by style
        if style_tags:
            style = style_tags.get(quote_id, "unknown")
            style_total[style] = style_total.get(style, 0) + 1
            if predicted_speaker == gold_speaker:
                style_correct[style] = style_correct.get(style, 0) + 1

    overall_acc = overall_correct / overall_total if overall_total > 0 else 0.0
    speech_acc = speech_correct / speech_total if speech_total > 0 else 0.0

    acc_by_style: dict[str, float] = {}
    for style, total in style_total.items():
        correct = style_correct.get(style, 0)
        acc_by_style[style] = correct / total if total > 0 else 0.0

    return overall_acc, speech_acc, acc_by_style


def compute_confusion_matrix(
    predictions: dict[str, str],
    gold_labels: dict[str, str],
    top_n: int = 10,
) -> list[ConfusionPair]:
    """
    Compute confusion matrix and return top N confused pairs.

    Analogy: like a teacher noting which students most often swap names
    on the attendance sheet.

    Returns
    -------
    list[ConfusionPair]
        Top N (predicted, actual) pairs with highest confusion counts.
    """
    confusion: dict[tuple[str, str], int] = {}

    for quote_id, predicted in predictions.items():
        gold = gold_labels.get(quote_id, "unknown")
        if predicted != gold:
            key = (predicted, gold)
            confusion[key] = confusion.get(key, 0) + 1

    # Sort by count descending
    sorted_pairs = sorted(confusion.items(), key=lambda x: x[1], reverse=True)
    top_pairs = [
        ConfusionPair(predicted=pred, actual=gold, count=count)
        for (pred, gold), count in sorted_pairs[:top_n]
    ]

    return top_pairs


def compute_alias_merge_metrics(
    gold_aliases: dict[str, list[str]],
    predicted_aliases: dict[str, list[str]],
) -> AliasMergeMetrics:
    """
    Compute pairwise precision, recall, and F1 for alias merging.

    Analogy: like checking whether two people's contact lists agree on which
    nicknames belong to which full names. Cohen's kappa measures agreement
    between two raters accounting for chance.

    Parameters
    ----------
    gold_aliases:
        Mapping from canonical name to list of aliases (gold standard).
    predicted_aliases:
        Mapping from canonical name to list of aliases (system output).

    Returns
    -------
    AliasMergeMetrics
        Precision, recall, and F1 scores.
    """
    # Flatten to pairs (canonical, alias)
    gold_pairs = set()
    for canonical, aliases in gold_aliases.items():
        for alias in aliases:
            gold_pairs.add((canonical, alias))

    predicted_pairs = set()
    for canonical, aliases in predicted_aliases.items():
        for alias in aliases:
            predicted_pairs.add((canonical, alias))

    # True positives: pairs in both
    true_positives = gold_pairs & predicted_pairs

    # False positives: predicted but not in gold
    false_positives = predicted_pairs - gold_pairs

    # False negatives: in gold but not predicted
    false_negatives = gold_pairs - predicted_pairs

    tp = len(true_positives)
    fp = len(false_positives)
    fn = len(false_negatives)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    return AliasMergeMetrics(precision=precision, recall=recall, f1=f1)


def bootstrap_confidence_interval(
    values: list[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """
    Compute bootstrap confidence interval for a list of values.

    Analogy: like estimating the margin of error in a poll by repeatedly
    resampling the respondents and seeing how much the result varies.

    Parameters
    ----------
    values:
        List of values (e.g., per-paragraph accuracies).
    n_bootstrap:
        Number of bootstrap samples.
    confidence:
        Confidence level (e.g., 0.95 for 95% CI).

    Returns
    -------
    tuple
        (lower_bound, upper_bound) of the confidence interval.
    """
    if not values:
        return 0.0, 0.0

    import random

    n = len(values)
    sample_means = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        sample = random.choices(values, k=n)
        sample_means.append(sum(sample) / n)

    # Sort and compute percentiles
    sample_means.sort()
    alpha = 1 - confidence
    lower_idx = int((alpha / 2) * n_bootstrap)
    upper_idx = int((1 - alpha / 2) * n_bootstrap)

    return sample_means[lower_idx], sample_means[upper_idx]


def compute_labeller_consistency(
    labels_1: dict[str, str],
    labels_2: dict[str, str],
) -> tuple[float, float]:
    """
    Compute agreement percentage and Cohen's kappa between two labelings.

    Analogy: like two teachers grading the same test—agreement is how
    often they give the same grade, kappa accounts for chance agreement.

    Parameters
    ----------
    labels_1, labels_2:
        Two mappings from quote_id to speaker label.

    Returns
    -------
    tuple
        (agreement_percentage, cohens_kappa)
    """
    if not labels_1 or not labels_2:
        return 0.0, 0.0

    # Ensure both have the same keys
    common_keys = set(labels_1.keys()) & set(labels_2.keys())
    if not common_keys:
        return 0.0, 0.0

    n = len(common_keys)
    agreement = sum(1 for k in common_keys if labels_1[k] == labels_2[k])
    agreement_pct = agreement / n

    # Cohen's kappa
    # Compute observed agreement
    p_observed = agreement / n

    # Compute expected agreement by chance
    # Count occurrences of each label in both labelings
    all_labels_1 = [labels_1[k] for k in common_keys]
    all_labels_2 = [labels_2[k] for k in common_keys]

    counter_1 = Counter(all_labels_1)
    counter_2 = Counter(all_labels_2)

    # Expected agreement for each label
    p_expected = 0.0
    for label in set(all_labels_1 + all_labels_2):
        p1 = counter_1.get(label, 0) / n
        p2 = counter_2.get(label, 0) / n
        p_expected += p1 * p2

    # Cohen's kappa
    if p_expected == 1.0:
        kappa = 1.0  # Perfect agreement
    else:
        kappa = (p_observed - p_expected) / (1 - p_expected)

    return agreement_pct, kappa
