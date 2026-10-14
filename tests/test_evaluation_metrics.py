"""Unit tests for evaluation metrics module."""

from ebook2audiobook.evaluation.metrics import (
    bootstrap_confidence_interval,
    compute_accuracy,
    compute_alias_merge_metrics,
    compute_confusion_matrix,
    compute_labeller_consistency,
)


def test_compute_accuracy_basic():
    """Test basic accuracy calculation."""
    predictions = {"Q1": "alice", "Q2": "bob", "Q3": "alice"}
    gold_labels = {"Q1": "alice", "Q2": "alice", "Q3": "alice"}

    overall, speech, by_style = compute_accuracy(predictions, gold_labels)

    assert overall == 2 / 3  # Q1 and Q3 correct
    assert speech == 2 / 3
    assert by_style == {}  # No style tags


def test_compute_accuracy_with_style_tags():
    """Test accuracy by style category."""
    predictions = {"Q1": "alice", "Q2": "bob", "Q3": "alice"}
    gold_labels = {"Q1": "alice", "Q2": "alice", "Q3": "alice"}
    style_tags = {"Q1": "tagged", "Q2": "untagged", "Q3": "tagged"}

    overall, speech, by_style = compute_accuracy(predictions, gold_labels, style_tags)

    assert overall == 2 / 3
    assert by_style["tagged"] == 1.0  # Both tagged quotes correct
    assert by_style["untagged"] == 0.0  # Untagged quote wrong


def test_compute_accuracy_exclude_group_not_speech():
    """Test speech-only accuracy excludes GROUP and NOT_SPEECH."""
    predictions = {
        "Q1": "alice",
        "Q2": "GROUP",
        "Q3": "NOT_SPEECH",
        "Q4": "bob",
    }
    gold_labels = {
        "Q1": "alice",
        "Q2": "GROUP",
        "Q3": "NOT_SPEECH",
        "Q4": "alice",
    }
    group_labels = {"GROUP"}
    not_speech_labels = {"NOT_SPEECH"}

    overall, speech, by_style = compute_accuracy(
        predictions, gold_labels, group_labels=group_labels, not_speech_labels=not_speech_labels
    )

    assert overall == 3 / 4  # Q2 and Q3 correct as special labels
    assert speech == 1 / 2  # Only Q1 and Q4 counted, Q1 correct


def test_compute_confusion_matrix():
    """Test confusion matrix computation."""
    from ebook2audiobook.evaluation.metrics import ConfusionPair

    predictions = {"Q1": "alice", "Q2": "bob", "Q3": "alice", "Q4": "bob"}
    gold_labels = {"Q1": "alice", "Q2": "alice", "Q3": "bob", "Q4": "alice"}

    confusion = compute_confusion_matrix(predictions, gold_labels, top_n=5)

    # Should capture (bob, alice) twice and (alice, bob) twice
    assert len(confusion) <= 5
    for pair in confusion:
        assert isinstance(pair, ConfusionPair)
        assert pair.count >= 1


def test_compute_alias_merge_metrics():
    """Test alias merge precision/recall/F1."""
    gold_aliases = {"alice": ["ali", "ms_a"], "bob": ["bobby"]}
    predicted_aliases = {"alice": ["ali"], "bob": ["bobby", "robert"]}

    metrics = compute_alias_merge_metrics(gold_aliases, predicted_aliases)

    # True positives: (alice, ali), (bob, bobby)
    # False positives: (bob, robert)
    # False negatives: (alice, ms_a)
    assert 0.0 <= metrics.precision <= 1.0
    assert 0.0 <= metrics.recall <= 1.0
    assert 0.0 <= metrics.f1 <= 1.0


def test_compute_labeller_consistency():
    """Test labeller consistency (agreement and Cohen's kappa)."""
    labels_1 = {"Q1": "alice", "Q2": "bob", "Q3": "alice"}
    labels_2 = {"Q1": "alice", "Q2": "bob", "Q3": "bob"}

    agreement, kappa = compute_labeller_consistency(labels_1, labels_2)

    assert agreement == 2 / 3  # Q1 and Q2 agree
    assert 0.0 <= kappa <= 1.0  # Kappa between 0 and 1


def test_compute_labeller_consistency_perfect():
    """Test perfect agreement gives kappa = 1.0."""
    labels_1 = {"Q1": "alice", "Q2": "bob"}
    labels_2 = {"Q1": "alice", "Q2": "bob"}

    agreement, kappa = compute_labeller_consistency(labels_1, labels_2)

    assert agreement == 1.0
    assert kappa == 1.0


def test_bootstrap_confidence_interval():
    """Test bootstrap CI computation."""
    values = [0.8, 0.9, 0.85, 0.95, 0.88]

    lower, upper = bootstrap_confidence_interval(values, n_bootstrap=100)

    assert 0.0 <= lower <= 1.0
    assert 0.0 <= upper <= 1.0
    assert lower <= upper


def test_bootstrap_confidence_interval_empty():
    """Test bootstrap CI with empty list."""
    lower, upper = bootstrap_confidence_interval([])

    assert lower == 0.0
    assert upper == 0.0
