"""Tests for known multiclass evaluation metrics."""

import pytest

pytest.importorskip(
    "sklearn", reason="scikit-learn is required for evaluation metric tests"
)

from evaluation.metrics import calculate_classification_metrics


def test_known_metrics_and_confusion_shape() -> None:
    metrics = calculate_classification_metrics(
        [0, 0, 1, 1],
        [0, 1, 1, 1],
        {"stop": 0, "yield": 1},
    )

    assert metrics.top1_accuracy == 0.75
    assert len(metrics.per_class) == 2
    assert len(metrics.confusion_matrix) == 2
    assert all(len(row) == 2 for row in metrics.confusion_matrix)
