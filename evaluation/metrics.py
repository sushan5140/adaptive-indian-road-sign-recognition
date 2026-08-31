"""Closed-set multiclass classification metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from utils.dependencies import DependencyUnavailableError


@dataclass(frozen=True, slots=True)
class PerClassMetrics:
    """Precision, recall, F1, and support for one configured class."""

    label: str
    index: int
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    """Measured aggregate, per-class, and confusion-matrix results."""

    top1_accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_precision: float
    weighted_recall: float
    weighted_f1: float
    per_class: tuple[PerClassMetrics, ...]
    confusion_matrix: tuple[tuple[int, ...], ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def calculate_classification_metrics(
    true_indices: Sequence[int],
    predicted_indices: Sequence[int],
    class_mapping: Mapping[str, int],
) -> ClassificationMetrics:
    """Calculate safe multiclass metrics with explicit zero-division behavior."""
    metrics = _require_sklearn_metrics()
    if len(true_indices) != len(predicted_indices):
        raise ValueError("true and predicted sequences must have equal length")
    if not true_indices:
        raise ValueError("at least one prediction is required")
    ordered = sorted(class_mapping.items(), key=lambda item: item[1])
    expected_indices = list(range(len(ordered)))
    if [index for _, index in ordered] != expected_indices:
        raise ValueError("class mapping must be contiguous from zero")
    allowed = set(expected_indices)
    if any(index not in allowed for index in true_indices):
        raise ValueError("true indices contain an unknown class")
    if any(index not in allowed for index in predicted_indices):
        raise ValueError("predicted indices contain an unknown class")

    precision, recall, f1, support = metrics.precision_recall_fscore_support(
        true_indices,
        predicted_indices,
        labels=expected_indices,
        average=None,
        zero_division=0,
    )
    macro = metrics.precision_recall_fscore_support(
        true_indices,
        predicted_indices,
        labels=expected_indices,
        average="macro",
        zero_division=0,
    )
    weighted = metrics.precision_recall_fscore_support(
        true_indices,
        predicted_indices,
        labels=expected_indices,
        average="weighted",
        zero_division=0,
    )
    matrix = metrics.confusion_matrix(
        true_indices,
        predicted_indices,
        labels=expected_indices,
    )
    per_class = tuple(
        PerClassMetrics(
            label=label,
            index=index,
            precision=float(precision[index]),
            recall=float(recall[index]),
            f1=float(f1[index]),
            support=int(support[index]),
        )
        for label, index in ordered
    )
    return ClassificationMetrics(
        top1_accuracy=float(metrics.accuracy_score(true_indices, predicted_indices)),
        macro_precision=float(macro[0]),
        macro_recall=float(macro[1]),
        macro_f1=float(macro[2]),
        weighted_precision=float(weighted[0]),
        weighted_recall=float(weighted[1]),
        weighted_f1=float(weighted[2]),
        per_class=per_class,
        confusion_matrix=tuple(
            tuple(int(value) for value in row) for row in matrix.tolist()
        ),
    )


def _require_sklearn_metrics() -> Any:
    try:
        from sklearn import metrics
    except ImportError as error:
        raise DependencyUnavailableError(
            "scikit-learn is required for evaluation metrics"
        ) from error
    return metrics
