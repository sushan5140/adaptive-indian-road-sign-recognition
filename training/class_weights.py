"""Auditable class weights derived exclusively from a training manifest."""

from __future__ import annotations

from collections.abc import Mapping


def normalized_inverse_frequency_weights(
    class_counts: Mapping[str, int],
) -> dict[str, float]:
    """Return ``N / (C * n_c)`` weights in deterministic label order."""
    if not class_counts:
        raise ValueError("class_counts must not be empty")
    if any(not label or count <= 0 for label, count in class_counts.items()):
        raise ValueError("class labels must be non-empty and counts must be positive")
    total = sum(class_counts.values())
    class_count = len(class_counts)
    return {
        label: total / (class_count * class_counts[label])
        for label in sorted(class_counts)
    }
