"""Tests for auditable training-only class weighting."""

import pytest

from training.class_weights import normalized_inverse_frequency_weights


def test_normalized_inverse_frequency_weights() -> None:
    weights = normalized_inverse_frequency_weights({"b": 20, "a": 10})
    assert list(weights) == ["a", "b"]
    assert weights == pytest.approx({"a": 1.5, "b": 0.75})


@pytest.mark.parametrize("counts", [{}, {"a": 0}, {"": 1}])
def test_invalid_class_counts_are_rejected(counts: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        normalized_inverse_frequency_weights(counts)
