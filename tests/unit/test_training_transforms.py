"""Tests for conservative train and deterministic evaluation transforms."""

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch is required for transform tests")
pytest.importorskip("torchvision", reason="torchvision is required for transform tests")

from training.transforms import (
    TransformConfig,
    build_evaluation_transform,
    build_train_transform,
)


def test_evaluation_transform_is_deterministic() -> None:
    image = np.full((12, 10, 3), 127, dtype=np.uint8)
    transform = build_evaluation_transform(TransformConfig(image_size=16))

    first = transform(image)
    second = transform(image)

    assert first.shape == (3, 16, 16)
    assert torch.equal(first, second)


def test_default_train_transform_has_no_random_flip() -> None:
    transform = build_train_transform(TransformConfig())
    names = [operation.__class__.__name__ for operation in transform.transforms]

    assert "RandomHorizontalFlip" not in names
    assert "RandomVerticalFlip" not in names
