"""Tests for deterministic random-number seeding."""

import random

import numpy as np
import pytest

from utils.reproducibility import seed_everything, seed_python_numpy


def test_python_and_numpy_seeds_are_repeatable() -> None:
    seed_python_numpy(123)
    first = (random.random(), float(np.random.random()))
    seed_python_numpy(123)
    second = (random.random(), float(np.random.random()))

    assert first == second


def test_negative_seed_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        seed_python_numpy(-1)


def test_full_torch_seed_is_repeatable_when_available() -> None:
    torch = pytest.importorskip(
        "torch", reason="PyTorch is required for full reproducibility testing"
    )
    seed_everything(321)
    first = torch.rand(3)
    seed_everything(321)
    second = torch.rand(3)
    assert torch.equal(first, second)
