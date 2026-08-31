"""Reproducibility controls for Python, NumPy, and PyTorch."""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np

from utils.dependencies import DependencyUnavailableError


def seed_python_numpy(seed: int) -> None:
    """Seed Python and NumPy random-number generators."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, PyTorch CPU, and every available CUDA device.

    Args:
        seed: Non-negative random seed.
        deterministic: Enable deterministic PyTorch algorithms and deterministic
            cuDNN behavior. Deterministic kernels may be slower or unavailable for
            some operations.

    Raises:
        DependencyUnavailableError: If PyTorch is not installed.
    """
    torch = _require_torch()
    seed_python_numpy(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic


def seed_dataloader_worker(worker_id: int) -> None:
    """Seed one DataLoader worker from PyTorch's deterministic worker seed."""
    if worker_id < 0:
        raise ValueError("worker_id must be non-negative")
    torch = _require_torch()
    worker_seed = int(torch.initial_seed() % (2**32))
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise DependencyUnavailableError(
            "PyTorch is required for full reproducibility controls"
        ) from error
    return torch
