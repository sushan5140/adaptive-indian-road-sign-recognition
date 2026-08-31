"""Explicit loss, optimizer, and scheduler factories."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from utils.dependencies import DependencyUnavailableError


class TrainingConfigurationError(ValueError):
    """Raised when loss, optimizer, or scheduler configuration is invalid."""


@dataclass(frozen=True, slots=True)
class LossConfig:
    """Cross-entropy loss settings."""

    label_smoothing: float = 0.0


@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    """AdamW or SGD optimizer settings."""

    name: str = "adamw"
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    momentum: float = 0.9


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    """Optional scheduler settings."""

    name: str = "cosine"
    minimum_learning_rate: float = 0.000001
    patience: int = 3
    factor: float = 0.1


def build_loss(config: LossConfig) -> Any:
    """Build cross-entropy loss with optional label smoothing."""
    torch = _require_torch()
    if not 0.0 <= config.label_smoothing < 1.0:
        raise TrainingConfigurationError("label_smoothing must be in [0, 1)")
    return torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)


def build_optimizer(parameters: Iterable[Any], config: OptimizerConfig) -> Any:
    """Build AdamW or SGD using explicit hyperparameters."""
    torch = _require_torch()
    if config.learning_rate <= 0.0:
        raise TrainingConfigurationError("learning_rate must be positive")
    if config.weight_decay < 0.0:
        raise TrainingConfigurationError("weight_decay must be non-negative")
    name = config.name.strip().lower()
    if name == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    if name == "sgd":
        if not 0.0 <= config.momentum < 1.0:
            raise TrainingConfigurationError("SGD momentum must be in [0, 1)")
        return torch.optim.SGD(
            parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            momentum=config.momentum,
        )
    raise TrainingConfigurationError("optimizer name must be 'adamw' or 'sgd'")


def build_scheduler(
    optimizer: Any, config: SchedulerConfig, *, epochs: int
) -> Any | None:
    """Build no scheduler, cosine annealing, or ReduceLROnPlateau."""
    torch = _require_torch()
    name = config.name.strip().lower()
    if name in {"none", "off"}:
        return None
    if name in {"cosine", "cosine_annealing"}:
        if epochs <= 0:
            raise TrainingConfigurationError("epochs must be positive")
        if config.minimum_learning_rate < 0.0:
            raise TrainingConfigurationError(
                "minimum_learning_rate must be non-negative"
            )
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=config.minimum_learning_rate,
        )
    if name in {"plateau", "reduce_on_plateau"}:
        if config.patience < 0 or not 0.0 < config.factor < 1.0:
            raise TrainingConfigurationError(
                "plateau patience must be non-negative and factor in (0, 1)"
            )
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=config.patience,
            factor=config.factor,
            min_lr=config.minimum_learning_rate,
        )
    raise TrainingConfigurationError(
        "scheduler name must be none, cosine, or reduce_on_plateau"
    )


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise DependencyUnavailableError(
            "PyTorch is required for training factories"
        ) from error
    return torch
