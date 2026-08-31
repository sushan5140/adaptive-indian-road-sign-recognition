"""Structured epoch-based supervised training and validation engine."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from training.checkpoint import CheckpointManager, CheckpointMetadata
from training.history import EpochRecord, TrainingHistory
from utils.dependencies import DependencyUnavailableError

LOGGER = logging.getLogger(__name__)


class TrainingError(RuntimeError):
    """Raised when training encounters invalid data or numerical failure."""


@dataclass(frozen=True, slots=True)
class LoopMetrics:
    """Measured average loss and top-1 accuracy for one loader pass."""

    loss: float
    accuracy: float
    sample_count: int


@dataclass(frozen=True, slots=True)
class FitResult:
    """Completed history and optional best validation accuracy."""

    history: TrainingHistory
    best_validation_accuracy: float | None


class Trainer:
    """Train and validate a single-device supervised classifier."""

    def __init__(
        self,
        *,
        model: Any,
        optimizer: Any,
        loss_function: Any,
        device: Any,
        scheduler: Any | None = None,
        gradient_clip_norm: float | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ) -> None:
        if gradient_clip_norm is not None and gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive when provided")
        self.torch = _require_torch()
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.device = device
        self.scheduler = scheduler
        self.gradient_clip_norm = gradient_clip_norm
        self.progress_callback = progress_callback

    def fit(
        self,
        train_loader: Any,
        validation_loader: Any,
        *,
        epochs: int,
        start_epoch: int = 0,
        best_validation_accuracy: float | None = None,
        checkpoint_manager: CheckpointManager | None = None,
        checkpoint_metadata: CheckpointMetadata | None = None,
    ) -> FitResult:
        """Train through ``epochs`` and optionally write last/best checkpoints."""
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        if start_epoch < 0 or start_epoch >= epochs:
            raise ValueError("start_epoch must be in [0, epochs)")
        if (checkpoint_manager is None) != (checkpoint_metadata is None):
            raise ValueError(
                "checkpoint_manager and checkpoint_metadata must be provided together"
            )
        history = TrainingHistory()
        best = (
            best_validation_accuracy
            if best_validation_accuracy is not None
            else (-math.inf if validation_loader is not None else None)
        )
        if (
            validation_loader is None
            and self.scheduler is not None
            and self.scheduler.__class__.__name__ == "ReduceLROnPlateau"
        ):
            raise TrainingError(
                "ReduceLROnPlateau requires validation; use a fixed scheduler"
            )
        for epoch in range(start_epoch, epochs):
            started = time.perf_counter()
            train_metrics = self.train_one_epoch(train_loader)
            validation_metrics = (
                self.evaluate(validation_loader)
                if validation_loader is not None
                else None
            )
            self._step_scheduler(
                validation_metrics.loss if validation_metrics is not None else None
            )
            learning_rate = float(self.optimizer.param_groups[0]["lr"])
            record = EpochRecord(
                epoch=epoch,
                train_loss=train_metrics.loss,
                train_accuracy=train_metrics.accuracy,
                val_loss=(
                    validation_metrics.loss if validation_metrics is not None else None
                ),
                val_accuracy=(
                    validation_metrics.accuracy
                    if validation_metrics is not None
                    else None
                ),
                learning_rate=learning_rate,
                elapsed_seconds=time.perf_counter() - started,
            )
            history.append(record)
            improved = bool(
                validation_metrics is not None
                and best is not None
                and validation_metrics.accuracy > best
            )
            if improved:
                assert validation_metrics is not None
                best = validation_metrics.accuracy
            if checkpoint_manager is not None and checkpoint_metadata is not None:
                checkpoint_manager.save(
                    "last.pt",
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    best_validation_metric=best,
                    metadata=checkpoint_metadata,
                )
                if improved:
                    checkpoint_manager.save(
                        "best.pt",
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        epoch=epoch,
                        best_validation_metric=best,
                        metadata=checkpoint_metadata,
                    )
            self._report_progress(record)
        return FitResult(history=history, best_validation_accuracy=best)

    def train_one_epoch(self, loader: Any) -> LoopMetrics:
        """Run one optimizer-updating training pass."""
        return self._run_loader(loader, training=True)

    def evaluate(self, loader: Any) -> LoopMetrics:
        """Run deterministic evaluation under ``torch.no_grad``."""
        return self._run_loader(loader, training=False)

    def _run_loader(self, loader: Any, *, training: bool) -> LoopMetrics:
        try:
            if len(loader) == 0:
                raise TrainingError("Cannot run an empty DataLoader")
        except TypeError as error:
            raise TrainingError("DataLoader must provide a finite length") from error
        self.model.train(mode=training)
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        gradient_context = (
            self.torch.enable_grad() if training else self.torch.no_grad()
        )
        with gradient_context:
            for batch in loader:
                if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                    raise TrainingError("Each batch must contain images and targets")
                images, targets = batch[0], batch[1]
                images = images.to(self.device)
                targets = targets.to(self.device)
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(images)
                loss = self.loss_function(logits, targets)
                if not bool(self.torch.isfinite(loss).item()):
                    raise TrainingError("Encountered NaN or infinite loss")
                if training:
                    loss.backward()
                    if self.gradient_clip_norm is not None:
                        self.torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.gradient_clip_norm
                        )
                    self.optimizer.step()
                batch_size = int(targets.shape[0])
                predictions = logits.argmax(dim=1)
                total_loss += float(loss.detach().item()) * batch_size
                total_correct += int((predictions == targets).sum().item())
                total_samples += batch_size
        if total_samples == 0:
            raise TrainingError("DataLoader yielded no samples")
        return LoopMetrics(
            loss=total_loss / total_samples,
            accuracy=total_correct / total_samples,
            sample_count=total_samples,
        )

    def _step_scheduler(self, validation_loss: float | None) -> None:
        if self.scheduler is None:
            return
        if self.scheduler.__class__.__name__ == "ReduceLROnPlateau":
            if validation_loss is None:
                raise TrainingError("ReduceLROnPlateau requires validation loss")
            self.scheduler.step(validation_loss)
        else:
            self.scheduler.step()

    def _report_progress(self, record: EpochRecord) -> None:
        validation = (
            f"val_loss={record.val_loss:.6f} val_accuracy={record.val_accuracy:.4f}"
            if record.val_loss is not None and record.val_accuracy is not None
            else "validation=not_run"
        )
        message = (
            f"epoch={record.epoch} train_loss={record.train_loss:.6f} "
            f"train_accuracy={record.train_accuracy:.4f} {validation}"
        )
        LOGGER.info(
            "training_epoch_completed",
            extra={"event": "training_epoch_completed", **asdict(record)},
        )
        if self.progress_callback is not None:
            self.progress_callback(message)


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise DependencyUnavailableError("PyTorch is required for training") from error
    return torch
