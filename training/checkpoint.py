"""Atomic base-model checkpoint save and validated resume support."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from utils.dependencies import DependencyUnavailableError


class CheckpointError(RuntimeError):
    """Raised when a checkpoint is invalid, incompatible, or cannot be saved."""


@dataclass(frozen=True, slots=True)
class CheckpointMetadata:
    """Configuration and identity recorded with every checkpoint."""

    class_mapping: dict[str, int]
    model_config: dict[str, Any]
    preprocessing_config: dict[str, Any]
    random_seed: int
    training_config: dict[str, Any]
    project_metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ResumeState:
    """Validated state needed to continue at the next epoch."""

    next_epoch: int
    best_validation_metric: float | None
    metadata: CheckpointMetadata


class CheckpointManager:
    """Save ``last.pt`` and ``best.pt`` outside the immutable dataset root."""

    def __init__(
        self,
        directory: str | Path,
        *,
        dataset_root: str | Path | None = None,
    ) -> None:
        self.directory = Path(directory).expanduser().resolve()
        if dataset_root is not None:
            root = Path(dataset_root).expanduser().resolve()
            if self.directory == root or self.directory.is_relative_to(root):
                raise CheckpointError(
                    f"Checkpoint directory must be outside dataset root {root}"
                )
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        filename: str,
        *,
        model: Any,
        optimizer: Any,
        scheduler: Any | None,
        epoch: int,
        best_validation_metric: float | None,
        metadata: CheckpointMetadata,
    ) -> Path:
        """Atomically save one complete training checkpoint."""
        torch = _require_torch()
        if filename not in {"last.pt", "best.pt"}:
            raise CheckpointError("Checkpoint filename must be last.pt or best.pt")
        if epoch < 0:
            raise CheckpointError("Checkpoint epoch must be non-negative")
        payload = {
            "schema_version": 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": (
                scheduler.state_dict() if scheduler is not None else None
            ),
            "epoch": epoch,
            "best_validation_metric": (
                float(best_validation_metric)
                if best_validation_metric is not None
                else None
            ),
            "class_mapping": dict(metadata.class_mapping),
            "model_config": dict(metadata.model_config),
            "preprocessing_config": dict(metadata.preprocessing_config),
            "random_seed": metadata.random_seed,
            "training_config": dict(metadata.training_config),
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "project_metadata": dict(metadata.project_metadata),
        }
        destination = self.directory / filename
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=self.directory,
                prefix=f".{filename}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
            torch.save(payload, temporary_path)
            os.replace(temporary_path, destination)
        except (OSError, RuntimeError) as error:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise CheckpointError(f"Could not save checkpoint {destination}") from error
        return destination


def load_checkpoint(
    path: str | Path,
    *,
    model: Any,
    optimizer: Any | None = None,
    scheduler: Any | None = None,
    expected_class_mapping: Mapping[str, int] | None = None,
    expected_model_config: Mapping[str, Any] | None = None,
    map_location: Any = "cpu",
) -> ResumeState:
    """Load state dictionaries after validating mapping and architecture."""
    payload = read_checkpoint_payload(path, map_location=map_location)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise CheckpointError("Unsupported or malformed checkpoint schema")
    required = {
        "model_state_dict",
        "optimizer_state_dict",
        "epoch",
        "best_validation_metric",
        "class_mapping",
        "model_config",
        "preprocessing_config",
        "random_seed",
        "training_config",
        "project_metadata",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise CheckpointError(f"Checkpoint is missing fields: {missing}")
    class_mapping = _integer_mapping(payload["class_mapping"], "class_mapping")
    model_config = _string_mapping(payload["model_config"], "model_config")
    if expected_class_mapping is not None and class_mapping != dict(
        expected_class_mapping
    ):
        raise CheckpointError("Checkpoint class mapping is incompatible")
    if expected_model_config is not None and model_config != dict(
        expected_model_config
    ):
        raise CheckpointError("Checkpoint model architecture is incompatible")
    try:
        model.load_state_dict(payload["model_state_dict"], strict=True)
        if optimizer is not None:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler_state = payload.get("scheduler_state_dict")
        if scheduler is not None and scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)
    except (RuntimeError, ValueError, KeyError) as error:
        raise CheckpointError(
            "Checkpoint state dictionaries are incompatible"
        ) from error
    epoch = payload["epoch"]
    best_metric = payload["best_validation_metric"]
    if not isinstance(epoch, int) or epoch < 0:
        raise CheckpointError("Checkpoint epoch is invalid")
    if best_metric is not None and not isinstance(best_metric, (int, float)):
        raise CheckpointError("Checkpoint best metric must be numeric or null")
    metadata = CheckpointMetadata(
        class_mapping=class_mapping,
        model_config=model_config,
        preprocessing_config=_string_mapping(
            payload["preprocessing_config"], "preprocessing_config"
        ),
        random_seed=int(payload["random_seed"]),
        training_config=_string_mapping(payload["training_config"], "training_config"),
        project_metadata=_string_mapping(
            payload["project_metadata"], "project_metadata"
        ),
    )
    return ResumeState(
        next_epoch=epoch + 1,
        best_validation_metric=(
            float(best_metric) if best_metric is not None else None
        ),
        metadata=metadata,
    )


def read_checkpoint_payload(
    path: str | Path, *, map_location: Any = "cpu"
) -> dict[str, Any]:
    """Read a checkpoint payload without constructing a model."""
    torch = _require_torch()
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.exists() or not checkpoint_path.is_file():
        raise CheckpointError(f"Checkpoint does not exist: {checkpoint_path}")
    try:
        try:
            payload = torch.load(
                checkpoint_path,
                map_location=map_location,
                weights_only=True,
            )
        except TypeError:
            payload = torch.load(checkpoint_path, map_location=map_location)
    except (OSError, RuntimeError, ValueError) as error:
        raise CheckpointError(f"Could not load checkpoint {checkpoint_path}") from error
    if not isinstance(payload, dict):
        raise CheckpointError("Checkpoint payload must be a mapping")
    if payload.get("schema_version") != 1:
        raise CheckpointError("Unsupported or malformed checkpoint schema")
    return payload


def metadata_to_dict(metadata: CheckpointMetadata) -> dict[str, Any]:
    """Return checkpoint metadata as a serializable mapping."""
    return asdict(metadata)


def _integer_mapping(value: Any, name: str) -> dict[str, int]:
    if not isinstance(value, dict) or any(
        not isinstance(key, str) or isinstance(item, bool) or not isinstance(item, int)
        for key, item in value.items()
    ):
        raise CheckpointError(f"Checkpoint {name} must map strings to integers")
    return dict(value)


def _string_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise CheckpointError(f"Checkpoint {name} must be a string-keyed mapping")
    return dict(value)


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise DependencyUnavailableError(
            "PyTorch is required for checkpoint operations"
        ) from error
    return torch
