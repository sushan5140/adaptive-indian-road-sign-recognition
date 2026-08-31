"""Tests for atomic checkpoints and validated resume."""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="PyTorch is required for checkpoint tests")

from training.checkpoint import (
    CheckpointError,
    CheckpointManager,
    CheckpointMetadata,
    load_checkpoint,
)


def test_checkpoint_round_trip_and_resume_epoch(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    metadata = _metadata()
    manager = CheckpointManager(tmp_path / "checkpoints")
    path = manager.save(
        "last.pt",
        model=model,
        optimizer=optimizer,
        scheduler=None,
        epoch=2,
        best_validation_metric=0.75,
        metadata=metadata,
    )

    state = load_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        expected_class_mapping=metadata.class_mapping,
        expected_model_config=metadata.model_config,
    )

    assert state.next_epoch == 3
    assert state.best_validation_metric == 0.75


def test_incompatible_mapping_is_rejected(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    manager = CheckpointManager(tmp_path / "checkpoints")
    path = manager.save(
        "best.pt",
        model=model,
        optimizer=optimizer,
        scheduler=None,
        epoch=0,
        best_validation_metric=0.5,
        metadata=_metadata(),
    )

    with pytest.raises(CheckpointError, match="class mapping"):
        load_checkpoint(
            path,
            model=model,
            expected_class_mapping={"different": 0},
        )


def test_checkpoint_supports_no_validation_metric(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    manager = CheckpointManager(tmp_path / "checkpoints")
    path = manager.save(
        "last.pt",
        model=model,
        optimizer=optimizer,
        scheduler=None,
        epoch=0,
        best_validation_metric=None,
        metadata=_metadata(),
    )

    state = load_checkpoint(path, model=model, optimizer=optimizer)

    assert state.best_validation_metric is None


def _metadata() -> CheckpointMetadata:
    return CheckpointMetadata(
        class_mapping={"stop": 0, "yield": 1},
        model_config={"backbone": "test", "num_classes": 2},
        preprocessing_config={"image_size": 8},
        random_seed=42,
        training_config={"epochs": 3},
        project_metadata={"version": "test"},
    )
