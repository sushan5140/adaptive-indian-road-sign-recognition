"""Tests for one training epoch and validation mode behavior."""

import pytest

torch = pytest.importorskip("torch", reason="PyTorch is required for trainer tests")

from training.checkpoint import CheckpointManager, CheckpointMetadata
from training.trainer import Trainer


def test_one_epoch_training_and_validation() -> None:
    images = torch.randn(8, 4)
    targets = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(images, targets), batch_size=4
    )
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_function=torch.nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
    )

    result = trainer.fit(loader, loader, epochs=1)

    assert len(result.history.records) == 1
    assert 0.0 <= result.history.records[0].val_accuracy <= 1.0
    assert model.training is False


def test_fixed_epoch_training_has_no_validation_or_best_checkpoint(tmp_path) -> None:
    images = torch.randn(4, 4)
    targets = torch.tensor([0, 1, 0, 1])
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(images, targets), batch_size=2
    )
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_function=torch.nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
    )
    manager = CheckpointManager(tmp_path / "checkpoints")
    metadata = CheckpointMetadata(
        class_mapping={"a": 0, "b": 1},
        model_config={"backbone": "linear"},
        preprocessing_config={"image_size": 4},
        random_seed=42,
        training_config={"model_selection": "fixed_epochs_no_validation"},
        project_metadata={"name": "test"},
    )

    result = trainer.fit(
        loader,
        None,
        epochs=1,
        checkpoint_manager=manager,
        checkpoint_metadata=metadata,
    )

    assert result.best_validation_accuracy is None
    assert result.history.records[0].val_loss is None
    assert result.history.records[0].val_accuracy is None
    assert (manager.directory / "last.pt").exists()
    assert not (manager.directory / "best.pt").exists()
