"""Infrastructure-only tiny synthetic training smoke test."""

from pathlib import Path
from typing import Callable

import numpy as np
import pytest

torch = pytest.importorskip(
    "torch", reason="PyTorch is required for synthetic training smoke testing"
)
pytest.importorskip(
    "torchvision", reason="torchvision is required for synthetic training smoke testing"
)
pytest.importorskip("timm", reason="timm is required for model smoke testing")

from data.road_sign_dataset import RoadSignDatasetConfig
from models.factory import ModelConfig, build_classifier
from training.dataloaders import LoaderConfig, build_dataloaders
from training.factories import LossConfig, OptimizerConfig, build_loss, build_optimizer
from training.trainer import Trainer

ImageFactory = Callable[..., Path]


def test_tiny_synthetic_pipeline_runs_one_epoch(
    tmp_path: Path, make_image: ImageFactory
) -> None:
    root = tmp_path / "synthetic_fixture"
    for split in ("train", "val"):
        for label, value in (("circle", 40), ("square", 210)):
            for index in range(2):
                make_image(
                    root / split / label / f"{index}.png",
                    width=32,
                    height=32,
                    value=value,
                )

    def to_tensor(image: np.ndarray) -> object:
        return torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255.0

    loaders = build_dataloaders(
        RoadSignDatasetConfig(root=root, mode="split_directory"),
        LoaderConfig(batch_size=2, seed=42),
        train_transform=to_tensor,
        evaluation_transform=to_tensor,
        include_test=False,
    )
    model = build_classifier(ModelConfig(pretrained=False), loaders.class_mapping)
    optimizer = build_optimizer(
        model.parameters(), OptimizerConfig(learning_rate=0.0001)
    )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_function=build_loss(LossConfig()),
        device=torch.device("cpu"),
    )

    result = trainer.fit(loaders.train, loaders.validation, epochs=1)

    assert len(result.history.records) == 1
    # This assertion validates infrastructure only, not model quality.
    assert result.history.records[0].train_loss >= 0.0
