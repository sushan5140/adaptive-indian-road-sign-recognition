"""Tests for split mapping consistency and DataLoader sampler behavior."""

import csv
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch is required for DataLoader tests")

from data.road_sign_dataset import RoadSignDatasetConfig
from training.dataloaders import (
    DataLoaderConfigurationError,
    LoaderConfig,
    build_dataloaders,
)

ImageFactory = Callable[..., Path]


def test_loader_shuffle_and_shared_class_mapping(
    tmp_path: Path, make_image: ImageFactory
) -> None:
    root = tmp_path / "dataset"
    for split in ("train", "val", "test"):
        for label in ("stop", "yield"):
            make_image(root / split / label / "one.jpg")

    def to_tensor(image: np.ndarray) -> object:
        return torch.from_numpy(image.copy()).permute(2, 0, 1).float()

    loaders = build_dataloaders(
        RoadSignDatasetConfig(root=root, mode="split_directory"),
        LoaderConfig(batch_size=2, seed=7),
        train_transform=to_tensor,
        evaluation_transform=to_tensor,
    )

    assert loaders.class_mapping == {"stop": 0, "yield": 1}
    assert isinstance(loaders.train.sampler, torch.utils.data.RandomSampler)
    assert isinstance(loaders.validation.sampler, torch.utils.data.SequentialSampler)
    assert loaders.test is not None
    assert isinstance(loaders.test.sampler, torch.utils.data.SequentialSampler)


def test_manifest_overlap_between_splits_is_rejected(
    tmp_path: Path, make_image: ImageFactory
) -> None:
    root = tmp_path / "dataset"
    make_image(root / "images" / "one.jpg")
    manifest = root / "samples.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("image_path", "label", "split"))
        writer.writerow(("images/one.jpg", "stop", "train"))
        writer.writerow(("images/one.jpg", "stop", "validation"))

    def to_tensor(image: np.ndarray) -> object:
        return torch.from_numpy(image.copy()).permute(2, 0, 1).float()

    with pytest.raises(DataLoaderConfigurationError, match="overlaps"):
        build_dataloaders(
            RoadSignDatasetConfig(
                root=root,
                mode="csv_manifest",
                manifest_path=manifest,
            ),
            LoaderConfig(batch_size=1),
            train_transform=to_tensor,
            evaluation_transform=to_tensor,
            include_test=False,
        )


def test_training_only_manifest_does_not_require_validation(
    tmp_path: Path, make_image: ImageFactory
) -> None:
    root = tmp_path / "dataset"
    make_image(root / "images" / "one.jpg")
    manifest = root / "samples.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("image_path", "label", "split"))
        writer.writerow(("images/one.jpg", "stop", "train"))

    def to_tensor(image: np.ndarray) -> object:
        return torch.from_numpy(image.copy()).permute(2, 0, 1).float()

    loaders = build_dataloaders(
        RoadSignDatasetConfig(
            root=root,
            mode="csv_manifest",
            manifest_path=manifest,
        ),
        LoaderConfig(batch_size=1),
        train_transform=to_tensor,
        evaluation_transform=to_tensor,
        include_test=False,
        require_validation=False,
    )

    assert loaders.validation is None
    assert loaders.test is None
    assert loaders.dataset_sizes == {"train": 1, "validation": 0, "test": 0}
