"""Deterministic DataLoader construction from the validated dataset adapter."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from data.road_sign_dataset import (
    DatasetConfigurationError,
    RoadSignDataset,
    RoadSignDatasetConfig,
)
from utils.dependencies import DependencyUnavailableError
from utils.reproducibility import seed_dataloader_worker


class DataLoaderConfigurationError(ValueError):
    """Raised when loaders cannot be created without leakage or relabelling."""


@dataclass(frozen=True, slots=True)
class LoaderConfig:
    """Batching and worker settings shared by all splits."""

    batch_size: int = 32
    num_workers: int = 0
    pin_memory: bool = False
    seed: int = 42


@dataclass(frozen=True, slots=True)
class DataLoaders:
    """Train, optional validation/test loaders, and their shared mapping."""

    train: Any
    validation: Any | None
    test: Any | None
    class_mapping: dict[str, int]
    dataset_sizes: dict[str, int]


def build_dataloaders(
    dataset_config: RoadSignDatasetConfig,
    loader_config: LoaderConfig,
    *,
    train_transform: Callable[[Any], Any],
    evaluation_transform: Callable[[Any], Any],
    explicit_class_mapping: Mapping[str, int] | None = None,
    include_test: bool = True,
    require_validation: bool = True,
    return_metadata: bool = False,
) -> DataLoaders:
    """Build deterministic loaders with one exact class mapping across splits."""
    torch = _require_torch()
    if loader_config.batch_size <= 0:
        raise DataLoaderConfigurationError("batch_size must be positive")
    if loader_config.num_workers < 0:
        raise DataLoaderConfigurationError("num_workers must be non-negative")

    base = replace(dataset_config, return_metadata=return_metadata)
    train_dataset = RoadSignDataset(
        replace(base, split="train"),
        transform=train_transform,
        class_mapping=explicit_class_mapping,
    )
    if all(record.split is None for record in train_dataset.records):
        raise DataLoaderConfigurationError(
            "Training requires predefined split directories or a generated/explicit "
            "manifest; an unsplit class directory would leak samples into validation"
        )
    class_mapping = dict(train_dataset.class_to_index)
    shared_base = replace(base, class_mapping_path=None)
    validation_dataset = (
        _build_required_split(
            shared_base,
            split="validation",
            transform=evaluation_transform,
            class_mapping=class_mapping,
        )
        if require_validation
        else None
    )
    test_dataset = None
    if include_test:
        test_dataset = _build_optional_test_split(
            shared_base,
            transform=evaluation_transform,
            class_mapping=class_mapping,
        )
    if validation_dataset is not None:
        _validate_mapping(
            validation_dataset.class_to_index, class_mapping, "validation"
        )
    if test_dataset is not None:
        _validate_mapping(test_dataset.class_to_index, class_mapping, "test")
    _validate_no_overlap(train_dataset, validation_dataset, test_dataset)

    train_generator = torch.Generator()
    train_generator.manual_seed(loader_config.seed)
    validation_generator = torch.Generator()
    validation_generator.manual_seed(loader_config.seed + 1)
    test_generator = torch.Generator()
    test_generator.manual_seed(loader_config.seed + 2)
    common = {
        "batch_size": loader_config.batch_size,
        "num_workers": loader_config.num_workers,
        "pin_memory": loader_config.pin_memory,
        "worker_init_fn": seed_dataloader_worker,
    }
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        generator=train_generator,
        **common,
    )
    validation_loader = (
        torch.utils.data.DataLoader(
            validation_dataset,
            shuffle=False,
            generator=validation_generator,
            **common,
        )
        if validation_dataset is not None
        else None
    )
    test_loader = (
        torch.utils.data.DataLoader(
            test_dataset,
            shuffle=False,
            generator=test_generator,
            **common,
        )
        if test_dataset is not None
        else None
    )
    return DataLoaders(
        train=train_loader,
        validation=validation_loader,
        test=test_loader,
        class_mapping=class_mapping,
        dataset_sizes={
            "train": len(train_dataset),
            "validation": (
                len(validation_dataset) if validation_dataset is not None else 0
            ),
            "test": len(test_dataset) if test_dataset is not None else 0,
        },
    )


def _build_required_split(
    config: RoadSignDatasetConfig,
    *,
    split: str,
    transform: Callable[[Any], Any],
    class_mapping: Mapping[str, int],
) -> RoadSignDataset:
    try:
        return RoadSignDataset(
            replace(config, split=split),
            transform=transform,
            class_mapping=class_mapping,
        )
    except DatasetConfigurationError as error:
        raise DataLoaderConfigurationError(
            f"Could not build required {split!r} dataset: {error}"
        ) from error


def _build_optional_test_split(
    config: RoadSignDatasetConfig,
    *,
    transform: Callable[[Any], Any],
    class_mapping: Mapping[str, int],
) -> RoadSignDataset | None:
    try:
        return RoadSignDataset(
            replace(config, split="test"),
            transform=transform,
            class_mapping=class_mapping,
        )
    except DatasetConfigurationError as error:
        if "No samples found" in str(
            error
        ) or "Requested split 'test' was not found" in str(error):
            return None
        raise DataLoaderConfigurationError(f"Invalid test dataset: {error}") from error


def _validate_mapping(
    candidate: Mapping[str, int], expected: Mapping[str, int], split: str
) -> None:
    if dict(candidate) != dict(expected):
        raise DataLoaderConfigurationError(
            f"{split.capitalize()} class mapping differs from the training mapping"
        )


def _validate_no_overlap(
    train_dataset: RoadSignDataset,
    validation_dataset: RoadSignDataset | None,
    test_dataset: RoadSignDataset | None,
) -> None:
    owners: dict[Path, str] = {}
    datasets = (
        ("train", train_dataset),
        ("validation", validation_dataset),
        ("test", test_dataset),
    )
    for split_name, dataset in datasets:
        if dataset is None:
            continue
        for record in dataset.records:
            path = record.image_path.resolve()
            if path in owners:
                raise DataLoaderConfigurationError(
                    f"Image path overlaps {owners[path]!r} and {split_name!r}: {path}"
                )
            owners[path] = split_name


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise DependencyUnavailableError(
            "PyTorch is required to construct DataLoaders"
        ) from error
    return torch
