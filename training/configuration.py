"""Typed construction of training configuration objects from resolved YAML."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from data.road_sign_dataset import RoadSignDatasetConfig
from models.factory import ModelConfig
from training.dataloaders import LoaderConfig
from training.factories import (
    LossConfig,
    OptimizerConfig,
    SchedulerConfig,
)
from training.transforms import TransformConfig
from utils.config import ConfigurationError, require_mapping
from utils.image_validation import DEFAULT_IMAGE_EXTENSIONS


def dataset_config_from_yaml(
    config: Mapping[str, Any],
    *,
    root: str | Path,
    split: str | None,
) -> RoadSignDatasetConfig:
    """Build validated dataset configuration from a resolved mapping."""
    section = require_mapping(config, "dataset")
    extensions = section.get("allowed_extensions", DEFAULT_IMAGE_EXTENSIONS)
    if (
        not isinstance(extensions, Sequence)
        or isinstance(extensions, str)
        or any(not isinstance(item, str) for item in extensions)
    ):
        raise ConfigurationError("dataset.allowed_extensions must be string values")
    return RoadSignDatasetConfig(
        root=root,
        mode=_string(section, "mode", "auto"),
        manifest_path=_optional_path(section.get("manifest_path")),
        image_path_column=_string(section, "image_path_column", "image_path"),
        label_column=_string(section, "label_column", "label"),
        split_column=_optional_string(section.get("split_column", "split")),
        split=split,
        allowed_extensions=tuple(extensions),
        convert_to_rgb=_boolean(section, "convert_to_rgb", True),
        corrupt_image_policy=_string(section, "corrupt_image_policy", "error"),
        unknown_label_policy=_string(section, "unknown_label_policy", "error"),
        return_metadata=_boolean(section, "return_metadata", False),
        class_mapping_path=_optional_path(section.get("class_mapping_path")),
    )


def model_config_from_yaml(config: Mapping[str, Any]) -> ModelConfig:
    """Build model factory configuration."""
    section = require_mapping(config, "model")
    raw_classes = section.get("num_classes", "auto")
    if not isinstance(raw_classes, (str, int)) or isinstance(raw_classes, bool):
        raise ConfigurationError("model.num_classes must be 'auto' or an integer")
    return ModelConfig(
        backbone=_string(section, "backbone", "mobilenetv3_small_100"),
        pretrained=_boolean(section, "pretrained", False),
        num_classes=raw_classes,
        dropout=_float(section, "dropout", 0.2),
    )


def transform_config_from_yaml(config: Mapping[str, Any]) -> TransformConfig:
    """Build deterministic input and conservative augmentation settings."""
    input_section = require_mapping(config, "input")
    augmentation = require_mapping(config, "augmentation")
    return TransformConfig(
        image_size=_integer(input_section, "image_size", 224),
        horizontal_flip_probability=_float(
            augmentation, "horizontal_flip_probability", 0.0
        ),
        max_rotation_degrees=_float(augmentation, "max_rotation_degrees", 5.0),
        brightness=_float(augmentation, "brightness", 0.15),
        contrast=_float(augmentation, "contrast", 0.15),
        normalization_mean=_float_triplet(
            input_section.get("normalization_mean", (0.485, 0.456, 0.406)),
            "input.normalization_mean",
        ),
        normalization_std=_float_triplet(
            input_section.get("normalization_std", (0.229, 0.224, 0.225)),
            "input.normalization_std",
        ),
    )


def loader_config_from_yaml(config: Mapping[str, Any], *, seed: int) -> LoaderConfig:
    """Build deterministic loader settings."""
    section = require_mapping(config, "loader")
    return LoaderConfig(
        batch_size=_integer(section, "batch_size", 32),
        num_workers=_integer(section, "num_workers", 0),
        pin_memory=_boolean(section, "pin_memory", False),
        seed=seed,
    )


def loss_config_from_yaml(config: Mapping[str, Any]) -> LossConfig:
    """Build cross-entropy settings."""
    section = require_mapping(config, "training")
    return LossConfig(label_smoothing=_float(section, "label_smoothing", 0.0))


def optimizer_config_from_yaml(config: Mapping[str, Any]) -> OptimizerConfig:
    """Build optimizer settings."""
    section = require_mapping(config, "optimizer")
    return OptimizerConfig(
        name=_string(section, "name", "adamw"),
        learning_rate=_float(section, "learning_rate", 0.001),
        weight_decay=_float(section, "weight_decay", 0.0001),
        momentum=_float(section, "momentum", 0.9),
    )


def scheduler_config_from_yaml(config: Mapping[str, Any]) -> SchedulerConfig:
    """Build scheduler settings."""
    section = require_mapping(config, "scheduler")
    return SchedulerConfig(
        name=_string(section, "name", "cosine"),
        minimum_learning_rate=_float(section, "minimum_learning_rate", 0.000001),
        patience=_integer(section, "patience", 3),
        factor=_float(section, "factor", 0.1),
    )


def _string(section: Mapping[str, Any], key: str, default: str) -> str:
    value = section.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{key} must be a non-empty string")
    return value


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfigurationError("optional configuration value must be a string")
    return value


def _optional_path(value: Any) -> str | Path | None:
    if value is None:
        return None
    if not isinstance(value, (str, Path)):
        raise ConfigurationError("optional path must be a string or path")
    return value


def _boolean(section: Mapping[str, Any], key: str, default: bool) -> bool:
    value = section.get(key, default)
    if not isinstance(value, bool):
        raise ConfigurationError(f"{key} must be boolean")
    return value


def _integer(section: Mapping[str, Any], key: str, default: int) -> int:
    value = section.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{key} must be an integer")
    return int(value)


def _float(section: Mapping[str, Any], key: str, default: float) -> float:
    value = section.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{key} must be numeric")
    return float(value)


def _float_triplet(value: Any, name: str) -> tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, str) or len(value) != 3:
        raise ConfigurationError(f"{name} must contain three numbers")
    if any(
        isinstance(item, bool) or not isinstance(item, (int, float)) for item in value
    ):
        raise ConfigurationError(f"{name} must contain three numbers")
    return (float(value[0]), float(value[1]), float(value[2]))
