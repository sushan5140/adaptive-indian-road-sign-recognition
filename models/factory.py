"""Validated factory for the supervised base classifier."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.classifier import RoadSignClassifier


class ModelConfigurationError(ValueError):
    """Raised when model configuration conflicts with the dataset."""


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Configuration for creating a base road-sign classifier."""

    backbone: str = "mobilenetv3_small_100"
    pretrained: bool = False
    num_classes: int | str = "auto"
    dropout: float = 0.2


def build_classifier(
    config: ModelConfig, class_mapping: Mapping[str, int]
) -> RoadSignClassifier:
    """Build a classifier with a class count compatible with the dataset."""
    from models.classifier import RoadSignClassifier

    dataset_classes = len(class_mapping)
    if dataset_classes <= 0:
        raise ModelConfigurationError("Dataset class mapping must not be empty")
    if config.num_classes == "auto":
        num_classes = dataset_classes
    elif isinstance(config.num_classes, int) and not isinstance(
        config.num_classes, bool
    ):
        num_classes = config.num_classes
        if num_classes != dataset_classes:
            raise ModelConfigurationError(
                f"Configured num_classes={num_classes} conflicts with "
                f"dataset class count {dataset_classes}"
            )
    else:
        raise ModelConfigurationError("num_classes must be 'auto' or an integer")
    return RoadSignClassifier(
        num_classes=num_classes,
        backbone_name=config.backbone,
        pretrained=config.pretrained,
        dropout=config.dropout,
    )
