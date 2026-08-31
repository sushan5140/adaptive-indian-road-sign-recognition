"""Conservative, configurable transforms for traffic-sign classification."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

from utils.dependencies import DependencyUnavailableError

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True, slots=True)
class TransformConfig:
    """Image preprocessing and conservative augmentation settings."""

    image_size: int = 224
    horizontal_flip_probability: float = 0.0
    max_rotation_degrees: float = 5.0
    brightness: float = 0.15
    contrast: float = 0.15
    normalization_mean: tuple[float, float, float] = IMAGENET_MEAN
    normalization_std: tuple[float, float, float] = IMAGENET_STD

    def validate(self) -> None:
        """Validate transform ranges before constructing torchvision objects."""
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        if not 0.0 <= self.horizontal_flip_probability <= 1.0:
            raise ValueError("horizontal_flip_probability must be in [0, 1]")
        if not 0.0 <= self.max_rotation_degrees <= 15.0:
            raise ValueError("max_rotation_degrees must be in [0, 15]")
        if not 0.0 <= self.brightness <= 1.0 or not 0.0 <= self.contrast <= 1.0:
            raise ValueError("brightness and contrast must be in [0, 1]")
        if len(self.normalization_mean) != 3 or len(self.normalization_std) != 3:
            raise ValueError("normalization mean and standard deviation need 3 values")
        if any(value <= 0.0 for value in self.normalization_std):
            raise ValueError("normalization standard deviations must be positive")


def build_train_transform(config: TransformConfig) -> Callable[[Any], Any]:
    """Build conservative stochastic training preprocessing.

    Horizontal flipping remains disabled by default because it can change traffic
    sign semantics. Vertical flipping is never applied.
    """
    transforms = _require_torchvision_transforms()
    config.validate()
    operations: list[Any] = [
        transforms.ToPILImage(),
        transforms.Resize((config.image_size, config.image_size)),
    ]
    if config.horizontal_flip_probability > 0.0:
        operations.append(
            transforms.RandomHorizontalFlip(config.horizontal_flip_probability)
        )
    if config.max_rotation_degrees > 0.0:
        operations.append(transforms.RandomAffine(degrees=config.max_rotation_degrees))
    if config.brightness > 0.0 or config.contrast > 0.0:
        operations.append(
            transforms.ColorJitter(
                brightness=config.brightness,
                contrast=config.contrast,
            )
        )
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(config.normalization_mean, config.normalization_std),
        ]
    )
    return cast(Callable[[Any], Any], transforms.Compose(operations))


def build_evaluation_transform(config: TransformConfig) -> Callable[[Any], Any]:
    """Build deterministic evaluation preprocessing."""
    transforms = _require_torchvision_transforms()
    config.validate()
    return cast(
        Callable[[Any], Any],
        transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((config.image_size, config.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    config.normalization_mean, config.normalization_std
                ),
            ]
        ),
    )


def _require_torchvision_transforms() -> Any:
    try:
        from torchvision import transforms
    except ImportError as error:
        raise DependencyUnavailableError(
            "torchvision is required to build image transforms"
        ) from error
    return transforms
