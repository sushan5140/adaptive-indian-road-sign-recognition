"""Safe image-path and image-content validation helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

DEFAULT_IMAGE_EXTENSIONS: tuple[str, ...] = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
)


class ImageValidationError(ValueError):
    """Raised when an image path or image payload is invalid."""


@dataclass(frozen=True, slots=True)
class ImageInfo:
    """Lightweight properties read from one decoded image."""

    width: int
    height: int
    colour_mode: str


def extended_length_path(path: str | Path) -> Path:
    """Resolve a path and add the Windows extended-length prefix when needed."""
    resolved = Path(path).expanduser().resolve()
    value = str(resolved)
    if os.name != "nt" or value.startswith("\\\\?\\"):
        return resolved
    if value.startswith("\\\\"):
        return Path(f"\\\\?\\UNC\\{value[2:]}")
    return Path(f"\\\\?\\{value}")


def normalize_extensions(extensions: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Normalize and validate a collection of file extensions."""
    normalized: list[str] = []
    for extension in extensions:
        if not isinstance(extension, str) or not extension.strip():
            raise ValueError("allowed extensions must be non-empty strings")
        value = extension.strip().lower()
        if not value.startswith("."):
            value = f".{value}"
        if value not in normalized:
            normalized.append(value)
    if not normalized:
        raise ValueError("at least one image extension must be allowed")
    return tuple(normalized)


def resolve_path_within_root(root: str | Path, candidate: str | Path) -> Path:
    """Resolve a path and reject traversal outside ``root``.

    The path does not need to exist, allowing callers to produce useful missing-file
    errors after containment has been established.
    """
    resolved_root = Path(root).expanduser().resolve()
    candidate_path = Path(candidate).expanduser()
    resolved_candidate = (
        candidate_path.resolve()
        if candidate_path.is_absolute()
        else (resolved_root / candidate_path).resolve()
    )
    if not resolved_candidate.is_relative_to(resolved_root):
        raise ImageValidationError(
            f"Path {candidate!s} resolves outside dataset root {resolved_root}"
        )
    return resolved_candidate


def validate_image_path(
    path: str | Path,
    *,
    root: str | Path,
    allowed_extensions: tuple[str, ...] | list[str] = DEFAULT_IMAGE_EXTENSIONS,
    require_exists: bool = True,
) -> Path:
    """Validate containment, extension, and optional existence for an image path."""
    resolved = resolve_path_within_root(root, path)
    normalized_extensions = normalize_extensions(allowed_extensions)
    if resolved.suffix.lower() not in normalized_extensions:
        raise ImageValidationError(
            f"Unsupported image extension {resolved.suffix!r} for {resolved}"
        )
    if require_exists and (not resolved.exists() or not resolved.is_file()):
        raise ImageValidationError(f"Image file does not exist: {resolved}")
    return resolved


def decode_image(path: str | Path, *, convert_to_rgb: bool = True) -> np.ndarray:
    """Decode one image without modifying it on disk."""
    image_path = Path(path)
    try:
        encoded = np.fromfile(image_path, dtype=np.uint8)
    except OSError as error:
        raise ImageValidationError(f"Could not read image file {image_path}") from error
    if encoded.size == 0:
        raise ImageValidationError(f"Image file is empty or unreadable: {image_path}")
    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ImageValidationError(f"Image is corrupt or unreadable: {image_path}")
    if convert_to_rgb:
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return np.asarray(image)


def inspect_image(path: str | Path) -> ImageInfo:
    """Decode one image and return dimensions and its source colour mode."""
    image = decode_image(path, convert_to_rgb=False)
    height, width = image.shape[:2]
    if image.ndim == 2:
        mode = "grayscale"
    elif image.shape[2] == 4:
        mode = "BGRA"
    elif image.shape[2] == 3:
        mode = "BGR"
    else:
        mode = f"{image.shape[2]}-channel"
    return ImageInfo(width=int(width), height=int(height), colour_mode=mode)
