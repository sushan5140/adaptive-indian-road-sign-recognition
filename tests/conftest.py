"""Shared temporary-image helpers for dataset tests."""

from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture
def make_image() -> object:
    """Return a helper that writes a deterministic tiny image."""

    def _make_image(
        path: Path,
        *,
        width: int = 8,
        height: int = 6,
        value: int = 127,
        grayscale: bool = False,
    ) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        shape = (height, width) if grayscale else (height, width, 3)
        image = np.full(shape, value, dtype=np.uint8)
        success, encoded = cv2.imencode(path.suffix, image)
        assert success
        encoded.tofile(path)
        return path

    return _make_image
