"""Tests for consistent RGB decoding across source colour modes."""

from pathlib import Path

from PIL import Image

from utils.image_validation import decode_image


def test_bgra_png_is_decoded_as_three_channel_rgb(tmp_path: Path) -> None:
    path = tmp_path / "transparent.png"
    Image.new("RGBA", (4, 3), (10, 20, 30, 40)).save(path)

    decoded = decode_image(path, convert_to_rgb=True)

    assert decoded.shape == (3, 4, 3)
    assert decoded[0, 0].tolist() == [10, 20, 30]
