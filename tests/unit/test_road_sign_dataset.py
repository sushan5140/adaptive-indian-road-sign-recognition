"""Tests for the configurable road-sign dataset adapter."""

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

from data.manifest import ManifestError, load_class_mapping
from data.road_sign_dataset import (
    DatasetConfigurationError,
    RoadSignDataset,
    RoadSignDatasetConfig,
)

ImageFactory = Callable[..., Path]


def test_directory_mode_has_alphabetical_mapping_and_preserves_sources(
    tmp_path: Path, make_image: ImageFactory
) -> None:
    root = tmp_path / "dataset"
    first = make_image(root / "yield" / "one.jpg", value=50)
    second = make_image(root / "stop" / "two.jpg", value=100)
    before = {path: _digest(path) for path in (first, second)}

    dataset = RoadSignDataset(
        RoadSignDatasetConfig(root=root, mode="directory", split=None)
    )

    assert dataset.class_to_index == {"stop": 0, "yield": 1}
    assert dataset.index_to_class == {0: "stop", 1: "yield"}
    assert len(dataset) == 2
    image, class_index = dataset[0]
    assert isinstance(image, np.ndarray)
    assert class_index in {0, 1}
    assert before == {path: _digest(path) for path in (first, second)}


def test_split_directory_selects_requested_split(
    tmp_path: Path, make_image: ImageFactory
) -> None:
    root = tmp_path / "dataset"
    make_image(root / "train" / "stop" / "train.jpg")
    make_image(root / "val" / "yield" / "val.jpg")

    dataset = RoadSignDataset(
        RoadSignDatasetConfig(root=root, mode="split_directory", split="val")
    )

    assert len(dataset) == 1
    assert dataset.class_to_index == {"yield": 0}


def test_csv_manifest_metadata_and_transform(
    tmp_path: Path, make_image: ImageFactory
) -> None:
    root = tmp_path / "dataset"
    make_image(root / "images" / "one.jpg")
    manifest = root / "samples.csv"
    _write_csv(manifest, [("images/one.jpg", "stop", "train")])
    called = False

    def transform(image: Any) -> str:
        nonlocal called
        called = True
        assert isinstance(image, np.ndarray)
        return "transformed"

    dataset = RoadSignDataset(
        RoadSignDatasetConfig(
            root=root,
            mode="csv_manifest",
            manifest_path=manifest,
            split="train",
            return_metadata=True,
        ),
        transform=transform,
    )

    image, class_index, metadata = dataset[0]
    assert called
    assert image == "transformed"
    assert class_index == 0
    assert metadata["label"] == "stop"
    assert metadata["relative_image_path"] == "images/one.jpg"
    assert metadata["row_number"] == 2


def test_csv_manifest_preserves_review_id_metadata(
    tmp_path: Path, make_image: ImageFactory
) -> None:
    root = tmp_path / "dataset"
    make_image(root / "images" / "one.jpg")
    manifest = root / "samples.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("image_path", "label", "review_id"))
        writer.writerow(("images/one.jpg", "stop", "B6-0001"))
    dataset = RoadSignDataset(
        RoadSignDatasetConfig(
            root=root,
            mode="csv_manifest",
            manifest_path=manifest,
            split=None,
            return_metadata=True,
        )
    )

    _, _, metadata = dataset[0]

    assert metadata["review_id"] == "B6-0001"


def test_json_manifest_is_supported(tmp_path: Path, make_image: ImageFactory) -> None:
    root = tmp_path / "dataset"
    make_image(root / "images" / "one.png")
    manifest = root / "samples.json"
    manifest.write_text(
        json.dumps(
            {"samples": [{"path": "images/one.png", "name": "stop", "set": "train"}]}
        ),
        encoding="utf-8",
    )

    dataset = RoadSignDataset(
        RoadSignDatasetConfig(
            root=root,
            mode="json_manifest",
            manifest_path=manifest,
            image_path_column="path",
            label_column="name",
            split_column="set",
            split="train",
        )
    )

    assert len(dataset) == 1
    assert dataset.class_to_index == {"stop": 0}


def test_missing_image_has_manifest_row_context(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    manifest = root / "samples.csv"
    _write_csv(manifest, [("missing.jpg", "stop", "train")])

    with pytest.raises(DatasetConfigurationError, match=r"row 2.*does not exist"):
        RoadSignDataset(
            RoadSignDatasetConfig(
                root=root, mode="csv_manifest", manifest_path=manifest
            )
        )


def test_corrupt_image_error_and_skip_policies(
    tmp_path: Path, make_image: ImageFactory
) -> None:
    root = tmp_path / "dataset"
    make_image(root / "stop" / "good.jpg")
    corrupt = root / "stop" / "corrupt.jpg"
    corrupt.write_bytes(b"broken")

    with pytest.raises(DatasetConfigurationError, match="corrupt or unreadable"):
        RoadSignDataset(RoadSignDatasetConfig(root=root, mode="directory"))

    dataset = RoadSignDataset(
        RoadSignDatasetConfig(root=root, mode="directory", corrupt_image_policy="skip")
    )
    assert len(dataset) == 1


def test_unsupported_extension_in_class_directory_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    (root / "stop").mkdir(parents=True)
    (root / "stop" / "image.gif").write_bytes(b"gif")

    with pytest.raises(DatasetConfigurationError, match="Unsupported files"):
        RoadSignDataset(RoadSignDatasetConfig(root=root, mode="directory"))


def test_manifest_path_traversal_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    (tmp_path / "outside.jpg").write_bytes(b"outside")
    manifest = root / "samples.csv"
    _write_csv(manifest, [("../outside.jpg", "stop", "train")])

    with pytest.raises(DatasetConfigurationError, match="outside dataset root"):
        RoadSignDataset(
            RoadSignDatasetConfig(
                root=root, mode="csv_manifest", manifest_path=manifest
            )
        )


def test_explicit_mapping_unknown_label_policies(
    tmp_path: Path, make_image: ImageFactory
) -> None:
    root = tmp_path / "dataset"
    make_image(root / "stop" / "one.jpg")
    make_image(root / "yield" / "two.jpg")
    mapping = {"stop": 0}

    with pytest.raises(DatasetConfigurationError, match="Unknown label"):
        RoadSignDataset(
            RoadSignDatasetConfig(root=root, mode="directory"),
            class_mapping=mapping,
        )
    dataset = RoadSignDataset(
        RoadSignDatasetConfig(root=root, mode="directory", unknown_label_policy="skip"),
        class_mapping=mapping,
    )
    assert len(dataset) == 1
    assert dataset.class_to_index == mapping


def test_duplicate_label_in_mapping_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "mapping.json"
    path.write_text('{"stop": 0, "stop": 1}', encoding="utf-8")

    with pytest.raises(ManifestError, match="duplicate labels"):
        load_class_mapping(path)


def _write_csv(path: Path, rows: list[tuple[str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("image_path", "label", "split"))
        writer.writerows(rows)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
