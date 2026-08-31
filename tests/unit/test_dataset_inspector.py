"""Tests for read-only dataset inspection."""

import csv
from pathlib import Path
from typing import Callable

from data.dataset_inspector import DatasetInspector

ImageFactory = Callable[..., Path]


def test_class_directory_discovery_reports_quality(
    tmp_path: Path, make_image: ImageFactory
) -> None:
    root = tmp_path / "dataset"
    make_image(root / "stop" / "one.jpg", width=10, height=5)
    make_image(root / "stop" / "two.png", width=20, height=15)
    make_image(root / "yield" / "one.jpg", width=30, height=25, grayscale=True)
    (root / "empty_class").mkdir(parents=True)
    (root / "stop" / "notes.pdf").write_bytes(b"not an image")
    (root / "annotations.xml").write_text("<annotations/>", encoding="utf-8")

    report = DatasetInspector(root).inspect()

    assert report.detected_mode == "directory"
    assert report.total_image_count == 3
    assert report.number_of_classes == 3
    assert report.class_names == ("empty_class", "stop", "yield")
    assert report.samples_per_class == {"empty_class": 0, "stop": 2, "yield": 1}
    assert report.class_imbalance_exists
    assert report.image_dimensions.minimum == (10, 5)
    assert report.image_dimensions.maximum == (30, 25)
    assert report.image_dimensions.median == (20.0, 15.0)
    assert len(report.empty_class_directories) == 1
    assert len(report.unsupported_files) == 1
    assert len(report.possible_annotation_files) == 1


def test_split_directory_discovery(tmp_path: Path, make_image: ImageFactory) -> None:
    root = tmp_path / "dataset"
    make_image(root / "train" / "stop" / "one.jpg")
    make_image(root / "val" / "stop" / "two.jpg")
    make_image(root / "test" / "yield" / "three.jpg")

    report = DatasetInspector(root).inspect()

    assert report.detected_mode == "split_directory"
    assert report.possible_splits == ("train", "validation", "test")
    assert report.samples_per_class == {"stop": 2, "yield": 1}


def test_flat_directory_is_reported_as_unlabelled(
    tmp_path: Path, make_image: ImageFactory
) -> None:
    root = tmp_path / "dataset"
    make_image(root / "one.jpg")

    report = DatasetInspector(root).inspect()

    assert report.detected_mode == "flat"
    assert report.total_image_count == 1
    assert report.number_of_classes == 0


def test_manifest_reports_duplicates_missing_and_corrupt_images(
    tmp_path: Path, make_image: ImageFactory
) -> None:
    root = tmp_path / "dataset"
    make_image(root / "images" / "good.jpg")
    corrupt = root / "images" / "corrupt.jpg"
    corrupt.write_bytes(b"broken")
    manifest = root / "samples.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("image_path", "label", "split"))
        writer.writerow(("images/good.jpg", "stop", "train"))
        writer.writerow(("images/good.jpg", "stop", "val"))
        writer.writerow(("images/corrupt.jpg", "yield", "train"))
        writer.writerow(("images/missing.jpg", "yield", "test"))

    report = DatasetInspector(root, manifest_path=manifest).inspect()

    assert report.detected_mode == "csv_manifest"
    assert report.total_image_count == 4
    assert len(report.duplicate_file_paths) == 1
    assert len(report.unreadable_files) == 2
    assert report.possible_splits == ("train", "validation", "test")


def test_max_images_limits_decoding(tmp_path: Path, make_image: ImageFactory) -> None:
    root = tmp_path / "dataset"
    make_image(root / "stop" / "one.jpg")
    make_image(root / "stop" / "two.jpg")

    report = DatasetInspector(root, max_images=1).inspect()

    assert report.verified_image_count == 1
    assert report.warnings
