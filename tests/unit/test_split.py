"""Tests for deterministic, non-destructive stratified splitting."""

from pathlib import Path
from typing import Callable

import pytest

from data.manifest import DatasetRecord
from data.split import SplitError, save_split_manifest, stratified_split

ImageFactory = Callable[..., Path]


def test_stratified_split_is_reproducible_and_has_no_overlap(
    tmp_path: Path, make_image: ImageFactory
) -> None:
    records = _records(tmp_path / "dataset", make_image, samples_per_class=10)

    first = stratified_split(records, random_seed=123)
    second = stratified_split(records, random_seed=123)

    assert first == second
    paths_by_split = {
        name: {record.image_path for record in split_records}
        for name, split_records in first.splits.items()
    }
    assert paths_by_split["train"].isdisjoint(paths_by_split["validation"])
    assert paths_by_split["train"].isdisjoint(paths_by_split["test"])
    assert paths_by_split["validation"].isdisjoint(paths_by_split["test"])
    assert sum(len(records) for records in first.splits.values()) == 20


def test_small_classes_produce_warning(
    tmp_path: Path, make_image: ImageFactory
) -> None:
    records = _records(tmp_path / "dataset", make_image, samples_per_class=2)

    result = stratified_split(records)

    assert len(result.warnings) == 2


def test_class_with_three_samples_populates_all_nonzero_splits(
    tmp_path: Path, make_image: ImageFactory
) -> None:
    root = tmp_path / "dataset"
    records = []
    for index in range(3):
        path = make_image(root / "stop" / f"{index}.jpg")
        records.append(DatasetRecord(image_path=path, label="stop"))

    result = stratified_split(records)

    assert {name: len(items) for name, items in result.splits.items()} == {
        "train": 1,
        "validation": 1,
        "test": 1,
    }


def test_ratios_must_sum_to_one(tmp_path: Path, make_image: ImageFactory) -> None:
    records = _records(tmp_path / "dataset", make_image, samples_per_class=2)

    with pytest.raises(SplitError, match="sum to 1"):
        stratified_split(records, train_ratio=0.8, validation_ratio=0.2, test_ratio=0.2)


def test_generated_manifest_is_outside_dataset_and_does_not_copy_images(
    tmp_path: Path, make_image: ImageFactory
) -> None:
    root = tmp_path / "dataset"
    records = _records(root, make_image, samples_per_class=4)
    before = sorted(path.relative_to(root) for path in root.rglob("*.jpg"))
    result = stratified_split(records)

    manifest = save_split_manifest(
        result, dataset_root=root, output_dir=tmp_path / "outputs"
    )

    assert manifest.exists()
    assert manifest.read_text(encoding="utf-8").splitlines()[0] == (
        "image_path,label,split"
    )
    assert before == sorted(path.relative_to(root) for path in root.rglob("*.jpg"))
    with pytest.raises(SplitError, match="outside dataset root"):
        save_split_manifest(result, dataset_root=root, output_dir=root / "manifests")


def _records(
    root: Path, make_image: ImageFactory, *, samples_per_class: int
) -> list[DatasetRecord]:
    records = []
    for label in ("stop", "yield"):
        for index in range(samples_per_class):
            path = make_image(root / label / f"{index}.jpg", value=index)
            records.append(DatasetRecord(image_path=path, label=label))
    return records
