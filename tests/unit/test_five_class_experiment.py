"""Tests for strict five-class artifact construction."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from data.five_class_experiment import (
    CLASS_MAPPING,
    FIVE_CLASS_SPECS,
    FiveClassExperimentError,
    build_five_class_external_test,
    build_five_class_training_pool,
)


def test_training_pool_excludes_class_41_and_preserves_fixed_mapping(
    tmp_path: Path,
) -> None:
    path = tmp_path / "six.csv"
    rows = []
    for spec in FIVE_CLASS_SPECS:
        rows.extend(
            _training_row(spec.class_name, spec.dataset_a_class_id, index)
            for index in range(spec.expected_training_count)
        )
    rows.append(_training_row("y_junction", "41", 0))
    _write(path, rows)

    selected = build_five_class_training_pool(path)

    assert len(selected) == 1420
    assert {row["dataset_a_class_id"] for row in selected} == {
        "0",
        "1",
        "16",
        "30",
        "55",
    }
    assert CLASS_MAPPING == {
        "give_way": 0,
        "no_entry": 1,
        "no_right_turn": 2,
        "road_hump": 3,
        "filling_station": 4,
    }


def test_external_manifest_is_unique_and_rejects_y_junction(tmp_path: Path) -> None:
    path = tmp_path / "external.csv"
    rows = []
    offset = 0
    for spec in FIVE_CLASS_SPECS:
        for index in range(spec.expected_external_count):
            offset += 1
            rows.append(
                {
                    "image_path": f"images/{offset}.jpg",
                    "class_name": spec.class_name,
                    "dataset_a_class_id": spec.dataset_a_class_id,
                    "source_image_id": f"image_{offset}.jpg",
                    "perceptual_group_id": f"group_{offset}",
                    "review_id": f"R-{offset}",
                }
            )
    _write(path, rows)
    assert len(build_five_class_external_test(path)) == 117

    rows[0]["class_name"] = "y_junction"
    rows[0]["dataset_a_class_id"] = "41"
    _write(path, rows)
    with pytest.raises(FiveClassExperimentError, match="Y-junction"):
        build_five_class_external_test(path)


def _training_row(label: str, class_id: str, index: int) -> dict[str, str]:
    return {
        "image_path": f"{class_id}/{index}.png",
        "class_name": label,
        "split": "train",
        "dataset_a_class_id": class_id,
        "dataset_a_class_name": label,
        "source_template_id": str(index),
        "augmentation_family_id": f"family_{class_id}",
        "augmentation_generation": "1",
        "exact_content_sha256": f"hash_{class_id}_{index}",
    }


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
