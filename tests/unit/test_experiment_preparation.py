"""Tests for deterministic six-class experiment artifact preparation."""

from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image

from data.experiment_preparation import (
    SIX_CLASS_SPECS,
    build_dataset_a_training_pool,
    build_review_rows,
    create_contact_sheets,
    derive_template_family,
)


def test_review_rows_are_filtered_unique_and_stable(tmp_path: Path) -> None:
    manifest = tmp_path / "classification.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "image_path",
                "class_name",
                "source_question",
                "source_answer",
                "source_image_id",
                "leakage_group_id",
            ),
        )
        writer.writeheader()
        writer.writerows(
            (
                {
                    "image_path": "traffic512final/b.jpg",
                    "class_name": "give_way",
                    "source_question": "What sign?",
                    "source_answer": "Give way",
                    "source_image_id": "b.jpg",
                    "leakage_group_id": "group_2",
                },
                {
                    "image_path": "traffic512final/a.jpg",
                    "class_name": "filling_station",
                    "source_question": "What sign?",
                    "source_answer": "Fuel",
                    "source_image_id": "a.jpg",
                    "leakage_group_id": "group_1",
                },
                {
                    "image_path": "traffic512final/c.jpg",
                    "class_name": "stop",
                    "source_question": "What sign?",
                    "source_answer": "Stop",
                    "source_image_id": "c.jpg",
                    "leakage_group_id": "group_3",
                },
            )
        )

    rows = build_review_rows(manifest)

    assert [row["review_id"] for row in rows] == ["B6-0001", "B6-0002"]
    assert [row["proposed_class"] for row in rows] == [
        "filling_station",
        "give_way",
    ]
    assert all(row["review_status"] == "pending" for row in rows)


def test_contact_sheet_uses_source_without_modifying_it(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    image_path = root / "traffic512final" / "a.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (40, 30), "red").save(image_path)
    original = image_path.read_bytes()
    rows = [
        {
            "review_id": "B6-0001",
            "image_path": "traffic512final/a.jpg",
            "source_image_id": "a.jpg",
            "proposed_class": "give_way",
            "perceptual_group_id": "group_1",
        }
    ]

    sheets = create_contact_sheets(
        rows, dataset_root=root, output_directory=tmp_path / "sheets"
    )

    assert len(sheets) == 1
    assert sheets[0].name == "give_way_01.jpg"
    assert image_path.read_bytes() == original


def test_dataset_a_pool_records_one_template_per_class(tmp_path: Path) -> None:
    root = tmp_path / "images"
    for spec in SIX_CLASS_SPECS:
        class_dir = root / spec.dataset_a_class_id
        class_dir.mkdir(parents=True)
        Image.new("L", (4, 4), 1).save(class_dir / f"{spec.source_template_id}.png")
        Image.new("L", (4, 4), 2).save(
            class_dir
            / f"{spec.dataset_a_class_id}_original_{spec.source_template_id}.png_x.png"
        )

    rows, report = build_dataset_a_training_pool(root)

    assert len(rows) == 12
    assert report["source_groups_per_class"] == 1
    assert report["independent_validation_possible"] is False
    assert {row["split"] for row in rows} == {"train"}


def test_template_family_tracks_recursive_augmentation_depth() -> None:
    assert derive_template_family("43.png") == ("43", 0)
    assert derive_template_family("0_original_43.png_x.png") == ("43", 1)
    assert derive_template_family("0_original_43_original_43.png_x.png") == (
        "43",
        2,
    )
