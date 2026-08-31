"""Tests for conservative Indian Traffic VQA auditing."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import pytest
from PIL import Image

from data.vqa_audit import VqaAuditConfig, VqaAuditError, VqaDatasetAuditor


def _write_csv(path: Path, rows: list[tuple[str, str, str, str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("id", "image_name", "question", "answer", "traffic  sign"))
        writer.writerows(rows)


def _build_dataset(tmp_path: Path) -> VqaAuditConfig:
    root = tmp_path / "dataset"
    images = root / "traffic512final"
    images.mkdir(parents=True)
    Image.new("RGB", (12, 10), color=(255, 0, 0)).save(images / "img_0001.jpg")
    shutil.copyfile(images / "img_0001.jpg", images / "img_0002.jpg")
    striped = Image.new("RGB", (12, 10))
    striped.putdata(
        [
            (255, 255, 255) if column % 2 else (0, 0, 0)
            for _row in range(10)
            for column in range(12)
        ]
    )
    striped.save(images / "img_0003.jpg")
    compact_rows = [
        (
            "1",
            "img_0001.jpg",
            "What does this traffic sign indicate?",
            "Stop",
            "Mandatory",
        ),
        (
            "2",
            "img_0002.jpg",
            "What does this traffic sign indicate?",
            "Stop",
            "Mandatory",
        ),
        (
            "3",
            "img_0003.jpg",
            "What does this traffic sign indicate?",
            "Speed Limit",
            "Mandatory",
        ),
    ]
    full_rows = compact_rows + [
        ("1", "img_0001.jpg", "What should the driver do?", "Stop", "Mandatory"),
        ("2", "img_0002.jpg", "What should the driver do?", "Stop", "Mandatory"),
        (
            "3",
            "img_0003.jpg",
            "What speed limit does this traffic sign indicate?",
            "40 km/h",
            "Mandatory",
        ),
    ]
    _write_csv(root / "traffic_vqa_1085.csv", compact_rows)
    _write_csv(root / "traffic_vqa_4341.csv", full_rows)
    return VqaAuditConfig(
        dataset_root=root,
        images_directory=images,
        compact_csv=root / "traffic_vqa_1085.csv",
        full_csv=root / "traffic_vqa_4341.csv",
        minimum_class_images=1,
        minimum_independent_groups=1,
        near_duplicate_hamming_distance=0,
    )


def test_audit_derives_one_label_per_photo_and_specific_speed(tmp_path: Path) -> None:
    result = VqaDatasetAuditor(_build_dataset(tmp_path)).audit()

    assert len(result.candidate_manifest) == 3
    labels = {str(row["label"]) for row in result.candidate_manifest}
    assert labels == {"stop", "maximum_speed_limit_40_km_h"}
    assert len({row["image_id"] for row in result.candidate_manifest}) == 3
    assert result.class_mapping == {"maximum_speed_limit_40_km_h": 0, "stop": 1}
    assert result.report["compact_csv"]["is_subset_of_full_csv"] is True


def test_audit_reports_exact_duplicates_and_normalizes_category(tmp_path: Path) -> None:
    result = VqaDatasetAuditor(_build_dataset(tmp_path)).audit()

    assert len(result.exact_duplicate_groups) == 1
    assert result.exact_duplicate_groups[0]["images"] == [
        "img_0001.jpg",
        "img_0002.jpg",
    ]
    assert result.report["annotation_quality"]["category_frequency"] == {"Mandatory": 6}


def test_unsupported_identity_answer_is_excluded(tmp_path: Path) -> None:
    config = _build_dataset(tmp_path)
    rows = [
        (
            "1",
            "img_0001.jpg",
            "What does this traffic sign indicate?",
            "Directions",
            "Informatory",
        )
    ]
    _write_csv(config.compact_csv, rows)
    _write_csv(config.full_csv, rows)

    result = VqaDatasetAuditor(config).audit()

    assert result.candidate_manifest == ()
    assert result.excluded_candidates[0]["reason"] == (
        "unsupported_or_ambiguous_identity_answer"
    )


def test_conflicting_answers_for_same_image_question_are_reported(
    tmp_path: Path,
) -> None:
    config = _build_dataset(tmp_path)
    rows = [
        (
            "1",
            "img_0001.jpg",
            "What does this traffic sign indicate?",
            "Stop",
            "Mandatory",
        ),
        (
            "1",
            "img_0001.jpg",
            "What does this traffic sign indicate?",
            "No Entry",
            "Mandatory",
        ),
    ]
    _write_csv(config.compact_csv, rows[:1])
    _write_csv(config.full_csv, rows)

    result = VqaDatasetAuditor(config).audit()

    conflicts = result.report["annotation_quality"][
        "conflicting_image_question_answers"
    ]
    assert len(conflicts) == 1
    assert result.candidate_manifest == ()
    assert any(
        row["reason"] == "conflicting_identity_labels"
        for row in result.excluded_candidates
    )


def test_conflicting_labels_in_duplicate_group_are_excluded(tmp_path: Path) -> None:
    config = _build_dataset(tmp_path)
    rows = [
        (
            "1",
            "img_0001.jpg",
            "What does this traffic sign indicate?",
            "Stop",
            "Mandatory",
        ),
        (
            "2",
            "img_0002.jpg",
            "What does this traffic sign indicate?",
            "No Entry",
            "Mandatory",
        ),
    ]
    _write_csv(config.compact_csv, rows)
    _write_csv(config.full_csv, rows)

    result = VqaDatasetAuditor(config).audit()

    assert result.candidate_manifest == ()
    assert result.report["conflicting_duplicate_groups"]
    assert (
        sum(
            row["reason"] == "conflicting_labels_within_duplicate_group"
            for row in result.excluded_candidates
        )
        == 2
    )


def test_missing_required_column_is_rejected(tmp_path: Path) -> None:
    config = _build_dataset(tmp_path)
    config.full_csv.write_text(
        "id,image_name,question\n1,img_0001.jpg,What?\n", encoding="utf-8"
    )

    with pytest.raises(VqaAuditError, match="requires id"):
        VqaDatasetAuditor(config).audit()


def test_invalid_threshold_is_rejected(tmp_path: Path) -> None:
    config = _build_dataset(tmp_path)

    with pytest.raises(VqaAuditError, match="minimum_class_images"):
        VqaAuditConfig(
            dataset_root=config.dataset_root,
            images_directory=config.images_directory,
            compact_csv=config.compact_csv,
            full_csv=config.full_csv,
            minimum_class_images=0,
        )
