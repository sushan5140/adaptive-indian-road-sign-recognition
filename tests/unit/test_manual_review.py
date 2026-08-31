"""Tests for strict manual-review validation and decision application."""

from __future__ import annotations

import csv
from copy import deepcopy
from pathlib import Path

import pytest

from data.experiment_preparation import REVIEW_COLUMNS
from data.manual_review import (
    ManualReviewError,
    apply_manual_review,
    write_applied_review,
)


def _expected() -> list[dict[str, str]]:
    rows = []
    for index, label in enumerate(("give_way", "no_entry"), start=1):
        rows.append(
            {
                "review_id": f"B6-{index:04d}",
                "image_path": f"traffic512final/{index}.jpg",
                "source_image_id": f"{index}.jpg",
                "proposed_class": label,
                "dataset_a_class_id": "0" if label == "give_way" else "1",
                "dataset_a_class_name": (
                    "Give way" if label == "give_way" else "No entry"
                ),
                "source_question": "What does this sign indicate?",
                "source_answer": "Give way" if label == "give_way" else "No entry",
                "perceptual_group_id": f"group_{index}",
                "review_status": "pending",
                "review_notes": "",
                "review_label": "",
            }
        )
    return rows


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def test_approved_rejected_and_relabel_decisions(tmp_path: Path) -> None:
    expected = _expected()
    reviewed = deepcopy(expected)
    reviewed[0]["review_status"] = "relabel"
    reviewed[0]["review_label"] = "road_hump"
    reviewed[1]["review_status"] = "rejected"
    path = tmp_path / "review.csv"
    _write(path, reviewed)

    result = apply_manual_review(path, expected)

    assert len(result.external_rows) == 1
    assert result.external_rows[0]["class_name"] == "road_hump"
    assert result.external_rows[0]["dataset_a_class_id"] == "30"
    assert result.summary["per_proposed_class"]["no_entry"]["rejected"] == 1


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda rows: rows[0].update(review_status="unknown"), "unknown status"),
        (lambda rows: rows[0].update(review_status="relabel"), "requires review_label"),
        (
            lambda rows: rows[0].update(
                review_status="relabel", review_label="not_a_class"
            ),
            "invalid target label",
        ),
        (lambda rows: rows[0].update(image_path="changed.jpg"), "protected column"),
    ),
)
def test_malformed_decisions_are_rejected(
    tmp_path: Path, mutation, message: str
) -> None:
    expected = _expected()
    reviewed = deepcopy(expected)
    reviewed[0]["review_status"] = "approved"
    reviewed[1]["review_status"] = "approved"
    mutation(reviewed)
    path = tmp_path / "review.csv"
    _write(path, reviewed)

    with pytest.raises(ManualReviewError, match=message):
        apply_manual_review(path, expected)


def test_pending_missing_and_duplicate_rows_are_rejected(tmp_path: Path) -> None:
    expected = _expected()
    path = tmp_path / "review.csv"
    _write(path, expected)
    with pytest.raises(ManualReviewError, match="incomplete"):
        apply_manual_review(path, expected)

    missing = deepcopy(expected[:1])
    missing[0]["review_status"] = "approved"
    _write(path, missing)
    with pytest.raises(ManualReviewError, match="row set mismatch"):
        apply_manual_review(path, expected)

    duplicate = deepcopy(expected)
    duplicate[1]["review_id"] = duplicate[0]["review_id"]
    _write(path, duplicate)
    with pytest.raises(ManualReviewError, match="Duplicate review ID"):
        apply_manual_review(path, expected)

    duplicate_source = deepcopy(expected)
    duplicate_source[1]["source_image_id"] = duplicate_source[0]["source_image_id"]
    _write(path, duplicate_source)
    with pytest.raises(ManualReviewError, match="Duplicate source image ID"):
        apply_manual_review(path, expected)


def test_applied_review_writes_once_and_refuses_overwrite(tmp_path: Path) -> None:
    expected = _expected()
    reviewed = deepcopy(expected)
    for row in reviewed:
        row["review_status"] = "approved"
    review_path = tmp_path / "review.csv"
    _write(review_path, reviewed)
    result = apply_manual_review(review_path, expected)
    manifest = tmp_path / "external.csv"
    summary = tmp_path / "summary.json"

    written_manifest, written_summary = write_applied_review(
        result, manifest_path=manifest, summary_path=summary
    )

    assert written_manifest.read_text(encoding="utf-8").count("\n") == 3
    assert '"external_test_is_for_model_selection": false' in (
        written_summary.read_text(encoding="utf-8")
    )
    with pytest.raises(ManualReviewError, match="Refusing to overwrite"):
        write_applied_review(result, manifest_path=manifest, summary_path=summary)
