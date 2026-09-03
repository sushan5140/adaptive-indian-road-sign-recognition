"""Tests for unseen-class intake auditing and review protection."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from PIL import Image

from data.unseen_audit import (
    REVIEW_COLUMNS,
    UnseenAuditConfig,
    UnseenAuditError,
    UnseenDatasetAuditor,
)


def _setup(tmp_path: Path, classes: tuple[str, ...]) -> tuple[Path, Path]:
    raw = tmp_path / "raw"
    for class_name in classes:
        (raw / class_name).mkdir(parents=True)
    metadata = tmp_path / "source_metadata.csv"
    metadata.write_text(
        "relative_path,source_identifier,source_url,license,attribution\n",
        encoding="utf-8",
    )
    return raw, metadata


def _config(
    raw: Path,
    metadata: Path,
    classes: tuple[str, ...],
    existing: Path | None = None,
) -> UnseenAuditConfig:
    return UnseenAuditConfig(
        raw_root=raw,
        class_names=classes,
        source_metadata_csv=metadata,
        existing_review_csv=existing,
    )


def test_empty_intake_has_zero_rows_and_no_partition(tmp_path: Path) -> None:
    classes = ("stop", "no_left_turn")
    raw, metadata = _setup(tmp_path, classes)

    result = UnseenDatasetAuditor(_config(raw, metadata, classes)).audit()

    assert result.review_rows == ()
    assert result.summary["raw_photo_count"] == 0
    assert result.summary["automatic_approval_performed"] is False
    assert result.summary["partition_created"] is False
    assert all(
        not value["minimum_groups_met"]
        for value in result.summary["per_class"].values()
    )


def test_exact_cross_label_duplicates_share_independence_group(
    tmp_path: Path,
) -> None:
    classes = ("stop", "no_left_turn")
    raw, metadata = _setup(tmp_path, classes)
    image = Image.new("RGB", (40, 30), color=(200, 30, 10))
    image.save(raw / "stop" / "one.png")
    image.save(raw / "no_left_turn" / "two.png")

    result = UnseenDatasetAuditor(_config(raw, metadata, classes)).audit()

    assert len(result.review_rows) == 2
    assert {row["review_status"] for row in result.review_rows} == {"pending"}
    assert {row["exact_duplicate"] for row in result.review_rows} == {"true"}
    assert {row["cross_label_group_conflict"] for row in result.review_rows} == {"true"}
    assert len({row["independence_group_id"] for row in result.review_rows}) == 1
    assert result.summary["exact_duplicate_group_count"] == 1


def test_source_identifier_and_corrupt_file_are_flagged(tmp_path: Path) -> None:
    classes = ("stop",)
    raw, metadata = _setup(tmp_path, classes)
    Image.new("RGB", (20, 20), color="white").save(raw / "stop" / "one.png")
    (raw / "stop" / "broken.jpg").write_bytes(b"not-an-image")
    metadata.write_text(
        "relative_path,source_identifier,source_url,license,attribution\n"
        "stop/one.png,source-1,https://example.test/1,CC-BY-4.0,Example\n"
        "stop/broken.jpg,source-1,https://example.test/1,CC-BY-4.0,Example\n",
        encoding="utf-8",
    )

    result = UnseenDatasetAuditor(_config(raw, metadata, classes)).audit()

    assert len({row["independence_group_id"] for row in result.review_rows}) == 1
    broken = next(
        row for row in result.review_rows if row["source_filename"] == "broken.jpg"
    )
    assert broken["technical_status"] == "corrupt_or_unsupported"
    assert "corrupt_or_unsupported_image" in broken["audit_flags"]
    assert broken["review_status"] == "pending"


def test_existing_decision_is_preserved_after_protected_validation(
    tmp_path: Path,
) -> None:
    classes = ("stop",)
    raw, metadata = _setup(tmp_path, classes)
    Image.new("RGB", (20, 20), color="white").save(raw / "stop" / "one.png")
    initial = UnseenDatasetAuditor(_config(raw, metadata, classes)).audit()
    review_path = tmp_path / "review.csv"
    row = dict(initial.review_rows[0])
    row["review_status"] = "approved"
    row["review_notes"] = "human checked"
    with review_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerow(row)

    repeated = UnseenDatasetAuditor(
        _config(raw, metadata, classes, review_path)
    ).audit()

    assert repeated.review_rows[0]["review_status"] == "approved"
    assert repeated.review_rows[0]["review_notes"] == "human checked"

    row["sha256"] = "tampered"
    with review_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerow(row)
    with pytest.raises(UnseenAuditError, match="Protected field changed"):
        UnseenDatasetAuditor(_config(raw, metadata, classes, review_path)).audit()


def test_metadata_cannot_reference_missing_image(tmp_path: Path) -> None:
    classes = ("stop",)
    raw, metadata = _setup(tmp_path, classes)
    metadata.write_text(
        "relative_path,source_identifier,source_url,license,attribution\n"
        "stop/missing.jpg,source-1,https://example.test/1,CC-BY-4.0,Example\n",
        encoding="utf-8",
    )

    with pytest.raises(UnseenAuditError, match="references missing images"):
        UnseenDatasetAuditor(_config(raw, metadata, classes)).audit()
