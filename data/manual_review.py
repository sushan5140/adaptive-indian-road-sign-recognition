"""Strict validation and application of six-class human review decisions."""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data.experiment_preparation import REVIEW_COLUMNS, SPECS_BY_B_CLASS

ALLOWED_STATUSES = {"pending", "approved", "rejected", "relabel"}
PROTECTED_COLUMNS = tuple(
    column
    for column in REVIEW_COLUMNS
    if column not in {"review_status", "review_notes", "review_label"}
)
EXTERNAL_COLUMNS = (
    "image_path",
    "class_name",
    "dataset_a_class_id",
    "source_image_id",
    "perceptual_group_id",
    "review_id",
)


class ManualReviewError(ValueError):
    """Raised when a human-edited review CSV is incomplete or malformed."""


@dataclass(frozen=True, slots=True)
class AppliedReview:
    """Validated external-test rows and per-proposed-class decision summary."""

    external_rows: tuple[dict[str, str], ...]
    summary: dict[str, Any]


def apply_manual_review(
    reviewed_csv: str | Path, expected_rows: list[dict[str, str]]
) -> AppliedReview:
    """Validate all decisions and build deterministic approved external rows."""
    reviewed_rows, columns = _read_review_csv(Path(reviewed_csv))
    required = set(REVIEW_COLUMNS) - {"review_label"}
    missing_columns = sorted(required.difference(columns))
    if missing_columns:
        raise ManualReviewError(f"Review CSV is missing columns: {missing_columns}")
    expected_by_id = {row["review_id"]: row for row in expected_rows}
    if len(expected_by_id) != len(expected_rows):
        raise ManualReviewError("Expected review rows contain duplicate review IDs")
    reviewed_by_id: dict[str, dict[str, str]] = {}
    seen_sources: set[str] = set()
    for row_number, row in enumerate(reviewed_rows, start=2):
        review_id = row.get("review_id", "").strip()
        source_id = row.get("source_image_id", "").strip()
        if not review_id:
            raise ManualReviewError(f"Review row {row_number}: review_id is blank")
        if review_id in reviewed_by_id:
            raise ManualReviewError(f"Duplicate review ID: {review_id}")
        if not source_id:
            raise ManualReviewError(
                f"Review row {row_number}: source_image_id is blank"
            )
        if source_id in seen_sources:
            raise ManualReviewError(f"Duplicate source image ID: {source_id}")
        seen_sources.add(source_id)
        reviewed_by_id[review_id] = row
    missing_rows = sorted(set(expected_by_id).difference(reviewed_by_id))
    extra_rows = sorted(set(reviewed_by_id).difference(expected_by_id))
    if missing_rows or extra_rows:
        raise ManualReviewError(
            f"Review row set mismatch; missing={missing_rows}, extra={extra_rows}"
        )
    decisions: list[tuple[dict[str, str], str, str | None]] = []
    pending_ids: list[str] = []
    for review_id in sorted(expected_by_id):
        expected = expected_by_id[review_id]
        reviewed = reviewed_by_id[review_id]
        for column in PROTECTED_COLUMNS:
            if reviewed.get(column, "") != expected[column]:
                raise ManualReviewError(
                    f"Review {review_id}: protected column {column!r} does not match"
                )
        status = reviewed.get("review_status", "").strip().casefold()
        if status not in ALLOWED_STATUSES:
            raise ManualReviewError(f"Review {review_id}: unknown status {status!r}")
        label: str | None = None
        if status == "pending":
            pending_ids.append(review_id)
        elif status == "approved":
            label = expected["proposed_class"]
        elif status == "relabel":
            label = reviewed.get("review_label", "").strip()
            if not label:
                raise ManualReviewError(
                    f"Review {review_id}: relabel requires review_label"
                )
            if label not in SPECS_BY_B_CLASS:
                raise ManualReviewError(
                    f"Review {review_id}: invalid target label {label!r}"
                )
        decisions.append((expected, status, label))
    if pending_ids:
        raise ManualReviewError(
            f"Manual review is incomplete; pending review IDs: {pending_ids}"
        )
    external_rows: list[dict[str, str]] = []
    counts: dict[str, Counter[str]] = {
        name: Counter() for name in sorted(SPECS_BY_B_CLASS)
    }
    for expected, status, label in decisions:
        proposed = expected["proposed_class"]
        counts[proposed]["reviewed"] += 1
        counts[proposed][status] += 1
        if label is None:
            continue
        target = SPECS_BY_B_CLASS[label]
        external_rows.append(
            {
                "image_path": expected["image_path"],
                "class_name": label,
                "dataset_a_class_id": target.dataset_a_class_id,
                "source_image_id": expected["source_image_id"],
                "perceptual_group_id": expected["perceptual_group_id"],
                "review_id": expected["review_id"],
            }
        )
        counts[proposed]["external_test_included"] += 1
    external_rows.sort(key=lambda row: (row["class_name"], row["source_image_id"]))
    if len({row["review_id"] for row in external_rows}) != len(external_rows):
        raise ManualReviewError("External rows contain duplicate review IDs")
    if len({row["source_image_id"] for row in external_rows}) != len(external_rows):
        raise ManualReviewError("External rows contain duplicate source image IDs")
    summary = {
        "review_complete": True,
        "external_test_sample_count": len(external_rows),
        "external_test_is_for_model_selection": False,
        "per_proposed_class": {
            class_name: {
                key: values[key]
                for key in (
                    "reviewed",
                    "approved",
                    "rejected",
                    "relabel",
                    "external_test_included",
                )
            }
            for class_name, values in counts.items()
        },
    }
    return AppliedReview(external_rows=tuple(external_rows), summary=summary)


def write_applied_review(
    result: AppliedReview,
    *,
    manifest_path: str | Path,
    summary_path: str | Path,
) -> tuple[Path, Path]:
    """Write final external-test artifacts and refuse to overwrite them."""
    manifest = Path(manifest_path).expanduser().resolve()
    summary = Path(summary_path).expanduser().resolve()
    if manifest.exists() or summary.exists():
        raise ManualReviewError("Refusing to overwrite external-test review artifacts")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXTERNAL_COLUMNS)
        writer.writeheader()
        writer.writerows(result.external_rows)
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(
        json.dumps(result.summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest, summary


def _read_review_csv(path: Path) -> tuple[list[dict[str, str]], set[str]]:
    try:
        with (
            path.expanduser().resolve().open(encoding="utf-8-sig", newline="") as handle
        ):
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ManualReviewError(f"Review CSV has no header: {path}")
            return [dict(row) for row in reader], set(reader.fieldnames)
    except (OSError, csv.Error) as error:
        raise ManualReviewError(f"Could not read review CSV: {path}") from error
