"""Prepare a deterministic human-review queue for Baseline V2."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

V2_REVIEW_COLUMNS = (
    "review_id",
    "image_path",
    "source_image_id",
    "proposed_class",
    "source_question",
    "source_answer",
    "perceptual_group_id",
    "review_status",
    "review_label",
    "review_notes",
)

_CANDIDATE_COLUMNS = (
    "image_path",
    "class_name",
    "source_question",
    "source_answer",
    "source_image_id",
    "leakage_group_id",
)
_REUSABLE_STATUSES = {"approved", "rejected"}


class V2ReviewError(ValueError):
    """Raised when a safe and deterministic V2 review queue cannot be built."""


@dataclass(frozen=True, slots=True)
class V2ReviewPreparation:
    """Review rows and measured preparation summary for Baseline V2."""

    rows: tuple[dict[str, str], ...]
    summary: dict[str, Any]


def prepare_v2_review(
    candidate_manifest: str | Path,
    previous_review_manifest: str | Path,
    excluded_candidates: str | Path,
) -> V2ReviewPreparation:
    """Build all-candidate review rows and reuse only exact approved/rejected decisions.

    Previous decisions are keyed by both source image and proposed class. A decision
    for the same image under a different class is deliberately not reused.
    """
    candidates = _read_csv(Path(candidate_manifest), "candidate manifest")
    previous = _read_csv(Path(previous_review_manifest), "previous review manifest")
    excluded = _read_csv(Path(excluded_candidates), "excluded-candidate manifest")
    _require_columns(candidates, _CANDIDATE_COLUMNS, "candidate manifest")

    previous_by_key: dict[tuple[str, str], dict[str, str]] = {}
    for row_number, row in enumerate(previous, start=2):
        source_id = _required(row, "source_image_id", row_number)
        proposed_class = _required(row, "proposed_class", row_number)
        key = (source_id, proposed_class)
        if key in previous_by_key:
            raise V2ReviewError(f"Duplicate previous review identity: {key}")
        previous_by_key[key] = row

    normalized: list[dict[str, str]] = []
    seen_sources: set[str] = set()
    for row_number, row in enumerate(candidates, start=2):
        source_id = _required(row, "source_image_id", row_number)
        if source_id in seen_sources:
            raise V2ReviewError(f"Duplicate candidate source image: {source_id}")
        seen_sources.add(source_id)
        proposed_class = _required(row, "class_name", row_number)
        previous_row = previous_by_key.get((source_id, proposed_class))
        status = "pending"
        label = ""
        notes = ""
        if previous_row is not None:
            previous_status = previous_row.get("review_status", "").strip().casefold()
            if previous_status in _REUSABLE_STATUSES:
                status = previous_status
                label = previous_row.get("review_label", "").strip()
                notes = previous_row.get("review_notes", "").strip()
        normalized.append(
            {
                "image_path": _required(row, "image_path", row_number),
                "source_image_id": source_id,
                "proposed_class": proposed_class,
                "source_question": _required(row, "source_question", row_number),
                "source_answer": _required(row, "source_answer", row_number),
                "perceptual_group_id": _required(row, "leakage_group_id", row_number),
                "review_status": status,
                "review_label": label,
                "review_notes": notes,
            }
        )

    normalized.sort(key=lambda row: (row["proposed_class"], row["source_image_id"]))
    rows = tuple(
        {"review_id": f"V2-{index:04d}", **row}
        for index, row in enumerate(normalized, start=1)
    )
    summary = _build_summary(rows, excluded)
    return V2ReviewPreparation(rows=rows, summary=summary)


def pending_review_rows(rows: tuple[dict[str, str], ...]) -> list[dict[str, str]]:
    """Return pending rows in their deterministic manifest order."""
    return [row for row in rows if row["review_status"] == "pending"]


def write_v2_review(
    preparation: V2ReviewPreparation,
    *,
    review_path: str | Path,
    summary_path: str | Path,
) -> tuple[Path, Path]:
    """Write V2 review artifacts while protecting any existing human edits."""
    review = Path(review_path).expanduser().resolve()
    summary = Path(summary_path).expanduser().resolve()
    if review.exists() or summary.exists():
        raise V2ReviewError("Refusing to overwrite existing V2 review artifacts")
    review.parent.mkdir(parents=True, exist_ok=True)
    with review.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=V2_REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerows(preparation.rows)
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(
        json.dumps(preparation.summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return review, summary


def _build_summary(
    rows: tuple[dict[str, str], ...], excluded: list[dict[str, str]]
) -> dict[str, Any]:
    per_class: dict[str, Counter[str]] = defaultdict(Counter)
    groups_by_class: dict[str, set[str]] = defaultdict(set)
    groups_by_class_and_status: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        class_name = row["proposed_class"]
        status = row["review_status"]
        per_class[class_name]["candidates"] += 1
        per_class[class_name][status] += 1
        groups_by_class[class_name].add(row["perceptual_group_id"])
        groups_by_class_and_status[(class_name, status)].add(row["perceptual_group_id"])
    excluded_reasons = Counter(row.get("reason", "").strip() for row in excluded)
    statuses = Counter(row["review_status"] for row in rows)
    return {
        "review_complete": statuses["pending"] == 0,
        "candidate_photos": len(rows),
        "candidate_classes": len(per_class),
        "independent_perceptual_groups": len(
            {row["perceptual_group_id"] for row in rows}
        ),
        "previously_reviewed_reused": statuses["approved"] + statuses["rejected"],
        "approved_reused": statuses["approved"],
        "rejected_reused": statuses["rejected"],
        "pending_manual_review": statuses["pending"],
        "excluded_before_review": len(excluded),
        "excluded_reasons": dict(sorted(excluded_reasons.items())),
        "minimum_retention_rule": {
            "approved_unique_photos": 10,
            "independent_perceptual_groups": 8,
            "preferred_photos": 15,
        },
        "per_class": {
            class_name: {
                "candidates": counts["candidates"],
                "perceptual_groups": len(groups_by_class[class_name]),
                "approved_reused": counts["approved"],
                "approved_groups_reused": len(
                    groups_by_class_and_status[(class_name, "approved")]
                ),
                "rejected_reused": counts["rejected"],
                "pending": counts["pending"],
                "pending_groups": len(
                    groups_by_class_and_status[(class_name, "pending")]
                ),
            }
            for class_name, counts in sorted(per_class.items())
        },
    }


def _read_csv(path: Path, description: str) -> list[dict[str, str]]:
    try:
        with (
            path.expanduser().resolve().open(encoding="utf-8-sig", newline="") as handle
        ):
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise V2ReviewError(f"{description} has no header: {path}")
            return [dict(row) for row in reader]
    except (OSError, csv.Error) as error:
        raise V2ReviewError(f"Could not read {description}: {path}") from error


def _require_columns(
    rows: list[dict[str, str]], required: tuple[str, ...], description: str
) -> None:
    if not rows:
        raise V2ReviewError(f"{description} is empty")
    missing = sorted(set(required).difference(rows[0]))
    if missing:
        raise V2ReviewError(f"{description} is missing columns: {missing}")


def _required(row: dict[str, str], column: str, row_number: int) -> str:
    value = row.get(column, "").strip()
    if not value:
        raise V2ReviewError(f"Row {row_number}: required column {column!r} is blank")
    return value
