"""Tests for safe Baseline V2 review preparation."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from data.v2_review import V2ReviewError, prepare_v2_review, write_v2_review


def test_exact_previous_decisions_are_reused_and_new_rows_remain_pending(
    tmp_path: Path,
) -> None:
    candidates, previous, excluded = _manifests(tmp_path)

    result = prepare_v2_review(candidates, previous, excluded)

    assert [row["review_id"] for row in result.rows] == ["V2-0001", "V2-0002"]
    assert result.rows[0]["review_status"] == "approved"
    assert result.rows[0]["review_notes"] == "checked"
    assert result.rows[1]["review_status"] == "pending"
    assert result.summary["previously_reviewed_reused"] == 1
    assert result.summary["pending_manual_review"] == 1
    assert result.summary["excluded_before_review"] == 1


def test_previous_decision_is_not_reused_for_a_different_class(tmp_path: Path) -> None:
    candidates, previous, excluded = _manifests(
        tmp_path, previous_class="stop", candidate_class="give_way"
    )

    result = prepare_v2_review(candidates, previous, excluded)

    assert result.rows[0]["review_status"] == "pending"
    assert result.summary["previously_reviewed_reused"] == 0


def test_review_rows_are_deterministic_and_report_perceptual_groups(
    tmp_path: Path,
) -> None:
    candidates, previous, excluded = _manifests(tmp_path)

    first = prepare_v2_review(candidates, previous, excluded)
    second = prepare_v2_review(candidates, previous, excluded)

    assert first == second
    assert first.summary["independent_perceptual_groups"] == 2
    assert first.summary["per_class"]["give_way"]["candidates"] == 1
    assert first.summary["per_class"]["filling_station"]["approved_groups_reused"] == 1
    assert first.summary["per_class"]["give_way"]["pending_groups"] == 1


def test_duplicate_candidate_source_is_rejected(tmp_path: Path) -> None:
    candidates, previous, excluded = _manifests(tmp_path)
    rows = _read(candidates)
    rows[1]["source_image_id"] = rows[0]["source_image_id"]
    _write(candidates, rows)

    with pytest.raises(V2ReviewError, match="Duplicate candidate source image"):
        prepare_v2_review(candidates, previous, excluded)


def test_write_refuses_to_overwrite_review_artifacts(tmp_path: Path) -> None:
    candidates, previous, excluded = _manifests(tmp_path)
    result = prepare_v2_review(candidates, previous, excluded)
    review = tmp_path / "review.csv"
    summary = tmp_path / "summary.json"
    write_v2_review(result, review_path=review, summary_path=summary)

    with pytest.raises(V2ReviewError, match="Refusing to overwrite"):
        write_v2_review(result, review_path=review, summary_path=summary)


def _manifests(
    tmp_path: Path,
    *,
    previous_class: str = "filling_station",
    candidate_class: str = "filling_station",
) -> tuple[Path, Path, Path]:
    candidates = tmp_path / "candidates.csv"
    previous = tmp_path / "previous.csv"
    excluded = tmp_path / "excluded.csv"
    _write(
        candidates,
        [
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
                "class_name": candidate_class,
                "source_question": "What sign?",
                "source_answer": "Fuel",
                "source_image_id": "a.jpg",
                "leakage_group_id": "group_1",
            },
        ],
    )
    _write(
        previous,
        [
            {
                "review_id": "B6-0001",
                "source_image_id": "a.jpg",
                "proposed_class": previous_class,
                "review_status": "approved",
                "review_label": "",
                "review_notes": "checked",
            }
        ],
    )
    _write(
        excluded,
        [
            {
                "image_id": "z.jpg",
                "reason": "unsupported_or_ambiguous_identity_answer",
            }
        ],
    )
    return candidates, previous, excluded


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
