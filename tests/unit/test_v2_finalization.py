"""Tests for final V2 review validation and group-safe splitting."""

from __future__ import annotations

from collections import Counter

import pytest

from data.v2_finalization import (
    V2FinalizationError,
    analyze_class_viability,
    apply_final_review_decisions,
    group_safe_split,
)


def test_final_review_transfers_only_decisions_and_preserves_completed_rows() -> None:
    source = [_review_row("V2-0001", "approved"), _review_row("V2-0002", "pending")]
    decision = dict(source[1], review_status="rejected", review_notes="not the sign")

    result = apply_final_review_decisions(source, [decision])

    assert result.rows[0] == source[0]
    assert result.rows[1]["review_status"] == "rejected"
    assert result.rows[1]["image_path"] == source[1]["image_path"]


def test_final_review_rejects_protected_field_change() -> None:
    source = [_review_row("V2-0001", "pending")]
    decision = dict(
        source[0], review_status="approved", source_image_id="different.jpg"
    )

    with pytest.raises(V2FinalizationError, match="protected columns"):
        apply_final_review_decisions(source, [decision])


def test_viability_uses_approved_photos_and_independent_groups() -> None:
    rows = [
        _review_row(
            f"V2-{index:04d}",
            "approved",
            class_name="enough",
            group=f"g-{index}",
        )
        for index in range(10)
    ]
    rows.extend(
        _review_row(
            f"V2-{index + 100:04d}",
            "approved",
            class_name="dependent",
            group=f"shared-{index % 2}",
        )
        for index in range(10)
    )

    result = {row["proposed_class"]: row for row in analyze_class_viability(rows)}

    assert result["enough"]["viable"] is True
    assert result["dependent"]["viable"] is False
    assert "approved perceptual groups 2 < 8" in result["dependent"]["exclusion_reason"]


def test_group_safe_split_is_deterministic_balanced_and_has_no_leakage() -> None:
    rows = []
    for class_name in ("alpha", "beta"):
        for group_index in range(10):
            group_size = 2 if group_index < 2 else 1
            for item_index in range(group_size):
                review_id = f"{class_name}-{group_index}-{item_index}"
                rows.append(
                    _review_row(
                        review_id,
                        "approved",
                        class_name=class_name,
                        group=f"{class_name}-group-{group_index}",
                    )
                )

    first = group_safe_split(rows, ("alpha", "beta"), random_seed=42)
    second = group_safe_split(rows, ("alpha", "beta"), random_seed=42)

    assert first == second
    owners: dict[str, set[str]] = {}
    for split_name, split_rows in first.splits.items():
        assert {row["class_name"] for row in split_rows} == {"alpha", "beta"}
        for row in split_rows:
            owners.setdefault(row["perceptual_group_id"], set()).add(split_name)
    assert all(len(split_names) == 1 for split_names in owners.values())
    assert first.summary["leakage_checks"]["all_checks_passed"] is True
    assert (
        Counter(
            row["source_image_id"] for rows_ in first.splits.values() for row in rows_
        ).most_common(1)[0][1]
        == 1
    )


def _review_row(
    review_id: str,
    status: str,
    *,
    class_name: str = "stop",
    group: str = "group-1",
) -> dict[str, str]:
    source_id = f"{review_id}.jpg"
    return {
        "review_id": review_id,
        "image_path": f"traffic512final/{source_id}",
        "source_image_id": source_id,
        "proposed_class": class_name,
        "source_question": "What sign?",
        "source_answer": class_name,
        "perceptual_group_id": group,
        "review_status": status,
        "review_label": "",
        "review_notes": "",
    }
