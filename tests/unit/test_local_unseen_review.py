"""Tests for local unseen-class manual-review bundle validation."""

from __future__ import annotations

from copy import deepcopy

import pytest

from data.local_unseen_review import (
    LocalUnseenReviewError,
    is_experiment_ready,
    validate_human_review_overlay,
    validate_local_unseen_review_rows,
)


def _valid_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    specifications = (
        ("bus_stop", 36, 31),
        ("no_left_turn", 2, 2),
        ("no_parking", 5, 4),
    )
    index = 0
    for label, image_count, group_count in specifications:
        for class_index in range(image_count):
            index += 1
            rows.append(
                {
                    "review_id": f"ULR-{index:04d}",
                    "source_image_id": f"img_{index:04d}.jpg",
                    "proposed_unseen_class": label,
                    "source_or_independence_group": (
                        f"{label}_group_{class_index % group_count:02d}"
                    ),
                    "review_status": "pending",
                    "review_label": "",
                    "licence_status": "pending_curator_confirmation",
                    "sha256": f"sha-{index}",
                    "dhash": f"dhash-{index}",
                    "v2_train_overlap": "no",
                    "v2_validation_overlap": "no",
                    "v2_test_overlap": "no",
                    "v2_exact_sha_overlap": "no",
                    "v2_perceptual_group_overlap": "no",
                }
            )
    return rows


def test_valid_locked_bundle_counts_images_and_groups() -> None:
    assert validate_local_unseen_review_rows(_valid_rows()) == {
        "bus_stop": {"images": 36, "groups": 31},
        "no_left_turn": {"images": 2, "groups": 2},
        "no_parking": {"images": 5, "groups": 4},
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("review_status", "approved", "must remain pending"),
        ("licence_status", "cleared", "licence status must remain pending"),
        ("v2_train_overlap", "yes", "Frozen V2 overlap"),
    ),
)
def test_invalid_review_state_fails_closed(
    field: str, value: str, message: str
) -> None:
    rows = _valid_rows()
    rows[0][field] = value
    with pytest.raises(LocalUnseenReviewError, match=message):
        validate_local_unseen_review_rows(rows)


def test_duplicate_source_id_fails_closed() -> None:
    rows = deepcopy(_valid_rows())
    rows[1]["source_image_id"] = rows[0]["source_image_id"]
    with pytest.raises(LocalUnseenReviewError, match="Source image IDs"):
        validate_local_unseen_review_rows(rows)


def _reviewed_rows() -> list[dict[str, str]]:
    rows = deepcopy(_valid_rows())
    rejected = {"ULR-0032", "ULR-0034", "ULR-0038", "ULR-0042"}
    for row in rows:
        if row["review_id"] in rejected:
            row["review_status"] = "rejected"
            row["review_label"] = ""
            row["review_notes"] = "Human reviewer rejected this candidate."
        else:
            row["review_status"] = "accepted"
            row["review_label"] = row["proposed_unseen_class"]
            row["review_notes"] = "Human reviewer accepted this candidate."
    return rows


def test_human_overlay_accepts_only_locked_decisions() -> None:
    assert validate_human_review_overlay(_valid_rows(), _reviewed_rows()) == {
        "bus_stop": {"accepted": 34, "rejected": 2},
        "no_left_turn": {"accepted": 1, "rejected": 1},
        "no_parking": {"accepted": 4, "rejected": 1},
    }


def test_human_overlay_rejects_protected_field_change() -> None:
    reviewed = _reviewed_rows()
    reviewed[0]["sha256"] = "modified"
    with pytest.raises(LocalUnseenReviewError, match="Protected field changed"):
        validate_human_review_overlay(_valid_rows(), reviewed)


def test_accepted_but_pending_licence_is_not_experiment_ready() -> None:
    row = _reviewed_rows()[0]
    assert row["review_status"] == "accepted"
    assert row["licence_status"] == "pending_curator_confirmation"
    assert is_experiment_ready(row) is False
    row["licence_status"] = "approved"
    assert is_experiment_ready(row) is True
