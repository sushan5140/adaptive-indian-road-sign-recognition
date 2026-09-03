"""Validation helpers for the local unseen-class manual-review bundle."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence


class LocalUnseenReviewError(ValueError):
    """Raised when the local unseen review bundle violates its locked protocol."""


EXPECTED_CLASS_COUNTS: dict[str, int] = {
    "bus_stop": 36,
    "no_left_turn": 2,
    "no_parking": 5,
}

EXPECTED_GROUP_COUNTS: dict[str, int] = {
    "bus_stop": 31,
    "no_left_turn": 2,
    "no_parking": 4,
}

DECISION_FIELDS = frozenset({"review_status", "review_label", "review_notes"})
EXPECTED_REJECTED_IDS = frozenset({"ULR-0032", "ULR-0034", "ULR-0038", "ULR-0042"})
APPROVED_LICENCE_STATUSES = frozenset({"approved", "confirmed"})

REVIEW_COLUMNS = (
    "review_id",
    "image_path",
    "source_image_id",
    "source_question",
    "source_answer",
    "source_dataset",
    "original_v2_review_id",
    "original_v2_proposed_class",
    "original_v2_review_status",
    "original_v2_review_label",
    "original_v2_review_notes",
    "original_v2_review_outcome",
    "proposed_unseen_class",
    "sha256",
    "dhash",
    "perceptual_group_id",
    "source_or_independence_group",
    "v2_train_overlap",
    "v2_validation_overlap",
    "v2_test_overlap",
    "v2_exact_sha_overlap",
    "v2_perceptual_group_overlap",
    "overlap_validation_notes",
    "eligibility_note",
    "licence_status",
    "review_status",
    "review_label",
    "review_notes",
)


def validate_local_unseen_review_rows(
    rows: Sequence[Mapping[str, str]],
) -> dict[str, dict[str, int]]:
    """Validate pending rows and return measured per-class image/group counts."""
    if len(rows) != 43:
        raise LocalUnseenReviewError(f"Expected 43 review rows, found {len(rows)}")

    review_ids = [row.get("review_id", "") for row in rows]
    source_ids = [row.get("source_image_id", "") for row in rows]
    if not all(review_ids) or len(set(review_ids)) != len(review_ids):
        raise LocalUnseenReviewError("Review IDs must be non-empty and unique")
    if not all(source_ids) or len(set(source_ids)) != len(source_ids):
        raise LocalUnseenReviewError("Source image IDs must be non-empty and unique")

    counts = Counter(row.get("proposed_unseen_class", "") for row in rows)
    if dict(counts) != EXPECTED_CLASS_COUNTS:
        raise LocalUnseenReviewError(
            f"Unexpected class counts: {dict(counts)}; expected {EXPECTED_CLASS_COUNTS}"
        )

    groups: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        label = row.get("proposed_unseen_class", "")
        group = row.get("source_or_independence_group", "")
        if not group:
            raise LocalUnseenReviewError(
                f"Missing independence group for {row.get('review_id', '<unknown>')}"
            )
        groups[label].add(group)
        if row.get("review_status") != "pending":
            raise LocalUnseenReviewError("Every local unseen row must remain pending")
        if row.get("review_label", ""):
            raise LocalUnseenReviewError("Pending rows must not have a review label")
        if row.get("licence_status") != "pending_curator_confirmation":
            raise LocalUnseenReviewError("Dataset B licence status must remain pending")
        if not row.get("sha256") or not row.get("dhash"):
            raise LocalUnseenReviewError("Every row must preserve SHA-256 and dHash")
        for field in (
            "v2_train_overlap",
            "v2_validation_overlap",
            "v2_test_overlap",
            "v2_exact_sha_overlap",
            "v2_perceptual_group_overlap",
        ):
            if row.get(field) != "no":
                raise LocalUnseenReviewError(
                    f"Frozen V2 overlap detected in {field}: {row['review_id']}"
                )

    measured = {
        label: {"images": counts[label], "groups": len(groups[label])}
        for label in EXPECTED_CLASS_COUNTS
    }
    expected = {
        label: {
            "images": EXPECTED_CLASS_COUNTS[label],
            "groups": EXPECTED_GROUP_COUNTS[label],
        }
        for label in EXPECTED_CLASS_COUNTS
    }
    if measured != expected:
        raise LocalUnseenReviewError(
            f"Unexpected image/group counts: {measured}; expected {expected}"
        )
    return measured


def validate_human_review_overlay(
    base_rows: Sequence[Mapping[str, str]],
    reviewed_rows: Sequence[Mapping[str, str]],
) -> dict[str, dict[str, int]]:
    """Validate that a completed workbook changes only human decision fields."""
    validate_local_unseen_review_rows(base_rows)
    if len(reviewed_rows) != len(base_rows):
        raise LocalUnseenReviewError("Reviewed row count differs from the base queue")
    base_by_id = {row["review_id"]: row for row in base_rows}
    reviewed_ids = [row.get("review_id", "") for row in reviewed_rows]
    base_ids = [row["review_id"] for row in base_rows]
    if reviewed_ids != base_ids or len(set(reviewed_ids)) != len(reviewed_ids):
        raise LocalUnseenReviewError(
            "Reviewed IDs must exactly match the ordered base review IDs"
        )

    class_status: dict[str, Counter[str]] = defaultdict(Counter)
    rejected_ids: set[str] = set()
    for reviewed in reviewed_rows:
        review_id = reviewed["review_id"]
        base = base_by_id[review_id]
        protected_fields = set(base) - DECISION_FIELDS
        for field in protected_fields:
            if _blank_normalized(base.get(field)) != _blank_normalized(
                reviewed.get(field)
            ):
                raise LocalUnseenReviewError(
                    f"Protected field changed for {review_id}: {field}"
                )
        status = _blank_normalized(reviewed.get("review_status"))
        label = _blank_normalized(reviewed.get("review_label"))
        notes = _blank_normalized(reviewed.get("review_notes"))
        proposed = reviewed["proposed_unseen_class"]
        if status not in {"accepted", "rejected"}:
            raise LocalUnseenReviewError(
                f"Invalid completed review status for {review_id}: {status}"
            )
        if status == "accepted" and label != proposed:
            raise LocalUnseenReviewError(
                f"Accepted label must equal proposed class for {review_id}"
            )
        if status == "rejected" and label:
            raise LocalUnseenReviewError(
                f"Rejected row must not have a review label: {review_id}"
            )
        if not notes:
            raise LocalUnseenReviewError(f"Review notes are required: {review_id}")
        if reviewed.get("licence_status") != "pending_curator_confirmation":
            raise LocalUnseenReviewError(f"Licence status changed for {review_id}")
        class_status[proposed][status] += 1
        if status == "rejected":
            rejected_ids.add(review_id)

    expected = {
        "bus_stop": {"accepted": 34, "rejected": 2},
        "no_left_turn": {"accepted": 1, "rejected": 1},
        "no_parking": {"accepted": 4, "rejected": 1},
    }
    measured = {
        label: {
            "accepted": class_status[label]["accepted"],
            "rejected": class_status[label]["rejected"],
        }
        for label in expected
    }
    if measured != expected:
        raise LocalUnseenReviewError(
            f"Unexpected human-review counts: {measured}; expected {expected}"
        )
    if rejected_ids != EXPECTED_REJECTED_IDS:
        raise LocalUnseenReviewError(f"Unexpected rejected IDs: {sorted(rejected_ids)}")
    return measured


def is_experiment_ready(row: Mapping[str, str]) -> bool:
    """Return whether both visual review and licence approval permit experiment use."""
    return (
        row.get("review_status") == "accepted"
        and row.get("licence_status") in APPROVED_LICENCE_STATUSES
    )


def _blank_normalized(value: object) -> str:
    return "" if value is None else str(value)
