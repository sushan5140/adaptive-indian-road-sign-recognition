"""Validated final-review application and group-safe Baseline V2 splitting."""

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from data.v2_review import V2_REVIEW_COLUMNS

EDITABLE_REVIEW_COLUMNS = ("review_status", "review_label", "review_notes")
PROTECTED_REVIEW_COLUMNS = tuple(
    column for column in V2_REVIEW_COLUMNS if column not in EDITABLE_REVIEW_COLUMNS
)
SPLIT_ORDER = ("train", "validation", "test")
SPLIT_MANIFEST_COLUMNS = (
    "image_path",
    "class_name",
    "source_image_id",
    "perceptual_group_id",
    "review_id",
    "split",
    "source_question",
    "source_answer",
)


class V2FinalizationError(ValueError):
    """Raised when V2 review finalization or group-safe splitting is unsafe."""


@dataclass(frozen=True, slots=True)
class FinalReviewResult:
    """Complete reviewed rows after safe decision transfer."""

    rows: tuple[dict[str, str], ...]
    status_counts: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class GroupSplitResult:
    """Deterministic V2 split rows and measured allocation summary."""

    splits: Mapping[str, tuple[dict[str, str], ...]]
    summary: Mapping[str, Any]


def apply_final_review_decisions(
    source_rows: Iterable[dict[str, str]],
    decision_rows: Iterable[dict[str, str]],
) -> FinalReviewResult:
    """Transfer only editable fields from decisions for the exact pending ID set."""
    source = [dict(row) for row in source_rows]
    decisions = [dict(row) for row in decision_rows]
    _validate_columns(source, "source review")
    _validate_columns(decisions, "decision review")
    source_by_id = _unique_by_id(source, "source review")
    decisions_by_id = _unique_by_id(decisions, "decision review")
    pending_ids = {
        review_id
        for review_id, row in source_by_id.items()
        if row["review_status"].strip().casefold() == "pending"
    }
    if set(decisions_by_id) != pending_ids:
        raise V2FinalizationError(
            "Decision review IDs must exactly equal the current pending review IDs"
        )
    for review_id, decision in decisions_by_id.items():
        original = source_by_id[review_id]
        differing = [
            column
            for column in PROTECTED_REVIEW_COLUMNS
            if decision[column] != original[column]
        ]
        if differing:
            raise V2FinalizationError(
                f"Review {review_id} changes protected columns: {differing}"
            )
        status = decision["review_status"].strip().casefold()
        if status not in {"approved", "rejected", "relabel"}:
            raise V2FinalizationError(
                f"Review {review_id} has invalid final status {status!r}"
            )
        if status == "relabel" and not decision["review_label"].strip():
            raise V2FinalizationError(
                f"Review {review_id} uses relabel without review_label"
            )

    completed_before = {
        review_id: dict(row)
        for review_id, row in source_by_id.items()
        if review_id not in pending_ids
    }
    updated: list[dict[str, str]] = []
    for row in source:
        merged = dict(row)
        pending_decision = decisions_by_id.get(row["review_id"])
        if pending_decision is not None:
            for column in EDITABLE_REVIEW_COLUMNS:
                merged[column] = pending_decision[column]
        updated.append(merged)
    updated_by_id = {row["review_id"]: row for row in updated}
    if any(updated_by_id[key] != value for key, value in completed_before.items()):
        raise V2FinalizationError("A previously completed decision was modified")
    counts = Counter(row["review_status"].strip().casefold() for row in updated)
    return FinalReviewResult(
        rows=tuple(updated),
        status_counts={
            status: counts[status]
            for status in ("approved", "rejected", "relabel", "pending")
        },
    )


def analyze_class_viability(
    rows: Iterable[dict[str, str]],
    *,
    minimum_approved_photos: int = 10,
    minimum_approved_groups: int = 8,
) -> tuple[dict[str, Any], ...]:
    """Measure every proposed class against the fixed V2 retention rule."""
    if minimum_approved_photos < 1 or minimum_approved_groups < 1:
        raise V2FinalizationError("Viability minimums must be positive")
    rows_list = list(rows)
    by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows_list:
        by_class[row["proposed_class"]].append(row)
    results: list[dict[str, Any]] = []
    for class_name in sorted(by_class):
        class_rows = by_class[class_name]
        approved = [
            row
            for row in class_rows
            if row["review_status"].strip().casefold() == "approved"
        ]
        rejected = [
            row
            for row in class_rows
            if row["review_status"].strip().casefold() == "rejected"
        ]
        group_count = len({row["perceptual_group_id"] for row in approved})
        reasons: list[str] = []
        if len(approved) < minimum_approved_photos:
            reasons.append(
                f"approved photographs {len(approved)} < {minimum_approved_photos}"
            )
        if group_count < minimum_approved_groups:
            reasons.append(
                f"approved perceptual groups {group_count} < "
                f"{minimum_approved_groups}"
            )
        results.append(
            {
                "proposed_class": class_name,
                "candidate_count": len(class_rows),
                "approved_count": len(approved),
                "rejected_count": len(rejected),
                "approved_perceptual_group_count": group_count,
                "viable": not reasons,
                "exclusion_reason": "; ".join(reasons),
            }
        )
    return tuple(results)


def validate_approved_set(
    rows: Iterable[dict[str, str]],
    *,
    image_hashes: Mapping[str, str],
    near_duplicate_pairs: Iterable[tuple[str, str]],
) -> dict[str, Any]:
    """Check approved identities, exact hashes, groups, and cross-label conflicts."""
    approved = [
        row for row in rows if row["review_status"].strip().casefold() == "approved"
    ]
    duplicate_review_ids = _duplicates(row["review_id"] for row in approved)
    duplicate_source_ids = _duplicates(row["source_image_id"] for row in approved)
    missing_hash_ids = sorted(
        row["source_image_id"]
        for row in approved
        if row["source_image_id"] not in image_hashes
    )
    by_hash: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_group: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_source = {row["source_image_id"]: row for row in approved}
    for row in approved:
        digest = image_hashes.get(row["source_image_id"])
        if digest:
            by_hash[digest].append(row)
        by_group[row["perceptual_group_id"]].append(row)
    exact_duplicates = [
        {
            "sha256": digest,
            "review_ids": sorted(row["review_id"] for row in hash_rows),
            "perceptual_group_ids": sorted(
                {row["perceptual_group_id"] for row in hash_rows}
            ),
        }
        for digest, hash_rows in sorted(by_hash.items())
        if len(hash_rows) > 1
    ]
    cross_label_conflicts = [
        {
            "perceptual_group_id": group_id,
            "labels": sorted({row["proposed_class"] for row in group_rows}),
            "review_ids": sorted(row["review_id"] for row in group_rows),
        }
        for group_id, group_rows in sorted(by_group.items())
        if len({row["proposed_class"] for row in group_rows}) > 1
    ]
    near_pair_inconsistencies = []
    for left, right in near_duplicate_pairs:
        if left not in by_source or right not in by_source:
            continue
        left_row, right_row = by_source[left], by_source[right]
        if left_row["perceptual_group_id"] != right_row["perceptual_group_id"]:
            near_pair_inconsistencies.append(
                {
                    "left_source_image_id": left,
                    "right_source_image_id": right,
                    "left_group": left_row["perceptual_group_id"],
                    "right_group": right_row["perceptual_group_id"],
                }
            )
    checks = {
        "approved_count": len(approved),
        "duplicate_review_ids": duplicate_review_ids,
        "duplicate_source_image_ids": duplicate_source_ids,
        "missing_image_hash_source_ids": missing_hash_ids,
        "exact_duplicate_groups": exact_duplicates,
        "near_duplicate_pair_group_inconsistencies": near_pair_inconsistencies,
        "cross_label_perceptual_group_conflicts": cross_label_conflicts,
    }
    checks["all_checks_passed"] = not any(
        (
            duplicate_review_ids,
            duplicate_source_ids,
            missing_hash_ids,
            exact_duplicates,
            near_pair_inconsistencies,
            cross_label_conflicts,
        )
    )
    return checks


def group_safe_split(
    rows: Iterable[dict[str, str]],
    viable_classes: Iterable[str],
    *,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> GroupSplitResult:
    """Allocate whole perceptual groups per class using deterministic dynamic programming."""
    ratios = (train_ratio, validation_ratio, test_ratio)
    if any(not math.isfinite(value) or value <= 0.0 for value in ratios):
        raise V2FinalizationError("All V2 split ratios must be finite and positive")
    if not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise V2FinalizationError("V2 split ratios must sum to 1")
    viable = set(viable_classes)
    selected = [
        dict(row)
        for row in rows
        if row["review_status"].strip().casefold() == "approved"
        and row["proposed_class"] in viable
    ]
    if not selected:
        raise V2FinalizationError("No approved viable rows are available to split")
    group_labels: dict[str, set[str]] = defaultdict(set)
    for row in selected:
        group_labels[row["perceptual_group_id"]].add(row["proposed_class"])
    conflicts = {
        group_id: sorted(labels)
        for group_id, labels in group_labels.items()
        if len(labels) > 1
    }
    if conflicts:
        raise V2FinalizationError(
            f"Cross-label group conflicts block splitting: {conflicts}"
        )

    by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in selected:
        by_class[row["proposed_class"]].append(row)
    missing_classes = sorted(viable.difference(by_class))
    if missing_classes:
        raise V2FinalizationError(
            f"Viable classes have no approved rows: {missing_classes}"
        )
    split_rows: dict[str, list[dict[str, str]]] = {
        split_name: [] for split_name in SPLIT_ORDER
    }
    per_class: dict[str, dict[str, int]] = {}
    per_class_groups: dict[str, dict[str, int]] = {}
    for class_name in sorted(by_class):
        groups: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in by_class[class_name]:
            groups[row["perceptual_group_id"]].append(row)
        if len(groups) < len(SPLIT_ORDER):
            raise V2FinalizationError(
                f"Class {class_name!r} has too few groups for all splits"
            )
        assignments = _allocate_class_groups(
            class_name,
            groups,
            ratios=ratios,
            random_seed=random_seed,
        )
        sample_counts: Counter[str] = Counter()
        group_counts: Counter[str] = Counter()
        for group_id, split_name in assignments.items():
            group_counts[split_name] += 1
            for row in groups[group_id]:
                output = {
                    "image_path": row["image_path"],
                    "class_name": class_name,
                    "source_image_id": row["source_image_id"],
                    "perceptual_group_id": group_id,
                    "review_id": row["review_id"],
                    "split": split_name,
                    "source_question": row["source_question"],
                    "source_answer": row["source_answer"],
                }
                split_rows[split_name].append(output)
                sample_counts[split_name] += 1
        per_class[class_name] = {name: sample_counts[name] for name in SPLIT_ORDER}
        per_class_groups[class_name] = {
            name: group_counts[name] for name in SPLIT_ORDER
        }
    for split_name in SPLIT_ORDER:
        split_rows[split_name].sort(
            key=lambda row: (row["class_name"], row["review_id"])
        )
    leakage = _split_leakage_checks(split_rows)
    if not leakage["all_checks_passed"]:
        raise V2FinalizationError(f"Generated split failed leakage checks: {leakage}")
    sizes = {name: len(split_rows[name]) for name in SPLIT_ORDER}
    total = sum(sizes.values())
    summary: dict[str, Any] = {
        "algorithm": (
            "Per class, sort groups deterministically, apply a seed-42 stable shuffle "
            "for tie ordering, then use dynamic programming to assign each complete "
            "perceptual group to train/validation/test. Select the reachable non-empty "
            "allocation minimizing squared sample-count deviation from 70/15/15; "
            "deterministic lexical assignment breaks exact ties."
        ),
        "random_seed": random_seed,
        "target_ratios": {
            name: ratio for name, ratio in zip(SPLIT_ORDER, ratios, strict=True)
        },
        "sizes": sizes,
        "actual_ratios": {name: sizes[name] / total for name in SPLIT_ORDER},
        "ratio_deviations": {
            name: sizes[name] / total - ratio
            for name, ratio in zip(SPLIT_ORDER, ratios, strict=True)
        },
        "per_class_sample_counts": per_class,
        "per_class_group_counts": per_class_groups,
        "leakage_checks": leakage,
        "test_use_policy": (
            "The test split is held out and must not be used for model selection. "
            "Validation macro F1 selects checkpoints."
        ),
    }
    return GroupSplitResult(
        splits={name: tuple(split_rows[name]) for name in SPLIT_ORDER},
        summary=summary,
    )


def _allocate_class_groups(
    class_name: str,
    groups: Mapping[str, list[dict[str, str]]],
    *,
    ratios: tuple[float, float, float],
    random_seed: int,
) -> dict[str, str]:
    ordered = sorted(groups)
    class_seed = int(hashlib.sha256(class_name.encode("utf-8")).hexdigest()[:8], 16)
    random.Random(random_seed + class_seed).shuffle(ordered)
    tie_order = {group_id: index for index, group_id in enumerate(ordered)}
    ordered.sort(key=lambda group_id: (-len(groups[group_id]), tie_order[group_id]))
    total = sum(len(groups[group_id]) for group_id in ordered)
    states: dict[tuple[int, int, int], tuple[int, ...]] = {(0, 0, 0): ()}
    for group_id in ordered:
        size = len(groups[group_id])
        next_states: dict[tuple[int, int, int], tuple[int, ...]] = {}
        for (train_count, validation_count, mask), assignment in states.items():
            for split_index in range(3):
                key = (
                    train_count + (size if split_index == 0 else 0),
                    validation_count + (size if split_index == 1 else 0),
                    mask | (1 << split_index),
                )
                candidate = assignment + (split_index,)
                previous = next_states.get(key)
                if previous is None or candidate < previous:
                    next_states[key] = candidate
        states = next_states
    candidates: list[tuple[tuple[float, ...], tuple[int, ...]]] = []
    for (train_count, validation_count, mask), assignment in states.items():
        if mask != 0b111:
            continue
        test_count = total - train_count - validation_count
        counts = (train_count, validation_count, test_count)
        deviations = tuple(counts[index] - total * ratios[index] for index in range(3))
        score = (
            sum(value * value for value in deviations),
            max(abs(value) for value in deviations),
            abs(deviations[0]),
            abs(deviations[1]),
            float(train_count),
            float(validation_count),
        )
        candidates.append((score, assignment))
    if not candidates:
        raise V2FinalizationError(f"Could not allocate class {class_name!r}")
    _, best_assignment = min(candidates, key=lambda item: (item[0], item[1]))
    return {
        group_id: SPLIT_ORDER[split_index]
        for group_id, split_index in zip(ordered, best_assignment, strict=True)
    }


def _split_leakage_checks(
    splits: Mapping[str, list[dict[str, str]]],
) -> dict[str, Any]:
    group_owners: dict[str, set[str]] = defaultdict(set)
    source_owners: dict[str, set[str]] = defaultdict(set)
    review_owners: dict[str, set[str]] = defaultdict(set)
    for split_name, rows in splits.items():
        for row in rows:
            group_owners[row["perceptual_group_id"]].add(split_name)
            source_owners[row["source_image_id"]].add(split_name)
            review_owners[row["review_id"]].add(split_name)
    group_overlap = sorted(
        key for key, owners in group_owners.items() if len(owners) > 1
    )
    source_overlap = sorted(
        key for key, owners in source_owners.items() if len(owners) > 1
    )
    review_overlap = sorted(
        key for key, owners in review_owners.items() if len(owners) > 1
    )
    checks = {
        "perceptual_group_overlap": group_overlap,
        "source_image_overlap": source_overlap,
        "review_id_overlap": review_overlap,
        "unique_perceptual_groups": len(group_owners),
        "unique_source_images": len(source_owners),
        "unique_review_ids": len(review_owners),
    }
    checks["all_checks_passed"] = not any(
        (group_overlap, source_overlap, review_overlap)
    )
    return checks


def _validate_columns(rows: list[dict[str, str]], description: str) -> None:
    if not rows:
        raise V2FinalizationError(f"{description} is empty")
    missing = sorted(set(V2_REVIEW_COLUMNS).difference(rows[0]))
    if missing:
        raise V2FinalizationError(f"{description} is missing columns: {missing}")


def _unique_by_id(
    rows: list[dict[str, str]], description: str
) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        review_id = row["review_id"]
        if not review_id:
            raise V2FinalizationError(f"{description} contains a blank review_id")
        if review_id in result:
            raise V2FinalizationError(
                f"{description} contains duplicate review_id {review_id}"
            )
        result[review_id] = row
    return result


def _duplicates(values: Iterable[str]) -> list[str]:
    counts = Counter(values)
    return sorted(value for value, count in counts.items() if count > 1)
