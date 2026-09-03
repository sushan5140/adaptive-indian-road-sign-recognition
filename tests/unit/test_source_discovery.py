"""Tests for source-discovery leakage and recovery helpers."""

from __future__ import annotations

import pytest

from data.source_discovery import (
    SourceDiscoveryError,
    build_perceptual_groups,
    eligible_counts_by_class,
    frozen_split_overlap,
    make_split_evidence,
)


def test_groups_are_deterministic_connected_components() -> None:
    pairs = (
        {"left_image": "b.jpg", "right_image": "c.jpg"},
        {"left_image": "a.jpg", "right_image": "b.jpg"},
    )
    groups = build_perceptual_groups(("d.jpg", "c.jpg", "a.jpg", "b.jpg"), pairs)

    assert groups["a.jpg"] == groups["b.jpg"] == groups["c.jpg"]
    assert groups["d.jpg"] != groups["a.jpg"]


def test_unknown_pair_image_fails_closed() -> None:
    with pytest.raises(SourceDiscoveryError, match="unknown image"):
        build_perceptual_groups(
            ("a.jpg",),
            ({"left_image": "a.jpg", "right_image": "missing.jpg"},),
        )


def test_overlap_reports_source_group_and_hash_matches() -> None:
    evidence = make_split_evidence(
        (
            {
                "source_image_id": "base.jpg",
                "perceptual_group_id": "group_1",
            },
        ),
        {"base.jpg": "digest"},
    )

    overlap, reason = frozen_split_overlap(
        source_id="base.jpg",
        perceptual_group_id="group_1",
        sha256="digest",
        evidence=evidence,
    )

    assert overlap is True
    assert reason == "source_image_id;perceptual_group_id;exact_sha256"


def test_duplicate_split_sources_fail_closed() -> None:
    rows = (
        {"source_image_id": "same.jpg", "perceptual_group_id": "group_1"},
        {"source_image_id": "same.jpg", "perceptual_group_id": "group_2"},
    )
    with pytest.raises(SourceDiscoveryError, match="Duplicate source ID"):
        make_split_evidence(rows, {"same.jpg": "digest"})


def test_eligible_counts_use_conservative_dependency_groups() -> None:
    rows = (
        {
            "eligible_for_unseen_review": "yes",
            "newly_proposed_unseen_label": "no_parking",
            "conservative_dependency_group": "source_1",
        },
        {
            "eligible_for_unseen_review": "yes",
            "newly_proposed_unseen_label": "no_parking",
            "conservative_dependency_group": "source_1",
        },
        {
            "eligible_for_unseen_review": "no",
            "newly_proposed_unseen_label": "no_parking",
            "conservative_dependency_group": "source_2",
        },
    )

    assert eligible_counts_by_class(rows) == {
        "no_parking": {"eligible_images": 2, "independent_groups": 1}
    }
