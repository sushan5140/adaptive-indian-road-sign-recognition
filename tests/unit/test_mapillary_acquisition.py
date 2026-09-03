"""Tests for metadata-only Mapillary acquisition planning."""

from __future__ import annotations

import pytest

from data.mapillary_acquisition import (
    build_no_parking_acquisition_plan,
    merge_expanded_candidates,
)
from data.mapillary_metadata import MapillaryMetadataError


def _row(
    image_id: str,
    feature_id: str,
    project_class: str,
    group_id: str,
    source: str,
) -> dict[str, object]:
    return {
        "mapillary_image_id": image_id,
        "map_feature_id": feature_id,
        "all_linked_map_feature_ids": feature_id,
        "project_class": project_class,
        "independence_group_id": group_id,
        "discovery_source": source,
        "latitude": 20.0 + int(feature_id[-1]),
        "longitude": 75.0,
        "captured_at_epoch_ms": 1_000_000,
        "sequence_id": f"sequence-{feature_id}",
        "contributor_id": f"user-{feature_id}",
        "contributor_name": f"name-{feature_id}",
        "capture_timestamp_utc": "2025-01-01T00:00:00+00:00",
        "city_search_area": "Test City",
        "source_reference": f"https://www.mapillary.com/app/?pKey={image_id}",
    }


def test_no_parking_plan_retains_every_group_and_blocks_pixels() -> None:
    rows = [
        _row("image-1", "feature-1", "no_parking", "group-a", "r07"),
        _row("image-2", "feature-1", "no_parking", "group-a", "r07"),
        _row("image-3", "feature-2", "no_parking", "group-b", "r07"),
    ]

    plan = build_no_parking_acquisition_plan(rows)

    assert [row["mapillary_image_id"] for row in plan[:2]] == ["image-1", "image-3"]
    assert {row["independence_group_id"] for row in plan} == {"group-a", "group-b"}
    assert {row["pixel_download_authorized"] for row in plan} == {"no"}
    assert {row["terms_status"] for row in plan} == {
        "blocked_pending_manual_logged_in_terms_confirmation"
    }


def test_no_parking_plan_rejects_duplicate_images() -> None:
    row = _row("image-1", "feature-1", "no_parking", "group-a", "r07")

    with pytest.raises(MapillaryMetadataError, match="must be unique"):
        build_no_parking_acquisition_plan([row, row])


def test_expanded_merge_removes_r07_image_and_recomputes_groups() -> None:
    existing = [_row("image-1", "feature-1", "stop", "old-group", "r07")]
    duplicate = _row("image-1", "feature-1", "stop", "", "expanded")
    new = _row("image-2", "feature-2", "stop", "", "expanded")
    new["latitude"] = 24.0

    combined, removed = merge_expanded_candidates(existing, [duplicate, new])

    assert removed == 1
    assert {row["mapillary_image_id"] for row in combined} == {"image-1", "image-2"}
    assert combined[0]["r07_independence_group_id"] == "old-group"


def test_expanded_merge_rejects_no_parking() -> None:
    prohibited = _row("image-2", "feature-2", "no_parking", "", "expanded")

    with pytest.raises(MapillaryMetadataError, match="prohibited class"):
        merge_expanded_candidates([], [prohibited])
