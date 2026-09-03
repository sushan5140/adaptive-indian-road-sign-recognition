"""Tests for Mapillary taxonomy and independence grouping."""

from __future__ import annotations

from data.mapillary_metadata import (
    assign_conservative_independence_groups,
    map_exact_taxonomy_label,
)


def test_exact_taxonomy_mapping_rejects_approximate_labels() -> None:
    assert map_exact_taxonomy_label("regulatory--stop--g1") == "stop"
    assert (
        map_exact_taxonomy_label("regulatory--maximum-speed-limit-50--g2")
        == "maximum_speed_limit_50_km_h"
    )
    assert map_exact_taxonomy_label("regulatory--no-left-turn--g1") == "no_left_turn"
    assert map_exact_taxonomy_label("regulatory--no-parking--g1") == "no_parking"
    assert map_exact_taxonomy_label("regulatory--no-turn--g1") is None
    assert map_exact_taxonomy_label("regulatory--maximum-speed-limit--g1") is None


def test_same_feature_and_nearby_features_share_group() -> None:
    rows = (
        {
            "map_feature_id": "feature-a",
            "mapillary_image_id": "image-1",
            "project_class": "stop",
            "latitude": 28.600000,
            "longitude": 77.200000,
            "sequence_id": "sequence-1",
            "contributor_id": "user-1",
            "captured_at_epoch_ms": 1_000_000,
        },
        {
            "map_feature_id": "feature-a",
            "mapillary_image_id": "image-2",
            "project_class": "stop",
            "latitude": 28.600000,
            "longitude": 77.200000,
            "sequence_id": "sequence-1",
            "contributor_id": "user-1",
            "captured_at_epoch_ms": 1_001_000,
        },
        {
            "map_feature_id": "feature-b",
            "mapillary_image_id": "image-3",
            "project_class": "stop",
            "latitude": 28.600050,
            "longitude": 77.200050,
            "sequence_id": "sequence-2",
            "contributor_id": "user-2",
            "captured_at_epoch_ms": 2_000_000,
        },
    )
    groups = assign_conservative_independence_groups(rows)

    assert groups["feature-a"] == groups["feature-b"]


def test_distant_features_in_same_sequence_remain_independent() -> None:
    rows = (
        {
            "map_feature_id": "feature-a",
            "mapillary_image_id": "image-1",
            "project_class": "no_parking",
            "latitude": 28.60,
            "longitude": 77.20,
            "sequence_id": "sequence-1",
            "contributor_id": "user-1",
            "captured_at_epoch_ms": 1_000_000,
        },
        {
            "map_feature_id": "feature-b",
            "mapillary_image_id": "image-2",
            "project_class": "no_parking",
            "latitude": 28.61,
            "longitude": 77.21,
            "sequence_id": "sequence-1",
            "contributor_id": "user-1",
            "captured_at_epoch_ms": 1_010_000,
        },
    )
    groups = assign_conservative_independence_groups(rows)

    assert groups["feature-a"] != groups["feature-b"]
