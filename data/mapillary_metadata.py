"""Validation and conservative grouping for Mapillary sign metadata."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


class MapillaryMetadataError(ValueError):
    """Raised when Mapillary metadata violates the locked discovery protocol."""


EXACT_TAXONOMY_PREFIXES: dict[str, str] = {
    "stop": "regulatory--stop--",
    "maximum_speed_limit_50_km_h": "regulatory--maximum-speed-limit-50--",
    "no_left_turn": "regulatory--no-left-turn--",
    "no_parking": "regulatory--no-parking--",
}


def map_exact_taxonomy_label(taxonomy_label: str) -> str | None:
    """Map an exact semantic Mapillary taxonomy prefix to a project class."""
    normalized = taxonomy_label.strip().lower()
    for project_class, prefix in EXACT_TAXONOMY_PREFIXES.items():
        if normalized.startswith(prefix) and len(normalized) > len(prefix):
            return project_class
    return None


def assign_conservative_independence_groups(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    """Group feature-image rows that may depict the same physical sign."""
    if not rows:
        return {}
    feature_rows: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        feature_id = str(row.get("map_feature_id", ""))
        image_id = str(row.get("mapillary_image_id", ""))
        if not feature_id or not image_id:
            raise MapillaryMetadataError("Rows require map feature and image IDs")
        existing = feature_rows.get(feature_id)
        if existing is not None and existing.get("project_class") != row.get(
            "project_class"
        ):
            raise MapillaryMetadataError(
                f"Feature {feature_id} has conflicting project classes"
            )
        feature_rows.setdefault(feature_id, row)

    feature_ids = sorted(feature_rows)
    parent = {feature_id: feature_id for feature_id in feature_ids}

    def find(feature_id: str) -> str:
        while parent[feature_id] != feature_id:
            parent[feature_id] = parent[parent[feature_id]]
            feature_id = parent[feature_id]
        return feature_id

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    by_class: dict[str, list[str]] = defaultdict(list)
    by_image: dict[str, list[str]] = defaultdict(list)
    for feature_id, row in feature_rows.items():
        by_class[str(row["project_class"])].append(feature_id)
    for row in rows:
        by_image[str(row["mapillary_image_id"])].append(str(row["map_feature_id"]))
    for image_features in by_image.values():
        first = image_features[0]
        for feature_id in image_features[1:]:
            if (
                feature_rows[first]["project_class"]
                == feature_rows[feature_id]["project_class"]
            ):
                union(first, feature_id)

    for class_features in by_class.values():
        ordered = sorted(class_features)
        for left_index, left_id in enumerate(ordered):
            left = feature_rows[left_id]
            for right_id in ordered[left_index + 1 :]:
                right = feature_rows[right_id]
                distance = haversine_metres(
                    float(left["latitude"]),
                    float(left["longitude"]),
                    float(right["latitude"]),
                    float(right["longitude"]),
                )
                if distance <= 20.0:
                    union(left_id, right_id)
                    continue
                same_sequence = bool(left.get("sequence_id")) and left.get(
                    "sequence_id"
                ) == right.get("sequence_id")
                same_contributor = bool(left.get("contributor_id")) and left.get(
                    "contributor_id"
                ) == right.get("contributor_id")
                time_delta = _capture_delta_seconds(left, right)
                if (
                    distance <= 75.0
                    and (same_sequence or same_contributor)
                    and time_delta is not None
                    and time_delta <= 120.0
                ):
                    union(left_id, right_id)

    roots = sorted({find(feature_id) for feature_id in feature_ids})
    root_to_group = {
        root: f"mapillary_india_group_{index:05d}"
        for index, root in enumerate(roots, 1)
    }
    return {feature_id: root_to_group[find(feature_id)] for feature_id in feature_ids}


def haversine_metres(
    latitude_a: float,
    longitude_a: float,
    latitude_b: float,
    longitude_b: float,
) -> float:
    """Calculate great-circle distance between two WGS84 points in metres."""
    radius = 6_371_000.0
    phi_a, phi_b = math.radians(latitude_a), math.radians(latitude_b)
    delta_phi = math.radians(latitude_b - latitude_a)
    delta_lambda = math.radians(longitude_b - longitude_a)
    value = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi_a) * math.cos(phi_b) * math.sin(delta_lambda / 2.0) ** 2
    )
    return 2.0 * radius * math.atan2(math.sqrt(value), math.sqrt(1.0 - value))


def _capture_delta_seconds(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> float | None:
    left_timestamp = left.get("captured_at_epoch_ms")
    right_timestamp = right.get("captured_at_epoch_ms")
    if left_timestamp in (None, "") or right_timestamp in (None, ""):
        return None
    return abs(float(str(left_timestamp)) - float(str(right_timestamp))) / 1000.0
