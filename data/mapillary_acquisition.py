"""Planning helpers for metadata-only Mapillary acquisition phases."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from data.mapillary_metadata import (
    MapillaryMetadataError,
    assign_conservative_independence_groups,
)

EXPANDED_TARGET_CLASSES = frozenset(
    {
        "stop",
        "maximum_speed_limit_50_km_h",
        "no_left_turn",
    }
)


def build_no_parking_acquisition_plan(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build a deterministic metadata-diverse plan from a frozen no-parking pool."""
    no_parking = [dict(row) for row in rows if row.get("project_class") == "no_parking"]
    if not no_parking:
        raise MapillaryMetadataError("The frozen r07 no-parking pool is empty")
    _validate_unique_image_ids(no_parking)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in no_parking:
        group_id = str(row.get("independence_group_id", ""))
        if not group_id:
            raise MapillaryMetadataError(
                "No-parking rows require independence_group_id"
            )
        grouped[group_id].append(row)

    for group_rows in grouped.values():
        group_rows.sort(key=_metadata_diversity_key)

    ordered: list[dict[str, Any]] = []
    depth = 0
    while any(depth < len(group_rows) for group_rows in grouped.values()):
        for group_id in sorted(grouped):
            if depth < len(grouped[group_id]):
                ordered.append(grouped[group_id][depth])
        depth += 1

    for index, row in enumerate(ordered, 1):
        contributor = str(row.get("contributor_name", "")) or "unknown contributor"
        source = str(row.get("source_reference", ""))
        row.update(
            {
                "acquisition_order": index,
                "acquisition_status": "proposed_metadata_only",
                "pixel_download_authorized": "no",
                "selection_basis": (
                    "All frozen r07 candidates retained; round-robin order across "
                    "dependency groups, then metadata diversity within each group"
                ),
                "attribution_title": f"Mapillary image {row['mapillary_image_id']}",
                "attribution_contributor": contributor,
                "attribution_source_url": source,
                "attribution_licence": "CC BY-SA (version to verify at download time)",
                "attribution_text_template": (
                    f"Mapillary image {row['mapillary_image_id']} by {contributor}; "
                    f"source {source}; licensed under CC BY-SA"
                ),
                "terms_status": ("blocked_pending_manual_logged_in_terms_confirmation"),
            }
        )
    return ordered


def merge_expanded_candidates(
    r07_rows: Sequence[Mapping[str, Any]],
    newly_discovered_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Merge new target metadata with r07 while removing exact image duplicates."""
    existing_image_ids = {str(row["mapillary_image_id"]) for row in r07_rows}
    retained_existing = [
        dict(row)
        for row in r07_rows
        if str(row.get("project_class", "")) in EXPANDED_TARGET_CLASSES
    ]
    eligible_new: dict[str, list[dict[str, Any]]] = defaultdict(list)
    removed = 0
    for source_row in newly_discovered_rows:
        row = dict(source_row)
        project_class = str(row.get("project_class", ""))
        if project_class not in EXPANDED_TARGET_CLASSES:
            raise MapillaryMetadataError(
                f"Expanded discovery contains prohibited class: {project_class}"
            )
        image_id = str(row.get("mapillary_image_id", ""))
        if not image_id or image_id in existing_image_ids:
            removed += 1
            continue
        eligible_new[image_id].append(row)

    retained_new: list[dict[str, Any]] = []
    for image_id in sorted(eligible_new):
        image_rows = eligible_new[image_id]
        classes = {str(row["project_class"]) for row in image_rows}
        if len(classes) != 1:
            removed += len(image_rows)
            continue
        retained_new.append(
            min(image_rows, key=lambda row: str(row.get("map_feature_id", "")))
        )
        removed += len(image_rows) - 1

    combined = retained_existing + retained_new
    _validate_unique_image_ids(combined)
    associations: list[dict[str, Any]] = []
    for row in combined:
        linked_ids = str(row.get("all_linked_map_feature_ids", "")).split(";")
        for feature_id in linked_ids:
            feature_id = feature_id.strip()
            if not feature_id:
                continue
            association = dict(row)
            association["map_feature_id"] = feature_id
            associations.append(association)
    group_map = assign_conservative_independence_groups(associations)
    for row in combined:
        row["r07_independence_group_id"] = (
            str(row.get("independence_group_id", ""))
            if str(row.get("discovery_source", "")) == "r07"
            else ""
        )
        linked_ids = [
            value.strip()
            for value in str(row.get("all_linked_map_feature_ids", "")).split(";")
            if value.strip()
        ]
        row["independence_group_id"] = min(group_map[value] for value in linked_ids)
    combined.sort(
        key=lambda row: (str(row["project_class"]), str(row["mapillary_image_id"]))
    )
    return combined, removed


def _metadata_diversity_key(row: Mapping[str, Any]) -> tuple[str, ...]:
    return (
        str(row.get("city_search_area", "")),
        str(row.get("contributor_id", "")),
        str(row.get("sequence_id", "")),
        str(row.get("capture_timestamp_utc", "")),
        str(row.get("mapillary_image_id", "")),
    )


def _validate_unique_image_ids(rows: Sequence[Mapping[str, Any]]) -> None:
    image_ids = [str(row.get("mapillary_image_id", "")) for row in rows]
    if any(not image_id for image_id in image_ids):
        raise MapillaryMetadataError("Mapillary candidate rows require image IDs")
    if len(image_ids) != len(set(image_ids)):
        raise MapillaryMetadataError("Mapillary candidate image IDs must be unique")
