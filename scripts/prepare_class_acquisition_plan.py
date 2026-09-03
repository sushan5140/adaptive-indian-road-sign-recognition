"""Freeze a metadata-only pixel-acquisition plan for one project class.

``scripts/prepare_mapillary_expanded_acquisition.py`` produced the frozen
``no_parking`` plan, but its ``main()`` is hardcoded to that class and performs
authenticated API discovery across 146 search boxes before planning. This script
does the planning half only, for any class, from metadata already catalogued in
the repository. It makes no network calls whatsoever.

Every emitted row is stamped ``pixel_download_authorized: no``,
``acquisition_status: proposed_metadata_only`` and
``terms_status: blocked_pending_manual_logged_in_terms_confirmation``. Those
values are written unconditionally and this script provides no option to change
them: a plan records what *would* be downloaded once the logged-in Mapillary
terms are manually confirmed, and freezing a plan is not that confirmation.

Ordering is the same round-robin used for ``no_parking``: candidates grouped by
conservative independence group, ordered within each group by metadata
diversity, then emitted one group at a time so the earliest acquisition_order
values span as many distinct physical signs as possible.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.mapillary_acquisition import build_class_acquisition_plan  # noqa: E402
from data.mapillary_metadata import MapillaryMetadataError  # noqa: E402

#: Column order of the frozen no_parking plan, reproduced exactly.
PLAN_COLUMNS: tuple[str, ...] = (
    "mapillary_image_id",
    "map_feature_id",
    "all_linked_map_feature_ids",
    "exact_mapillary_taxonomy_label",
    "project_class",
    "sequence_id",
    "capture_timestamp_utc",
    "captured_at_epoch_ms",
    "contributor_id",
    "contributor_name",
    "latitude",
    "longitude",
    "geographic_evidence_india",
    "city_search_area",
    "source_reference",
    "api_retrieval_timestamp_utc",
    "taxonomy_mapping_confidence",
    "independence_group_id",
    "review_status",
    "approved_photograph",
    "acquisition_order",
    "acquisition_status",
    "pixel_download_authorized",
    "selection_basis",
    "attribution_title",
    "attribution_contributor",
    "attribution_source_url",
    "attribution_licence",
    "attribution_text_template",
    "terms_status",
)

REQUIRED_UNAUTHORIZED = {
    "acquisition_status": "proposed_metadata_only",
    "pixel_download_authorized": "no",
    "terms_status": "blocked_pending_manual_logged_in_terms_confirmation",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read catalogued candidate metadata."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_plan(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write plan rows using the frozen column order, refusing to overwrite."""
    if path.exists():
        raise MapillaryMetadataError(
            f"{path} already exists; frozen plans are never overwritten in place"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PLAN_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in PLAN_COLUMNS})


def assert_unauthorized(rows: list[dict[str, Any]]) -> None:
    """Fail loudly if any row claims download authorization."""
    for column, expected in REQUIRED_UNAUTHORIZED.items():
        actual = {str(row.get(column, "")) for row in rows}
        if actual != {expected}:
            raise MapillaryMetadataError(
                f"{column} must be {expected!r} on every row, found {sorted(actual)}"
            )


def _display(path: Path) -> str:
    """Show a repo-relative path when possible, else the absolute one."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def main() -> int:
    """Build and write one class's frozen acquisition plan."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-class", required=True)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Catalogued metadata CSV to plan from",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = read_rows(args.source)
    plan = build_class_acquisition_plan(rows, args.project_class)
    assert_unauthorized(plan)
    write_plan(args.output, plan)

    groups = Counter(str(row["independence_group_id"]) for row in plan)
    print(
        f"{args.project_class}: {len(plan)} rows across {len(groups)} independence "
        f"groups -> {_display(args.output)}"
    )
    print(
        f"  largest group {max(groups.values())} rows, "
        f"smallest {min(groups.values())}; "
        f"first {len(groups)} rows span every group once"
    )
    print(
        "  pixel_download_authorized=no, acquisition_status=proposed_metadata_only, "
        "terms_status=blocked_pending_manual_logged_in_terms_confirmation"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
