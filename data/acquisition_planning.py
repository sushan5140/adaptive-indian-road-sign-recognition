"""Deterministic planning helpers for unseen-class data acquisition."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


class AcquisitionPlanningError(ValueError):
    """Raised when acquisition inputs violate the locked unseen protocol."""


@dataclass(frozen=True, slots=True)
class AcquisitionClassState:
    """Measured visual-review state for one locked unseen class."""

    class_name: str
    visually_accepted_photos: int
    visually_accepted_groups: int
    licence_ready_photos: int = 0


def build_acquisition_plan(
    states: Sequence[AcquisitionClassState],
    *,
    mapillary_counts: Mapping[str, int],
    mapillary_status: str,
    source_recommendations: Mapping[str, str],
    minimum_photos: int = 30,
    minimum_groups: int = 15,
) -> list[dict[str, Any]]:
    """Build per-class gaps without counting metadata as approved photographs."""
    expected = {
        "stop",
        "no_left_turn",
        "maximum_speed_limit_50_km_h",
        "no_parking",
        "bus_stop",
    }
    names = [state.class_name for state in states]
    if len(names) != len(set(names)) or set(names) != expected:
        raise AcquisitionPlanningError(
            f"States must contain each locked unseen class exactly once: {names}"
        )
    if minimum_photos <= 0 or minimum_groups <= 0:
        raise AcquisitionPlanningError("Acquisition thresholds must be positive")
    if set(mapillary_counts) != expected or set(source_recommendations) != expected:
        raise AcquisitionPlanningError(
            "Planning mappings must cover all locked classes"
        )

    rows: list[dict[str, Any]] = []
    for state in states:
        rows.append(
            {
                "class_name": state.class_name,
                "visually_accepted_photos": state.visually_accepted_photos,
                "visually_accepted_independent_groups": state.visually_accepted_groups,
                "licence_ready_photos": state.licence_ready_photos,
                "minimum_photos": minimum_photos,
                "minimum_independent_groups": minimum_groups,
                "remaining_photo_quantity_gap": max(
                    0, minimum_photos - state.visually_accepted_photos
                ),
                "remaining_group_gap": max(
                    0, minimum_groups - state.visually_accepted_groups
                ),
                "mapillary_metadata_candidate_count": mapillary_counts[
                    state.class_name
                ],
                "mapillary_query_status": mapillary_status,
                "metadata_candidates_counted_as_approved": "no",
                "recommended_source": source_recommendations[state.class_name],
                "acquisition_readiness": _readiness(
                    state, minimum_photos, minimum_groups
                ),
            }
        )
    return rows


def _readiness(
    state: AcquisitionClassState, minimum_photos: int, minimum_groups: int
) -> str:
    if state.visually_accepted_photos < minimum_photos:
        return "additional_photographs_required"
    if state.visually_accepted_groups < minimum_groups:
        return "additional_independent_groups_required"
    if state.licence_ready_photos < minimum_photos:
        return "quantity_met_but_licence_blocked"
    return "ready_for_separate_protocol_validation"
