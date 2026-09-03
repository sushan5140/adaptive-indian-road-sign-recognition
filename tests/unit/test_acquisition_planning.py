"""Tests for locked unseen-class acquisition planning."""

from __future__ import annotations

import pytest

from data.acquisition_planning import (
    AcquisitionClassState,
    AcquisitionPlanningError,
    build_acquisition_plan,
)


def _states() -> tuple[AcquisitionClassState, ...]:
    return (
        AcquisitionClassState("stop", 0, 0),
        AcquisitionClassState("no_left_turn", 1, 1),
        AcquisitionClassState("maximum_speed_limit_50_km_h", 0, 0),
        AcquisitionClassState("no_parking", 4, 3),
        AcquisitionClassState("bus_stop", 34, 29),
    )


def test_plan_preserves_thresholds_and_does_not_count_metadata() -> None:
    names = {state.class_name for state in _states()}
    rows = build_acquisition_plan(
        _states(),
        mapillary_counts={name: 0 for name in names},
        mapillary_status="not_queried_authentication_unavailable",
        source_recommendations={name: "source" for name in names},
    )
    by_class = {row["class_name"]: row for row in rows}

    assert by_class["no_left_turn"]["remaining_photo_quantity_gap"] == 29
    assert by_class["no_left_turn"]["remaining_group_gap"] == 14
    assert by_class["no_parking"]["remaining_photo_quantity_gap"] == 26
    assert by_class["no_parking"]["remaining_group_gap"] == 12
    assert by_class["bus_stop"]["remaining_photo_quantity_gap"] == 0
    assert by_class["bus_stop"]["acquisition_readiness"] == (
        "quantity_met_but_licence_blocked"
    )
    assert all(row["metadata_candidates_counted_as_approved"] == "no" for row in rows)


def test_missing_locked_class_fails_closed() -> None:
    states = _states()[:-1]
    names = {state.class_name for state in states}
    with pytest.raises(AcquisitionPlanningError, match="exactly once"):
        build_acquisition_plan(
            states,
            mapillary_counts={name: 0 for name in names},
            mapillary_status="not_queried",
            source_recommendations={name: "source" for name in names},
        )
