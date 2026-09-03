"""Unit tests for the open-set decision policy."""

import math

import pytest

from inference.decision import (
    BaseEvidence,
    OpenSetDecisionError,
    OpenSetThresholds,
    PrototypeEvidence,
    Strategy,
    Verdict,
    build_base_evidence,
    build_prototype_evidence,
    decide,
    decide_from_scores,
    softmax,
)
from models.prototype_registry import PrototypeMatch

BASE_CLASSES = ("stop", "give_way", "no_entry")


def _match(label: str, similarity: float) -> PrototypeMatch:
    return PrototypeMatch(label=label, similarity=similarity, metadata={})


def _thresholds(**overrides: object) -> OpenSetThresholds:
    defaults: dict[str, object] = {
        "base_confidence_threshold": 0.70,
        "prototype_similarity_threshold": 0.75,
    }
    defaults.update(overrides)
    return OpenSetThresholds(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
def test_default_thresholds_are_not_marked_calibrated() -> None:
    # Placeholder values must never be reported as measured.
    assert OpenSetThresholds().calibrated is False


@pytest.mark.parametrize(
    "overrides",
    [
        {"base_confidence_threshold": -0.1},
        {"base_confidence_threshold": 1.1},
        {"prototype_similarity_threshold": -1.5},
        {"prototype_similarity_threshold": 1.5},
        {"prototype_margin": -0.1},
        {"unknown_label": "   "},
    ],
)
def test_thresholds_reject_out_of_range_values(overrides: dict[str, object]) -> None:
    with pytest.raises(OpenSetDecisionError):
        _thresholds(**overrides)


def test_thresholds_from_config_reads_supported_keys() -> None:
    thresholds = OpenSetThresholds.from_config(
        {
            "base_confidence_threshold": 0.5,
            "prototype_similarity_threshold": 0.6,
            "prototype_margin": 0.05,
            "strategy": "prototype_priority",
            "unknown_label": "unrecognised",
            "calibrated": True,
            "calibration_reference": "run-123",
            "prototype_registry_path": "artifacts/prototypes/registry.npz",
        }
    )

    assert thresholds.base_confidence_threshold == 0.5
    assert thresholds.strategy is Strategy.PROTOTYPE_PRIORITY
    assert thresholds.unknown_label == "unrecognised"
    assert thresholds.calibrated is True
    assert thresholds.calibration_reference == "run-123"


def test_thresholds_from_config_rejects_unknown_keys() -> None:
    # A typo must fail loudly rather than silently leave a default in place.
    with pytest.raises(OpenSetDecisionError, match="Unsupported open_set"):
        OpenSetThresholds.from_config({"base_confidence_treshold": 0.5})


def test_thresholds_from_config_rejects_unknown_strategy() -> None:
    with pytest.raises(OpenSetDecisionError, match="strategy"):
        OpenSetThresholds.from_config({"strategy": "vibes"})


def test_thresholds_from_config_rejects_boolean_numbers() -> None:
    with pytest.raises(OpenSetDecisionError, match="must be a number"):
        OpenSetThresholds.from_config({"base_confidence_threshold": True})


def test_thresholds_round_trip_to_dict() -> None:
    payload = _thresholds().to_dict()
    assert payload["strategy"] == "classifier_first"
    assert payload["calibrated"] is False


# ---------------------------------------------------------------------------
# softmax
# ---------------------------------------------------------------------------
def test_softmax_sums_to_one_and_preserves_order() -> None:
    probabilities = softmax([1.0, 3.0, 2.0])

    assert math.isclose(sum(probabilities), 1.0, abs_tol=1e-9)
    assert probabilities[1] > probabilities[2] > probabilities[0]


def test_softmax_is_stable_for_large_logits() -> None:
    probabilities = softmax([1000.0, 999.0])

    assert all(math.isfinite(value) for value in probabilities)
    assert math.isclose(sum(probabilities), 1.0, abs_tol=1e-9)


@pytest.mark.parametrize("logits", [[], [1.0, math.nan], [math.inf]])
def test_softmax_rejects_invalid_logits(logits: list[float]) -> None:
    with pytest.raises(OpenSetDecisionError):
        softmax(logits)


# ---------------------------------------------------------------------------
# Evidence builders
# ---------------------------------------------------------------------------
def test_build_base_evidence_reports_top_class_and_ranking() -> None:
    evidence = build_base_evidence([0.1, 0.7, 0.2], BASE_CLASSES, top_k=2)

    assert evidence.label == "give_way"
    assert evidence.confidence == pytest.approx(0.7)
    assert evidence.ranking == (("give_way", 0.7), ("no_entry", 0.2))
    assert evidence.available is True


def test_build_base_evidence_breaks_ties_by_class_index() -> None:
    evidence = build_base_evidence([0.5, 0.5, 0.0], BASE_CLASSES)

    assert evidence.label == "stop"


@pytest.mark.parametrize(
    "probabilities",
    [
        [0.5, 0.5],  # length mismatch with BASE_CLASSES
        [0.2, 0.2, 0.2],  # does not sum to one
        [0.5, 0.6, -0.1],  # negative probability
        [0.5, 0.5, math.nan],  # non-finite
    ],
)
def test_build_base_evidence_rejects_invalid_distributions(
    probabilities: list[float],
) -> None:
    with pytest.raises(OpenSetDecisionError):
        build_base_evidence(probabilities, BASE_CLASSES)


def test_build_base_evidence_rejects_non_positive_top_k() -> None:
    with pytest.raises(OpenSetDecisionError, match="top_k"):
        build_base_evidence([1.0, 0.0, 0.0], BASE_CLASSES, top_k=0)


def test_build_prototype_evidence_handles_empty_registry() -> None:
    evidence = build_prototype_evidence([])

    assert evidence.label is None
    assert evidence.similarity == -1.0
    assert evidence.margin is None
    assert evidence.registry_size == 0


def test_build_prototype_evidence_has_no_margin_for_one_class() -> None:
    evidence = build_prototype_evidence([_match("school_ahead", 0.9)])

    assert evidence.label == "school_ahead"
    assert evidence.margin is None
    assert evidence.registry_size == 1


def test_build_prototype_evidence_computes_margin() -> None:
    evidence = build_prototype_evidence(
        [_match("school_ahead", 0.90), _match("road_hump", 0.82)]
    )

    assert evidence.runner_up_label == "road_hump"
    assert evidence.margin == pytest.approx(0.08)
    assert evidence.registry_size == 2


# ---------------------------------------------------------------------------
# decide
# ---------------------------------------------------------------------------
def test_confident_base_class_is_accepted() -> None:
    decision = decide_from_scores(
        [0.05, 0.90, 0.05], BASE_CLASSES, [], thresholds=_thresholds()
    )

    assert decision.verdict is Verdict.BASE_CLASS
    assert decision.label == "give_way"
    assert decision.score == pytest.approx(0.90)
    assert decision.is_unknown is False
    assert "base classifier confidence" in decision.reason


def test_confidence_exactly_at_the_threshold_is_accepted() -> None:
    decision = decide_from_scores(
        [0.30, 0.70, 0.0], BASE_CLASSES, [], thresholds=_thresholds()
    )

    assert decision.verdict is Verdict.BASE_CLASS


def test_registered_class_wins_when_base_is_unsure() -> None:
    decision = decide_from_scores(
        [0.34, 0.33, 0.33],
        BASE_CLASSES,
        [_match("school_ahead", 0.88)],
        thresholds=_thresholds(),
    )

    assert decision.verdict is Verdict.REGISTERED_CLASS
    assert decision.label == "school_ahead"
    assert decision.score == pytest.approx(0.88)
    assert "registered prototype similarity" in decision.reason


def test_unknown_when_neither_source_qualifies() -> None:
    decision = decide_from_scores(
        [0.34, 0.33, 0.33],
        BASE_CLASSES,
        [_match("school_ahead", 0.40)],
        thresholds=_thresholds(),
    )

    assert decision.verdict is Verdict.UNKNOWN
    assert decision.label == "unknown"
    # The score reports the strongest rejected evidence, not zero.
    assert decision.score == pytest.approx(0.40)
    assert "below the threshold" in decision.reason


def test_unknown_reports_an_empty_registry_explicitly() -> None:
    decision = decide_from_scores(
        [0.34, 0.33, 0.33], BASE_CLASSES, [], thresholds=_thresholds()
    )

    assert decision.verdict is Verdict.UNKNOWN
    assert "no incremental class is registered" in decision.reason


def test_custom_unknown_label_is_used() -> None:
    decision = decide_from_scores(
        [0.34, 0.33, 0.33],
        BASE_CLASSES,
        [],
        thresholds=_thresholds(unknown_label="unrecognised_sign"),
    )

    assert decision.label == "unrecognised_sign"


def test_margin_rejects_a_near_tie_between_registered_classes() -> None:
    decision = decide_from_scores(
        [0.34, 0.33, 0.33],
        BASE_CLASSES,
        [_match("school_ahead", 0.86), _match("children_playing", 0.85)],
        thresholds=_thresholds(prototype_margin=0.05),
    )

    assert decision.verdict is Verdict.UNKNOWN
    assert "margin" in decision.reason


def test_margin_is_skipped_when_only_one_class_is_registered() -> None:
    decision = decide_from_scores(
        [0.34, 0.33, 0.33],
        BASE_CLASSES,
        [_match("school_ahead", 0.86)],
        thresholds=_thresholds(prototype_margin=0.50),
    )

    assert decision.verdict is Verdict.REGISTERED_CLASS
    assert "no runner-up" in decision.reason


def test_classifier_first_prefers_a_confident_base_class() -> None:
    decision = decide_from_scores(
        [0.05, 0.90, 0.05],
        BASE_CLASSES,
        [_match("school_ahead", 0.99)],
        thresholds=_thresholds(strategy=Strategy.CLASSIFIER_FIRST),
    )

    assert decision.verdict is Verdict.BASE_CLASS
    assert decision.label == "give_way"


def test_prototype_priority_prefers_a_qualifying_registered_class() -> None:
    decision = decide_from_scores(
        [0.05, 0.90, 0.05],
        BASE_CLASSES,
        [_match("school_ahead", 0.99)],
        thresholds=_thresholds(strategy=Strategy.PROTOTYPE_PRIORITY),
    )

    assert decision.verdict is Verdict.REGISTERED_CLASS
    assert decision.label == "school_ahead"


def test_prototype_priority_still_falls_back_to_the_base_class() -> None:
    decision = decide_from_scores(
        [0.05, 0.90, 0.05],
        BASE_CLASSES,
        [_match("school_ahead", 0.10)],
        thresholds=_thresholds(strategy=Strategy.PROTOTYPE_PRIORITY),
    )

    assert decision.verdict is Verdict.BASE_CLASS


def test_registry_only_mode_needs_no_base_classifier() -> None:
    decision = decide_from_scores(
        [], [], [_match("school_ahead", 0.88)], thresholds=_thresholds()
    )

    assert decision.verdict is Verdict.REGISTERED_CLASS
    assert decision.base.available is False
    assert decision.base.label is None


def test_registry_only_mode_reports_the_missing_classifier_when_unknown() -> None:
    decision = decide_from_scores([], [], [], thresholds=_thresholds())

    assert decision.verdict is Verdict.UNKNOWN
    assert "no base classifier" in decision.reason


def test_decision_carries_the_thresholds_it_was_made_under() -> None:
    thresholds = _thresholds(calibrated=True, calibration_reference="run-abc")
    decision = decide_from_scores(
        [0.9, 0.05, 0.05], BASE_CLASSES, [], thresholds=thresholds
    )

    assert decision.uses_calibrated_thresholds is True
    assert decision.thresholds.calibration_reference == "run-abc"


def test_decision_serializes_evidence_for_reporting() -> None:
    decision = decide_from_scores(
        [0.05, 0.90, 0.05],
        BASE_CLASSES,
        [_match("school_ahead", 0.60)],
        thresholds=_thresholds(),
    )
    payload = decision.to_dict()

    assert payload["verdict"] == "base_class"
    assert payload["base"]["ranking"][0]["label"] == "give_way"
    assert payload["prototype"]["label"] == "school_ahead"
    assert payload["thresholds"]["calibrated"] is False


def test_decide_accepts_prebuilt_evidence() -> None:
    decision = decide(
        base=BaseEvidence(label=None, confidence=0.0, available=False),
        prototype=PrototypeEvidence(label="school_ahead", similarity=0.95),
        thresholds=_thresholds(),
    )

    assert decision.verdict is Verdict.REGISTERED_CLASS


def test_decide_uses_uncalibrated_defaults_when_thresholds_are_omitted() -> None:
    decision = decide(
        base=build_base_evidence([0.9, 0.05, 0.05], BASE_CLASSES),
        prototype=build_prototype_evidence([]),
    )

    assert decision.verdict is Verdict.BASE_CLASS
    assert decision.uses_calibrated_thresholds is False
