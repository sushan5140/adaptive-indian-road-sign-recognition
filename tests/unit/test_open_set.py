"""Unit tests for explicit open-set scoring and conservative decisions."""

import numpy as np
import pytest

from inference.open_set import OpenSetDecisionEngine, OpenSetScores, OpenSetThresholds
from models.prototype_registry import PrototypeRegistry


def _engine() -> OpenSetDecisionEngine:
    registry = PrototypeRegistry(embedding_dim=3)
    registry.add_class("new_sign", [[1.0, 0.0, 0.0]])
    return OpenSetDecisionEngine(
        class_mapping={"base_a": 0, "base_b": 1},
        registry=registry,
        thresholds=OpenSetThresholds(base_confidence=0.8, prototype_similarity=0.9),
    )


def test_scores_include_cosine_softmax_and_nearest_prototype_distance() -> None:
    scores = _engine().score([0.7, 0.3], [1.0, 0.0, 0.0])

    assert scores.base_label == "base_a"
    assert scores.base_confidence == pytest.approx(0.7)
    assert scores.prototype_label == "new_sign"
    assert scores.prototype_similarity == pytest.approx(1.0)
    assert scores.prototype_l2_distance == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("scores", "kind", "label"),
    [
        (OpenSetScores("base_a", 0.9, "new_sign", 0.5, 1.0), "base", "base_a"),
        (
            OpenSetScores("base_a", 0.5, "new_sign", 0.95, 0.2),
            "incremental",
            "new_sign",
        ),
        (OpenSetScores("base_a", 0.5, "new_sign", 0.5, 1.0), "unknown", None),
        (OpenSetScores("base_a", 0.9, "new_sign", 0.95, 0.2), "unknown", None),
    ],
)
def test_three_way_decision_policy(
    scores: OpenSetScores, kind: str, label: str | None
) -> None:
    decision = _engine().decide(scores)
    assert decision.kind == kind
    assert decision.label == label


def test_thresholds_have_no_defaults_and_validate_ranges() -> None:
    with pytest.raises(TypeError):
        OpenSetThresholds()  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        OpenSetThresholds(base_confidence=1.1, prototype_similarity=0.5)


def test_scoring_rejects_invalid_probability_vectors() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        _engine().score(np.asarray([0.4, 0.4]), [1.0, 0.0, 0.0])
