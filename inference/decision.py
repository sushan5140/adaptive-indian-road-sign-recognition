"""Open-set decision policy combining base-class confidence and prototypes.

This module answers the question the base classifier structurally cannot: is
this image one of the classes we trained on, one of the classes an operator
registered afterwards from a few reference photographs, or a sign we have never
been told about?

The policy is deliberately dependency-light. It consumes already-computed
probabilities and prototype matches, so it is unit-testable without torch, a
checkpoint, or a dataset, and the same code path serves batch evaluation, the
API, and the UI.

Thresholds default to the placeholder values in ``configs/config.yaml``. They
are *not* calibrated. :class:`OpenSetThresholds` carries an explicit
``calibrated`` flag so that every decision can report whether the numbers behind
it were measured or merely assumed.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from models.prototype_registry import PrototypeMatch

#: Tolerance applied when checking that a probability vector sums to one.
PROBABILITY_SUM_TOLERANCE: float = 1e-3


class OpenSetDecisionError(ValueError):
    """Raised when decision inputs or thresholds are invalid."""


class Verdict(StrEnum):
    """The three outcomes the system is allowed to return."""

    BASE_CLASS = "base_class"
    REGISTERED_CLASS = "registered_class"
    UNKNOWN = "unknown"


class Strategy(StrEnum):
    """How base-classifier and prototype evidence are arbitrated."""

    #: Accept a confident base class first; consult the registry only when the
    #: base classifier is not confident enough. This matches the architecture
    #: described in the project README.
    CLASSIFIER_FIRST = "classifier_first"

    #: Accept a qualifying prototype match first. Useful when an operator has
    #: deliberately registered a sign the base model is known to misread, since
    #: a confident-but-wrong base prediction would otherwise win.
    PROTOTYPE_PRIORITY = "prototype_priority"

    #: Reject as unknown whenever both sources qualify, instead of ranking one
    #: above the other. The two raw scores are not calibrated against each
    #: other, so a conflict is treated as evidence of ambiguity rather than as
    #: something to arbitrate. This is the most defensible policy to report
    #: before calibration, and is the recommended setting for experiments.
    CONSERVATIVE = "conservative"


@dataclass(frozen=True, slots=True)
class OpenSetThresholds:
    """Decision thresholds and the arbitration strategy.

    Args:
        base_confidence_threshold: Minimum maximum-softmax probability required
            to accept a base class.
        prototype_similarity_threshold: Minimum cosine similarity required to
            accept a registered incremental class.
        prototype_margin: Minimum amount by which the best prototype must beat
            the runner-up. Zero disables the margin test. A registry holding one
            class has no runner-up, and the test is skipped.
        strategy: Arbitration order, see :class:`Strategy`.
        unknown_label: Label returned when no evidence qualifies.
        calibrated: Whether these values were measured on held-out data. Defaults
            to ``False`` so that uncalibrated placeholders can never be silently
            reported as tuned.
        calibration_reference: Free-form provenance, for example the run
            identifier of the calibration that produced these values.
    """

    base_confidence_threshold: float = 0.70
    prototype_similarity_threshold: float = 0.75
    prototype_margin: float = 0.0
    strategy: Strategy = Strategy.CLASSIFIER_FIRST
    unknown_label: str = "unknown"
    calibrated: bool = False
    calibration_reference: str | None = None

    def __post_init__(self) -> None:
        """Validate ranges eagerly so a bad config fails at construction."""
        if not 0.0 <= self.base_confidence_threshold <= 1.0:
            raise OpenSetDecisionError(
                "base_confidence_threshold must be in the range [0, 1]"
            )
        if not -1.0 <= self.prototype_similarity_threshold <= 1.0:
            raise OpenSetDecisionError(
                "prototype_similarity_threshold must be in the range [-1, 1]"
            )
        if not 0.0 <= self.prototype_margin <= 2.0:
            raise OpenSetDecisionError("prototype_margin must be in the range [0, 2]")
        if not self.unknown_label.strip():
            raise OpenSetDecisionError("unknown_label must not be empty")

    @classmethod
    def from_config(cls, section: Mapping[str, Any]) -> OpenSetThresholds:
        """Build thresholds from an ``open_set`` configuration section.

        Unknown keys are rejected rather than ignored, so a typo in the YAML
        cannot silently leave a threshold at its default.

        Args:
            section: The ``open_set`` mapping from the project configuration.

        Returns:
            Validated thresholds.

        Raises:
            OpenSetDecisionError: If a key is unsupported or a value has the
                wrong type.
        """
        supported = {
            "base_confidence_threshold",
            "prototype_similarity_threshold",
            "prototype_margin",
            "strategy",
            "unknown_label",
            "calibrated",
            "calibration_reference",
            # Present in configs/config.yaml but consumed elsewhere.
            "prototype_registry_path",
        }
        unsupported = sorted(set(section).difference(supported))
        if unsupported:
            raise OpenSetDecisionError(
                f"Unsupported open_set configuration keys: {unsupported}"
            )

        strategy_value = section.get("strategy", Strategy.CLASSIFIER_FIRST.value)
        try:
            strategy = Strategy(str(strategy_value))
        except ValueError as error:
            allowed = sorted(item.value for item in Strategy)
            raise OpenSetDecisionError(
                f"Unsupported open_set strategy {strategy_value!r}; expected {allowed}"
            ) from error

        reference = section.get("calibration_reference")
        if reference is not None and not isinstance(reference, str):
            raise OpenSetDecisionError("calibration_reference must be a string or null")

        return cls(
            base_confidence_threshold=_as_float(
                section, "base_confidence_threshold", 0.70
            ),
            prototype_similarity_threshold=_as_float(
                section, "prototype_similarity_threshold", 0.75
            ),
            prototype_margin=_as_float(section, "prototype_margin", 0.0),
            strategy=strategy,
            unknown_label=str(section.get("unknown_label", "unknown")),
            calibrated=bool(section.get("calibrated", False)),
            calibration_reference=reference,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view of these thresholds."""
        return {
            "base_confidence_threshold": self.base_confidence_threshold,
            "prototype_similarity_threshold": self.prototype_similarity_threshold,
            "prototype_margin": self.prototype_margin,
            "strategy": self.strategy.value,
            "unknown_label": self.unknown_label,
            "calibrated": self.calibrated,
            "calibration_reference": self.calibration_reference,
        }


@dataclass(frozen=True, slots=True)
class BaseEvidence:
    """What the closed-set classifier reported.

    Attributes:
        label: Highest-probability base class, or ``None`` when no base
            classifier was consulted.
        confidence: Maximum softmax probability, ``0.0`` when unavailable.
        ranking: Up to the top few ``(label, probability)`` pairs, best first.
        available: Whether a base classifier contributed to this decision.
    """

    label: str | None
    confidence: float
    ranking: tuple[tuple[str, float], ...] = ()
    available: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view of this evidence."""
        return {
            "label": self.label,
            "confidence": self.confidence,
            "ranking": [
                {"label": label, "probability": probability}
                for label, probability in self.ranking
            ],
            "available": self.available,
        }


@dataclass(frozen=True, slots=True)
class PrototypeEvidence:
    """What the incremental prototype registry reported.

    Attributes:
        label: Best-matching registered label, or ``None`` when the registry is
            empty.
        similarity: Cosine similarity of the best match, ``-1.0`` when none.
        runner_up_label: Second-best registered label, if any.
        runner_up_similarity: Cosine similarity of the runner-up, if any.
        margin: ``similarity - runner_up_similarity``. ``None`` when the registry
            holds fewer than two classes, in which case the margin test does not
            apply.
        registry_size: Number of registered incremental classes consulted.
        l2_distance: Euclidean distance to the best prototype. Both vectors are
            unit-norm, so this is ``sqrt(2 - 2 * similarity)``; it is reported
            because open-set literature more often thresholds a distance than a
            similarity. ``None`` when nothing is registered.
    """

    label: str | None
    similarity: float
    runner_up_label: str | None = None
    runner_up_similarity: float | None = None
    margin: float | None = None
    registry_size: int = 0

    @property
    def l2_distance(self) -> float | None:
        """Euclidean distance to the best prototype, both vectors being unit-norm."""
        if self.label is None:
            return None
        return math.sqrt(max(0.0, 2.0 - 2.0 * self.similarity))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view of this evidence."""
        return {
            "label": self.label,
            "similarity": self.similarity,
            "runner_up_label": self.runner_up_label,
            "runner_up_similarity": self.runner_up_similarity,
            "margin": self.margin,
            "registry_size": self.registry_size,
            "l2_distance": self.l2_distance,
        }


@dataclass(frozen=True, slots=True)
class OpenSetDecision:
    """The final verdict together with the evidence that produced it.

    Attributes:
        verdict: Which of the three outcomes was selected.
        label: The returned label. Equals ``thresholds.unknown_label`` when the
            verdict is :attr:`Verdict.UNKNOWN`.
        score: The score behind the verdict — maximum softmax probability for a
            base class, cosine similarity for a registered class, and the higher
            of the two rejected scores for ``unknown``.
        reason: Human-readable explanation naming the rule that fired. Written
            for an operator reading a UI, and for the project report.
        base: Closed-set evidence.
        prototype: Incremental-registry evidence.
        thresholds: The thresholds this decision was made under.
    """

    verdict: Verdict
    label: str
    score: float
    reason: str
    base: BaseEvidence
    prototype: PrototypeEvidence
    thresholds: OpenSetThresholds = field(default_factory=OpenSetThresholds)

    @property
    def is_unknown(self) -> bool:
        """Whether the system declined to name this sign."""
        return self.verdict is Verdict.UNKNOWN

    @property
    def uses_calibrated_thresholds(self) -> bool:
        """Whether the thresholds behind this decision were measured."""
        return self.thresholds.calibrated

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view of the whole decision."""
        return {
            "verdict": self.verdict.value,
            "label": self.label,
            "score": self.score,
            "reason": self.reason,
            "base": self.base.to_dict(),
            "prototype": self.prototype.to_dict(),
            "thresholds": self.thresholds.to_dict(),
        }


def softmax(logits: Sequence[float]) -> tuple[float, ...]:
    """Convert raw logits to probabilities using a numerically stable softmax.

    Args:
        logits: Finite, non-empty raw classifier outputs.

    Returns:
        Probabilities in the same order, summing to one.

    Raises:
        OpenSetDecisionError: If ``logits`` is empty or contains a non-finite
            value.
    """
    values = [float(value) for value in logits]
    if not values:
        raise OpenSetDecisionError("logits must not be empty")
    if any(not math.isfinite(value) for value in values):
        raise OpenSetDecisionError("logits must contain only finite values")
    largest = max(values)
    exponentials = [math.exp(value - largest) for value in values]
    total = sum(exponentials)
    return tuple(value / total for value in exponentials)


def build_base_evidence(
    probabilities: Sequence[float],
    class_names: Sequence[str],
    *,
    top_k: int = 3,
) -> BaseEvidence:
    """Summarize a base-classifier probability vector.

    Args:
        probabilities: One probability per base class, summing to one.
        class_names: Base class labels in the classifier's index order.
        top_k: How many ranked entries to retain for reporting.

    Returns:
        The corresponding :class:`BaseEvidence`.

    Raises:
        OpenSetDecisionError: If the inputs disagree in length, are empty, or
            the probabilities are not a valid distribution.
    """
    if top_k < 1:
        raise OpenSetDecisionError("top_k must be positive")
    values = [float(value) for value in probabilities]
    labels = [str(name) for name in class_names]
    if len(values) != len(labels):
        raise OpenSetDecisionError(
            f"Received {len(values)} probabilities for {len(labels)} class names"
        )
    if not values:
        raise OpenSetDecisionError("probabilities must not be empty")
    if any(not math.isfinite(value) for value in values):
        raise OpenSetDecisionError("probabilities must contain only finite values")
    if any(value < 0.0 or value > 1.0 for value in values):
        raise OpenSetDecisionError("probabilities must lie in the range [0, 1]")
    if abs(sum(values) - 1.0) > PROBABILITY_SUM_TOLERANCE:
        raise OpenSetDecisionError(
            f"probabilities must sum to 1.0 within {PROBABILITY_SUM_TOLERANCE}, "
            f"got {sum(values)}"
        )

    order = sorted(range(len(values)), key=lambda index: (-values[index], index))
    ranking = tuple((labels[index], values[index]) for index in order[:top_k])
    best = order[0]
    return BaseEvidence(
        label=labels[best],
        confidence=values[best],
        ranking=ranking,
        available=True,
    )


def build_prototype_evidence(matches: Sequence[PrototypeMatch]) -> PrototypeEvidence:
    """Summarize the ranked prototype matches returned by the registry.

    Args:
        matches: Matches ordered best-first, as produced by
            :meth:`models.prototype_registry.PrototypeRegistry.search`. An empty
            sequence means nothing has been registered yet, which is a normal
            open-set state rather than an error.

    Returns:
        The corresponding :class:`PrototypeEvidence`.
    """
    if not matches:
        return PrototypeEvidence(label=None, similarity=-1.0, registry_size=0)

    best = matches[0]
    if len(matches) == 1:
        return PrototypeEvidence(
            label=best.label,
            similarity=float(best.similarity),
            registry_size=1,
        )
    runner_up = matches[1]
    return PrototypeEvidence(
        label=best.label,
        similarity=float(best.similarity),
        runner_up_label=runner_up.label,
        runner_up_similarity=float(runner_up.similarity),
        margin=float(best.similarity) - float(runner_up.similarity),
        registry_size=len(matches),
    )


def decide(
    *,
    base: BaseEvidence,
    prototype: PrototypeEvidence,
    thresholds: OpenSetThresholds | None = None,
) -> OpenSetDecision:
    """Arbitrate between closed-set and incremental evidence.

    ``classifier_first`` (the default) accepts a sufficiently confident base
    class, then falls back to the registry, then to ``unknown``.
    ``prototype_priority`` reverses the first two steps.

    A registered class is accepted only when its cosine similarity clears
    ``prototype_similarity_threshold`` *and*, when a runner-up exists, beats it
    by at least ``prototype_margin``. The margin guards against returning a
    near-coin-flip between two visually similar registered signs.

    Args:
        base: Closed-set evidence, from :func:`build_base_evidence`.
        prototype: Registry evidence, from :func:`build_prototype_evidence`.
        thresholds: Decision thresholds. Uncalibrated defaults when omitted.

    Returns:
        The decision, carrying both the verdict and the evidence behind it.
    """
    active = thresholds or OpenSetThresholds()
    base_accepted = base.available and base.confidence >= (
        active.base_confidence_threshold
    )
    prototype_accepted, prototype_rejection = _evaluate_prototype(prototype, active)

    if (
        active.strategy is Strategy.CONSERVATIVE
        and base_accepted
        and prototype_accepted
    ):
        return _ambiguous_decision(base, prototype, active)

    if active.strategy is Strategy.PROTOTYPE_PRIORITY:
        order: tuple[str, ...] = ("prototype", "base")
    else:
        order = ("base", "prototype")

    for source in order:
        if source == "base" and base_accepted:
            return _base_decision(base, prototype, active)
        if source == "prototype" and prototype_accepted:
            return _prototype_decision(base, prototype, active)

    return _unknown_decision(
        base,
        prototype,
        active,
        base_accepted=base_accepted,
        prototype_rejection=prototype_rejection,
    )


def _ambiguous_decision(
    base: BaseEvidence,
    prototype: PrototypeEvidence,
    thresholds: OpenSetThresholds,
) -> OpenSetDecision:
    """Reject an input for which both evidence sources qualify."""
    return OpenSetDecision(
        verdict=Verdict.UNKNOWN,
        label=thresholds.unknown_label,
        score=max(base.confidence, prototype.similarity),
        reason=(
            f"ambiguous: base class {base.label!r} ({base.confidence:.4f}) and "
            f"registered class {prototype.label!r} ({prototype.similarity:.4f}) "
            f"both cleared their thresholds; the two scores are not calibrated "
            f"against each other, so neither is preferred"
        ),
        base=base,
        prototype=prototype,
        thresholds=thresholds,
    )


def _evaluate_prototype(
    prototype: PrototypeEvidence, thresholds: OpenSetThresholds
) -> tuple[bool, str]:
    """Test the registry evidence, returning acceptance and a rejection reason."""
    if prototype.label is None:
        return False, "no incremental class is registered"
    if prototype.similarity < thresholds.prototype_similarity_threshold:
        return False, (
            f"best prototype similarity {prototype.similarity:.4f} is below the "
            f"threshold {thresholds.prototype_similarity_threshold:.4f}"
        )
    if (
        thresholds.prototype_margin > 0.0
        and prototype.margin is not None
        and prototype.margin < thresholds.prototype_margin
    ):
        return False, (
            f"best prototype beat the runner-up by only {prototype.margin:.4f}, "
            f"below the required margin {thresholds.prototype_margin:.4f}"
        )
    return True, ""


def _base_decision(
    base: BaseEvidence,
    prototype: PrototypeEvidence,
    thresholds: OpenSetThresholds,
) -> OpenSetDecision:
    """Build a decision that accepts the base classifier's answer."""
    assert base.label is not None  # guaranteed by base.available
    return OpenSetDecision(
        verdict=Verdict.BASE_CLASS,
        label=base.label,
        score=base.confidence,
        reason=(
            f"base classifier confidence {base.confidence:.4f} reached the "
            f"threshold {thresholds.base_confidence_threshold:.4f}"
        ),
        base=base,
        prototype=prototype,
        thresholds=thresholds,
    )


def _prototype_decision(
    base: BaseEvidence,
    prototype: PrototypeEvidence,
    thresholds: OpenSetThresholds,
) -> OpenSetDecision:
    """Build a decision that accepts a registered incremental class."""
    assert prototype.label is not None  # guaranteed by _evaluate_prototype
    margin_note = (
        "; no runner-up to compare against"
        if prototype.margin is None
        else f"; margin over runner-up {prototype.margin:.4f}"
    )
    return OpenSetDecision(
        verdict=Verdict.REGISTERED_CLASS,
        label=prototype.label,
        score=prototype.similarity,
        reason=(
            f"registered prototype similarity {prototype.similarity:.4f} reached "
            f"the threshold {thresholds.prototype_similarity_threshold:.4f}"
            f"{margin_note}"
        ),
        base=base,
        prototype=prototype,
        thresholds=thresholds,
    )


def _unknown_decision(
    base: BaseEvidence,
    prototype: PrototypeEvidence,
    thresholds: OpenSetThresholds,
    *,
    base_accepted: bool,
    prototype_rejection: str,
) -> OpenSetDecision:
    """Build a decision that declines to name the sign."""
    reasons: list[str] = []
    if not base.available:
        reasons.append("no base classifier was consulted")
    elif not base_accepted:
        reasons.append(
            f"base classifier confidence {base.confidence:.4f} is below the "
            f"threshold {thresholds.base_confidence_threshold:.4f}"
        )
    if prototype_rejection:
        reasons.append(prototype_rejection)

    score = max(base.confidence if base.available else 0.0, prototype.similarity)
    return OpenSetDecision(
        verdict=Verdict.UNKNOWN,
        label=thresholds.unknown_label,
        score=score,
        reason="; ".join(reasons) if reasons else "no evidence qualified",
        base=base,
        prototype=prototype,
        thresholds=thresholds,
    )


def decide_from_scores(
    probabilities: Sequence[float],
    class_names: Sequence[str],
    matches: Sequence[PrototypeMatch],
    *,
    thresholds: OpenSetThresholds | None = None,
    top_k: int = 3,
) -> OpenSetDecision:
    """Convenience wrapper: summarize both evidence sources and arbitrate.

    Args:
        probabilities: Base-class probabilities. Pass an empty sequence to run
            registry-only recognition with no base classifier.
        class_names: Base class labels in classifier index order.
        matches: Ranked prototype matches, best first.
        thresholds: Decision thresholds. Uncalibrated defaults when omitted.
        top_k: How many ranked base entries to retain for reporting.

    Returns:
        The open-set decision.

    Raises:
        OpenSetDecisionError: If the base inputs are inconsistent.
    """
    if len(probabilities) == 0 and len(class_names) == 0:
        base = BaseEvidence(label=None, confidence=0.0, ranking=(), available=False)
    else:
        base = build_base_evidence(probabilities, class_names, top_k=top_k)
    return decide(
        base=base,
        prototype=build_prototype_evidence(matches),
        thresholds=thresholds,
    )


def _as_float(section: Mapping[str, Any], key: str, default: float) -> float:
    """Read a numeric configuration value, rejecting booleans and non-numbers."""
    value = section.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OpenSetDecisionError(f"open_set.{key} must be a number")
    return float(value)
