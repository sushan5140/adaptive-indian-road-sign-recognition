"""Few-shot registration and threshold-explicit three-way inference decisions."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from torch import Tensor

from inference.frozen_embedding import FrozenEmbeddingPipeline
from models.prototype_registry import PrototypeRegistry

FloatArray = npt.NDArray[np.float32]
DecisionKind = Literal["base", "incremental", "unknown"]


@dataclass(frozen=True, slots=True)
class OpenSetThresholds:
    """Validation-calibrated thresholds; no defaults are intentionally provided."""

    base_confidence: float
    prototype_similarity: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.base_confidence <= 1.0:
            raise ValueError("base_confidence must be in [0, 1]")
        if not -1.0 <= self.prototype_similarity <= 1.0:
            raise ValueError("prototype_similarity must be in [-1, 1]")


@dataclass(frozen=True, slots=True)
class OpenSetScores:
    """Base softmax and nearest-prototype scores for one input."""

    base_label: str
    base_confidence: float
    prototype_label: str | None
    prototype_similarity: float | None
    prototype_l2_distance: float | None


@dataclass(frozen=True, slots=True)
class OpenSetDecision:
    """One conservative base, incremental, or unknown decision."""

    kind: DecisionKind
    label: str | None
    scores: OpenSetScores
    reason: str


class IncrementalClassRegistrar:
    """Register N-shot image prototypes without changing the frozen base model."""

    def __init__(
        self,
        pipeline: FrozenEmbeddingPipeline,
        registry: PrototypeRegistry,
    ) -> None:
        if registry.embedding_dim not in {None, pipeline.identity.embedding_dim}:
            raise ValueError("Registry and frozen embedding dimensions differ")
        self.pipeline = pipeline
        self.registry = registry

    def register_paths(
        self,
        label: str,
        paths: Sequence[str | Path],
        *,
        metadata: Mapping[str, Any] | None = None,
        overwrite: bool = False,
        batch_size: int = 32,
    ) -> FloatArray:
        """Embed one or more reference paths and register their mean prototype."""
        embeddings = self.pipeline.extract_paths(paths, batch_size=batch_size)
        return self._register(label, embeddings, metadata=metadata, overwrite=overwrite)

    def register_preprocessed(
        self,
        label: str,
        images: Tensor,
        *,
        metadata: Mapping[str, Any] | None = None,
        overwrite: bool = False,
        batch_size: int = 32,
    ) -> FloatArray:
        """Embed preprocessed reference tensors and register their mean prototype."""
        embeddings = self.pipeline.infer_preprocessed(
            images, batch_size=batch_size
        ).embeddings
        return self._register(label, embeddings, metadata=metadata, overwrite=overwrite)

    def _register(
        self,
        label: str,
        embeddings: FloatArray,
        *,
        metadata: Mapping[str, Any] | None,
        overwrite: bool,
    ) -> FloatArray:
        if label in self.pipeline.identity.class_mapping:
            raise ValueError("Incremental label collides with a frozen base class")
        before = self.pipeline.model_state_sha256()
        values = dict(metadata or {})
        values.setdefault("shot_count", int(embeddings.shape[0]))
        values.setdefault("checkpoint_sha256", self.pipeline.identity.sha256)
        prototype = self.registry.add_class(
            label, embeddings, metadata=values, overwrite=overwrite
        )
        if self.pipeline.model_state_sha256() != before:
            raise RuntimeError("Base model state changed during registration")
        return prototype


class OpenSetDecisionEngine:
    """Score and classify inputs with explicit calibration-dependent thresholds.

    If both the base and incremental paths exceed their thresholds, the input is
    conservatively rejected as ambiguous. A future calibration protocol may add a
    separately validated conflict rule; the two raw scores are not treated as
    automatically interchangeable.
    """

    def __init__(
        self,
        *,
        class_mapping: Mapping[str, int],
        registry: PrototypeRegistry,
        thresholds: OpenSetThresholds,
    ) -> None:
        ordered = sorted(class_mapping.items(), key=lambda item: item[1])
        if [index for _, index in ordered] != list(range(len(ordered))):
            raise ValueError("class_mapping must be contiguous from zero")
        self.index_to_label = tuple(label for label, _ in ordered)
        self.registry = registry
        self.thresholds = thresholds

    def score(
        self,
        base_probabilities: npt.ArrayLike,
        embedding: npt.ArrayLike,
    ) -> OpenSetScores:
        """Calculate maximum base softmax and nearest-prototype scores."""
        probabilities = np.asarray(base_probabilities, dtype=np.float32)
        if probabilities.shape != (len(self.index_to_label),):
            raise ValueError("base_probabilities has an incompatible shape")
        if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
            raise ValueError("base_probabilities must be finite and non-negative")
        if not np.isclose(float(probabilities.sum()), 1.0, atol=1e-5):
            raise ValueError("base_probabilities must sum to one")
        base_index = int(np.argmax(probabilities))
        matches = self.registry.search(embedding, top_k=1)
        if not matches:
            return OpenSetScores(
                base_label=self.index_to_label[base_index],
                base_confidence=float(probabilities[base_index]),
                prototype_label=None,
                prototype_similarity=None,
                prototype_l2_distance=None,
            )
        match = matches[0]
        distance = math.sqrt(max(0.0, 2.0 - 2.0 * match.similarity))
        return OpenSetScores(
            base_label=self.index_to_label[base_index],
            base_confidence=float(probabilities[base_index]),
            prototype_label=match.label,
            prototype_similarity=match.similarity,
            prototype_l2_distance=distance,
        )

    def decide(self, scores: OpenSetScores) -> OpenSetDecision:
        """Apply the conservative three-way policy to already measured scores."""
        base_pass = scores.base_confidence >= self.thresholds.base_confidence
        prototype_pass = bool(
            scores.prototype_similarity is not None
            and scores.prototype_similarity >= self.thresholds.prototype_similarity
        )
        if base_pass and prototype_pass:
            return OpenSetDecision(
                kind="unknown",
                label=None,
                scores=scores,
                reason="ambiguous_base_and_incremental_scores",
            )
        if prototype_pass:
            return OpenSetDecision(
                kind="incremental",
                label=scores.prototype_label,
                scores=scores,
                reason="prototype_threshold_passed",
            )
        if base_pass:
            return OpenSetDecision(
                kind="base",
                label=scores.base_label,
                scores=scores,
                reason="base_confidence_threshold_passed",
            )
        return OpenSetDecision(
            kind="unknown",
            label=None,
            scores=scores,
            reason="all_thresholds_failed",
        )
