"""Few-shot registration of new sign classes without retraining.

Registering a sign is the whole of "incremental learning" in this project:
reference photographs are embedded by the frozen backbone, the embeddings are
averaged into a unit-norm prototype, and that prototype is stored in the
separate NPZ registry. No gradient is computed, no weight is updated, and the
immutable base dataset is never written to.

The two invariants this module enforces mechanically rather than by convention:

* the registry file may never be written inside a dataset root, and
* the number of reference images must fall inside the configured bounds.

It also measures how coherent the reference set is. A caller who uploads five
photographs of three different signs would otherwise get a meaningless average
of unrelated directions, and the resulting prototype would quietly match
nothing. The coherence floor turns that into an explicit, refusable error.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from models.prototype_registry import FloatArray, PrototypeRegistry

#: Reference sets smaller than this cannot describe a class reliably.
DEFAULT_MIN_REFERENCES: int = 3

#: Upper bound, kept modest because this is few-shot registration, not training.
DEFAULT_MAX_REFERENCES: int = 32

#: Mean pairwise cosine similarity below which a reference set is judged to
#: describe more than one thing. Conservative default; tune per backbone.
DEFAULT_MIN_COHERENCE: float = 0.30


class RegistrationError(ValueError):
    """Raised when a new-class registration request is invalid or unsafe."""


@dataclass(frozen=True, slots=True)
class ReferenceCoherence:
    """Measured agreement among the reference embeddings of one class.

    Attributes:
        mean_pairwise_similarity: Average cosine similarity over every distinct
            reference pair. ``1.0`` when only one reference was supplied.
        minimum_pairwise_similarity: Worst pair, useful for spotting a single
            odd photograph in an otherwise clean set.
        reference_count: Number of reference embeddings measured.
    """

    mean_pairwise_similarity: float
    minimum_pairwise_similarity: float
    reference_count: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view of these diagnostics."""
        return {
            "mean_pairwise_similarity": self.mean_pairwise_similarity,
            "minimum_pairwise_similarity": self.minimum_pairwise_similarity,
            "reference_count": self.reference_count,
        }


@dataclass(frozen=True, slots=True)
class RegistrationResult:
    """Outcome of registering or replacing one incremental class.

    Attributes:
        label: The registered class label.
        reference_count: How many reference embeddings formed the prototype.
        embedding_dim: Width of the stored prototype.
        coherence: Measured agreement among the references.
        registry_path: Where the registry was persisted, or ``None`` when the
            caller asked to keep the change in memory only.
        registry_size: Number of incremental classes after this registration.
        overwritten: Whether an existing label was replaced.
        registered_at: ISO-8601 UTC timestamp.
        metadata: Metadata stored alongside the prototype.
    """

    label: str
    reference_count: int
    embedding_dim: int
    coherence: ReferenceCoherence
    registry_path: str | None
    registry_size: int
    overwritten: bool
    registered_at: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view of the result."""
        return {
            "label": self.label,
            "reference_count": self.reference_count,
            "embedding_dim": self.embedding_dim,
            "coherence": self.coherence.to_dict(),
            "registry_path": self.registry_path,
            "registry_size": self.registry_size,
            "overwritten": self.overwritten,
            "registered_at": self.registered_at,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class RegistrationPolicy:
    """Bounds and safety rules applied to every registration.

    Args:
        min_references: Fewest reference embeddings accepted.
        max_references: Most reference embeddings accepted.
        min_coherence: Mean pairwise cosine similarity floor. Set to ``-1.0`` to
            disable the check.
        protected_roots: Directories the registry must never be written inside,
            normally the immutable dataset roots.
    """

    min_references: int = DEFAULT_MIN_REFERENCES
    max_references: int = DEFAULT_MAX_REFERENCES
    min_coherence: float = DEFAULT_MIN_COHERENCE
    protected_roots: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        """Validate the policy itself before it is used to validate requests."""
        if self.min_references < 1:
            raise RegistrationError("min_references must be at least 1")
        if self.max_references < self.min_references:
            raise RegistrationError("max_references must be >= min_references")
        if not -1.0 <= self.min_coherence <= 1.0:
            raise RegistrationError("min_coherence must be in the range [-1, 1]")

    @classmethod
    def from_config(
        cls,
        section: Mapping[str, Any],
        *,
        project_root: str | Path | None = None,
    ) -> RegistrationPolicy:
        """Build a policy from a ``registration`` configuration section.

        Unknown keys are rejected so that a typo cannot silently leave a bound
        at its default.

        Args:
            section: The ``registration`` mapping from the project configuration.
            project_root: Base directory for resolving relative protected roots.
                Defaults to the current working directory.

        Returns:
            A validated policy.

        Raises:
            RegistrationError: If a key is unsupported or a value is malformed.
        """
        supported = {
            "min_references",
            "max_references",
            "min_coherence",
            "protected_roots",
        }
        unsupported = sorted(set(section).difference(supported))
        if unsupported:
            raise RegistrationError(
                f"Unsupported registration configuration keys: {unsupported}"
            )

        raw_roots = section.get("protected_roots") or ()
        if isinstance(raw_roots, (str, bytes)) or not isinstance(raw_roots, Sequence):
            raise RegistrationError("registration.protected_roots must be a list")
        base = Path(project_root) if project_root is not None else Path.cwd()
        roots = tuple(
            (Path(str(item)) if Path(str(item)).is_absolute() else base / str(item))
            .expanduser()
            .resolve()
            for item in raw_roots
        )

        return cls(
            min_references=_as_int(section, "min_references", DEFAULT_MIN_REFERENCES),
            max_references=_as_int(section, "max_references", DEFAULT_MAX_REFERENCES),
            min_coherence=_as_number(section, "min_coherence", DEFAULT_MIN_COHERENCE),
            protected_roots=roots,
        )


def measure_coherence(normalized: FloatArray) -> ReferenceCoherence:
    """Measure pairwise agreement among unit-norm reference embeddings.

    Args:
        normalized: A ``(n, d)`` array whose rows are L2-normalized.

    Returns:
        The measured coherence. A single reference trivially scores ``1.0``.

    Raises:
        RegistrationError: If ``normalized`` is not a non-empty 2-D array.
    """
    if normalized.ndim != 2 or normalized.shape[0] == 0:
        raise RegistrationError("coherence needs a non-empty two-dimensional array")

    count = int(normalized.shape[0])
    if count == 1:
        return ReferenceCoherence(1.0, 1.0, 1)

    similarities = normalized @ normalized.T
    upper = np.triu_indices(count, k=1)
    pairwise = np.clip(similarities[upper], -1.0, 1.0)
    return ReferenceCoherence(
        mean_pairwise_similarity=float(pairwise.mean()),
        minimum_pairwise_similarity=float(pairwise.min()),
        reference_count=count,
    )


class IncrementalRegistrar:
    """Register and unregister incremental classes against a prototype registry.

    The registrar owns the policy and the persistence path; the registry itself
    owns prototype storage and search. Separating them keeps the registry pure
    NumPy and independently testable.

    Args:
        registry: The prototype registry to mutate.
        registry_path: Where to persist after each change. ``None`` keeps every
            change in memory, which is what the unit tests use.
        policy: Bounds and safety rules. Defaults are conservative.
        base_labels: The frozen base classifier's labels. An incremental class
            may not reuse one of these names, because two independent mechanisms
            would then answer to the same label.
        model_fingerprint: Optional callable returning a digest of the base
            model's state. When supplied it is sampled before and after every
            registration, turning "registration must not change the model" from
            a convention into a checked runtime invariant.

    Raises:
        RegistrationError: If ``registry_path`` resolves inside a protected root.
    """

    def __init__(
        self,
        registry: PrototypeRegistry,
        *,
        registry_path: str | Path | None = None,
        policy: RegistrationPolicy | None = None,
        base_labels: Sequence[str] = (),
        model_fingerprint: Callable[[], str] | None = None,
    ) -> None:
        self.registry = registry
        self.policy = policy or RegistrationPolicy()
        self.base_labels = frozenset(str(label) for label in base_labels)
        self._model_fingerprint = model_fingerprint
        self._registry_path = (
            Path(registry_path).expanduser().resolve()
            if registry_path is not None
            else None
        )
        if self._registry_path is not None:
            self._assert_outside_protected_roots(self._registry_path)

    @property
    def registry_path(self) -> Path | None:
        """Where the registry is persisted, if anywhere."""
        return self._registry_path

    def register_embeddings(
        self,
        label: str,
        embeddings: npt.ArrayLike,
        *,
        metadata: Mapping[str, Any] | None = None,
        overwrite: bool = False,
        persist: bool = True,
    ) -> RegistrationResult:
        """Register a class from precomputed reference embeddings.

        Args:
            label: Stable, human-readable class name.
            embeddings: A ``(n, d)`` array of reference embeddings, or a single
                ``(d,)`` vector when the policy permits one-shot registration.
            metadata: JSON-serializable provenance to store with the prototype.
            overwrite: Whether an already-registered label may be replaced.
            persist: Whether to write the registry to disk afterwards. Ignored
                when no ``registry_path`` was configured.

        Returns:
            The measured :class:`RegistrationResult`.

        Raises:
            RegistrationError: If the reference count is outside policy bounds,
                the references are incoherent, the label already exists without
                ``overwrite``, or the embeddings are malformed.
        """
        self._assert_not_a_base_label(label)
        array = np.asarray(embeddings, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2:
            raise RegistrationError(
                f"reference embeddings must be one- or two-dimensional, got "
                f"{array.ndim} dimensions"
            )

        count = int(array.shape[0])
        if count < self.policy.min_references:
            raise RegistrationError(
                f"Class {label!r} needs at least {self.policy.min_references} "
                f"reference embeddings, received {count}"
            )
        if count > self.policy.max_references:
            raise RegistrationError(
                f"Class {label!r} accepts at most {self.policy.max_references} "
                f"reference embeddings, received {count}"
            )

        try:
            normalized = self.registry.normalize_embeddings(array)
        except ValueError as error:
            raise RegistrationError(
                f"Reference embeddings for {label!r} are unusable: {error}"
            ) from error

        coherence = measure_coherence(normalized)
        if coherence.mean_pairwise_similarity < self.policy.min_coherence:
            raise RegistrationError(
                f"Reference images for {label!r} disagree too much: mean pairwise "
                f"cosine similarity {coherence.mean_pairwise_similarity:.4f} is "
                f"below the required {self.policy.min_coherence:.4f}. They may "
                f"show more than one sign."
            )

        already_registered = label in self.registry
        stored_metadata = self._build_metadata(metadata, coherence, count)
        fingerprint_before = (
            self._model_fingerprint() if self._model_fingerprint is not None else None
        )
        try:
            prototype = self.registry.add_class(
                label,
                normalized,
                metadata=stored_metadata,
                overwrite=overwrite,
            )
        except (ValueError, TypeError) as error:
            raise RegistrationError(
                f"Could not register class {label!r}: {error}"
            ) from error

        if fingerprint_before is not None and self._model_fingerprint is not None:
            if self._model_fingerprint() != fingerprint_before:
                raise RegistrationError(
                    f"Base model state changed while registering {label!r}; "
                    f"registration must never modify the frozen model"
                )

        written_to: str | None = None
        if persist and self._registry_path is not None:
            self._persist()
            written_to = str(self._registry_path)

        return RegistrationResult(
            label=label,
            reference_count=count,
            embedding_dim=int(prototype.shape[0]),
            coherence=coherence,
            registry_path=written_to,
            registry_size=len(self.registry),
            overwritten=already_registered,
            registered_at=stored_metadata["registered_at"],
            metadata=stored_metadata,
        )

    def register_images(
        self,
        label: str,
        images: Sequence[np.ndarray],
        embed: Callable[[Sequence[np.ndarray]], npt.ArrayLike],
        *,
        metadata: Mapping[str, Any] | None = None,
        overwrite: bool = False,
        persist: bool = True,
    ) -> RegistrationResult:
        """Register a class directly from reference images.

        Args:
            label: Stable, human-readable class name.
            images: Decoded reference images.
            embed: Callable turning those images into an ``(n, d)`` embedding
                array. Normally
                :meth:`inference.pipeline.OpenSetRecognizer.embed_images`.
            metadata: JSON-serializable provenance.
            overwrite: Whether an existing label may be replaced.
            persist: Whether to write the registry afterwards.

        Returns:
            The measured :class:`RegistrationResult`.

        Raises:
            RegistrationError: If ``images`` is empty, the embedder returns an
                unusable array, or any registration rule fails.
        """
        if not images:
            raise RegistrationError(f"No reference images supplied for {label!r}")
        try:
            embeddings = np.asarray(embed(images), dtype=np.float32)
        except (ValueError, TypeError, RuntimeError) as error:
            raise RegistrationError(
                f"Could not embed reference images for {label!r}: {error}"
            ) from error
        if embeddings.ndim != 2 or embeddings.shape[0] != len(images):
            raise RegistrationError(
                f"Embedder returned {embeddings.shape} for {len(images)} reference "
                f"images; expected one embedding row per image"
            )
        enriched = dict(metadata or {})
        enriched.setdefault("source", "reference_images")
        return self.register_embeddings(
            label,
            embeddings,
            metadata=enriched,
            overwrite=overwrite,
            persist=persist,
        )

    def unregister(self, label: str, *, persist: bool = True) -> None:
        """Remove an incremental class.

        Args:
            label: The registered label to remove.
            persist: Whether to write the registry afterwards.

        Raises:
            RegistrationError: If ``label`` is not registered.
        """
        try:
            self.registry.remove_class(label)
        except KeyError as error:
            raise RegistrationError(f"Class {label!r} is not registered") from error
        if persist and self._registry_path is not None:
            self._persist()

    def _persist(self) -> None:
        """Write the registry, re-checking the destination every time."""
        assert self._registry_path is not None
        self._assert_outside_protected_roots(self._registry_path)
        try:
            self.registry.save(self._registry_path)
        except OSError as error:
            raise RegistrationError(
                f"Could not persist registry to {self._registry_path}: {error}"
            ) from error

    def _assert_not_a_base_label(self, label: str) -> None:
        """Refuse an incremental label that collides with a frozen base class."""
        if label in self.base_labels:
            raise RegistrationError(
                f"Label {label!r} is already a base class of the frozen "
                f"classifier. Incremental classes must use a distinct name, "
                f"otherwise one label would have two sources of truth."
            )

    def _assert_outside_protected_roots(self, path: Path) -> None:
        """Refuse to write derived prototypes inside an immutable dataset root."""
        for root in self.policy.protected_roots:
            resolved_root = Path(root).expanduser().resolve()
            if path == resolved_root or path.is_relative_to(resolved_root):
                raise RegistrationError(
                    f"Registry path {path} is inside protected dataset root "
                    f"{resolved_root}; incremental data must stay separate from "
                    f"the immutable base dataset"
                )

    @staticmethod
    def _build_metadata(
        metadata: Mapping[str, Any] | None,
        coherence: ReferenceCoherence,
        reference_count: int,
    ) -> dict[str, Any]:
        """Merge caller metadata with measured registration provenance."""
        stored = dict(metadata or {})
        stored["reference_count"] = reference_count
        stored["registered_at"] = datetime.now(UTC).isoformat()
        stored["mean_pairwise_similarity"] = coherence.mean_pairwise_similarity
        stored["minimum_pairwise_similarity"] = coherence.minimum_pairwise_similarity
        stored["registration_method"] = "frozen_backbone_prototype"
        return stored


def _as_int(section: Mapping[str, Any], key: str, default: int) -> int:
    """Read an integer configuration value, rejecting booleans and floats."""
    value = section.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RegistrationError(f"registration.{key} must be an integer")
    return int(value)


def _as_number(section: Mapping[str, Any], key: str, default: float) -> float:
    """Read a numeric configuration value, rejecting booleans and non-numbers."""
    value = section.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RegistrationError(f"registration.{key} must be a number")
    return float(value)
