"""Unit tests for few-shot incremental registration."""

from pathlib import Path

import numpy as np
import pytest

from inference.registration import (
    IncrementalRegistrar,
    RegistrationError,
    RegistrationPolicy,
    measure_coherence,
)
from models.prototype_registry import PrototypeRegistry

EMBEDDING_DIM = 8


def _coherent(count: int, *, axis: int = 0, spread: float = 0.05) -> np.ndarray:
    """Build reference embeddings that all point close to one basis direction."""
    generator = np.random.default_rng(20260903 + axis)
    base = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    base[axis] = 1.0
    jitter = generator.normal(scale=spread, size=(count, EMBEDDING_DIM))
    return (base + jitter).astype(np.float32)


def _incoherent(count: int) -> np.ndarray:
    """Build reference embeddings pointing along mutually orthogonal axes."""
    references = np.zeros((count, EMBEDDING_DIM), dtype=np.float32)
    for index in range(count):
        references[index, index % EMBEDDING_DIM] = 1.0
    return references


def _registrar(**kwargs: object) -> IncrementalRegistrar:
    return IncrementalRegistrar(PrototypeRegistry(embedding_dim=EMBEDDING_DIM), **kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------
def test_policy_defaults_require_a_few_shot_set() -> None:
    policy = RegistrationPolicy()

    assert policy.min_references >= 1
    assert policy.max_references >= policy.min_references


@pytest.mark.parametrize(
    "overrides",
    [
        {"min_references": 0},
        {"min_references": 5, "max_references": 2},
        {"min_coherence": 1.5},
        {"min_coherence": -2.0},
    ],
)
def test_policy_rejects_contradictory_settings(overrides: dict[str, object]) -> None:
    with pytest.raises(RegistrationError):
        RegistrationPolicy(**overrides)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Coherence
# ---------------------------------------------------------------------------
def test_single_reference_is_trivially_coherent() -> None:
    coherence = measure_coherence(np.eye(EMBEDDING_DIM, dtype=np.float32)[:1])

    assert coherence.mean_pairwise_similarity == 1.0
    assert coherence.reference_count == 1


def test_identical_references_are_perfectly_coherent() -> None:
    row = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    row[0] = 1.0
    coherence = measure_coherence(np.stack([row, row, row]))

    assert coherence.mean_pairwise_similarity == pytest.approx(1.0)
    assert coherence.minimum_pairwise_similarity == pytest.approx(1.0)


def test_orthogonal_references_score_zero_coherence() -> None:
    coherence = measure_coherence(np.eye(EMBEDDING_DIM, dtype=np.float32)[:4])

    assert coherence.mean_pairwise_similarity == pytest.approx(0.0, abs=1e-6)
    assert coherence.reference_count == 4


def test_coherence_rejects_malformed_input() -> None:
    with pytest.raises(RegistrationError):
        measure_coherence(np.zeros((0, EMBEDDING_DIM), dtype=np.float32))


# ---------------------------------------------------------------------------
# register_embeddings
# ---------------------------------------------------------------------------
def test_registering_a_new_sign_stores_a_prototype() -> None:
    registrar = _registrar()

    result = registrar.register_embeddings(
        "school_ahead", _coherent(5), metadata={"operator": "intern"}
    )

    assert result.label == "school_ahead"
    assert result.reference_count == 5
    assert result.embedding_dim == EMBEDDING_DIM
    assert result.registry_size == 1
    assert result.overwritten is False
    assert result.registry_path is None
    assert "school_ahead" in registrar.registry


def test_registration_records_measured_provenance() -> None:
    registrar = _registrar()

    result = registrar.register_embeddings("school_ahead", _coherent(4))

    assert result.metadata["registration_method"] == "frozen_backbone_prototype"
    assert result.metadata["reference_count"] == 4
    assert result.metadata["mean_pairwise_similarity"] > 0.9
    assert result.registered_at.endswith("+00:00")


def test_registration_does_not_change_the_base_classifier() -> None:
    # The registry is the only mutated object; this test documents that the
    # registrar has no access to a model at all.
    registrar = _registrar()
    registrar.register_embeddings("school_ahead", _coherent(4))

    assert not hasattr(registrar, "model")
    assert registrar.registry.labels == ("school_ahead",)


def test_too_few_references_are_rejected() -> None:
    registrar = _registrar()

    with pytest.raises(RegistrationError, match="at least 3"):
        registrar.register_embeddings("school_ahead", _coherent(2))


def test_too_many_references_are_rejected() -> None:
    registrar = _registrar(policy=RegistrationPolicy(max_references=4))

    with pytest.raises(RegistrationError, match="at most 4"):
        registrar.register_embeddings("school_ahead", _coherent(5))


def test_one_shot_registration_is_possible_under_an_explicit_policy() -> None:
    registrar = _registrar(policy=RegistrationPolicy(min_references=1))

    result = registrar.register_embeddings("school_ahead", _coherent(1)[0])

    assert result.reference_count == 1


def test_incoherent_references_are_rejected() -> None:
    registrar = _registrar()

    with pytest.raises(RegistrationError, match="disagree too much"):
        registrar.register_embeddings("mixed_bag", _incoherent(4))


def test_coherence_check_can_be_disabled() -> None:
    registrar = _registrar(policy=RegistrationPolicy(min_coherence=-1.0))

    result = registrar.register_embeddings("mixed_bag", _incoherent(4))

    assert result.coherence.mean_pairwise_similarity == pytest.approx(0.0, abs=1e-6)


def test_duplicate_label_is_rejected_without_overwrite() -> None:
    registrar = _registrar()
    registrar.register_embeddings("school_ahead", _coherent(4))

    with pytest.raises(RegistrationError, match="already registered"):
        registrar.register_embeddings("school_ahead", _coherent(4))


def test_overwrite_replaces_an_existing_prototype() -> None:
    registrar = _registrar()
    registrar.register_embeddings("school_ahead", _coherent(4, axis=0))

    result = registrar.register_embeddings(
        "school_ahead", _coherent(4, axis=3), overwrite=True
    )

    assert result.overwritten is True
    assert len(registrar.registry) == 1
    assert registrar.registry.get_prototype("school_ahead")[3] > 0.9


def test_mismatched_embedding_width_is_rejected() -> None:
    registrar = _registrar()

    with pytest.raises(RegistrationError):
        registrar.register_embeddings("school_ahead", np.ones((4, 3), dtype=np.float32))


def test_degenerate_references_are_rejected() -> None:
    registrar = _registrar()

    with pytest.raises(RegistrationError, match="unusable"):
        registrar.register_embeddings(
            "school_ahead", np.zeros((4, EMBEDDING_DIM), dtype=np.float32)
        )


def test_three_dimensional_input_is_rejected() -> None:
    registrar = _registrar()

    with pytest.raises(RegistrationError, match="one- or two-dimensional"):
        registrar.register_embeddings("school_ahead", np.ones((2, 2, EMBEDDING_DIM)))


# ---------------------------------------------------------------------------
# register_images
# ---------------------------------------------------------------------------
def test_registering_from_images_uses_the_supplied_embedder() -> None:
    registrar = _registrar()
    images = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(4)]

    def embed(batch: object) -> np.ndarray:
        return _coherent(len(batch))  # type: ignore[arg-type]

    result = registrar.register_images("school_ahead", images, embed)

    assert result.reference_count == 4
    assert result.metadata["source"] == "reference_images"


def test_registering_from_images_rejects_an_empty_set() -> None:
    registrar = _registrar()

    with pytest.raises(RegistrationError, match="No reference images"):
        registrar.register_images("school_ahead", [], lambda batch: _coherent(1))


def test_embedder_returning_the_wrong_row_count_is_rejected() -> None:
    registrar = _registrar()
    images = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(4)]

    with pytest.raises(RegistrationError, match="one embedding row per image"):
        registrar.register_images("school_ahead", images, lambda batch: _coherent(2))


def test_embedder_failure_is_reported_clearly() -> None:
    registrar = _registrar()
    images = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(4)]

    def failing(batch: object) -> np.ndarray:
        raise RuntimeError("backbone exploded")

    with pytest.raises(RegistrationError, match="Could not embed"):
        registrar.register_images("school_ahead", images, failing)


# ---------------------------------------------------------------------------
# Unregistration
# ---------------------------------------------------------------------------
def test_unregistering_removes_the_class() -> None:
    registrar = _registrar()
    registrar.register_embeddings("school_ahead", _coherent(4))

    registrar.unregister("school_ahead")

    assert len(registrar.registry) == 0


def test_unregistering_an_unknown_class_is_rejected() -> None:
    registrar = _registrar()

    with pytest.raises(RegistrationError, match="not registered"):
        registrar.unregister("never_seen")


# ---------------------------------------------------------------------------
# Persistence and dataset protection
# ---------------------------------------------------------------------------
def test_registration_persists_and_reloads(tmp_path: Path) -> None:
    path = tmp_path / "artifacts" / "registry.npz"
    registrar = _registrar(registry_path=path)

    result = registrar.register_embeddings(
        "school_ahead", _coherent(4), metadata={"city": "Bengaluru"}
    )

    assert result.registry_path == str(path.resolve())
    assert path.is_file()

    reloaded = PrototypeRegistry.load(path)
    assert reloaded.labels == ("school_ahead",)
    assert reloaded.get_metadata("school_ahead")["city"] == "Bengaluru"
    np.testing.assert_allclose(
        reloaded.get_prototype("school_ahead"),
        registrar.registry.get_prototype("school_ahead"),
    )


def test_unregistering_persists_the_removal(tmp_path: Path) -> None:
    path = tmp_path / "registry.npz"
    registrar = _registrar(registry_path=path)
    registrar.register_embeddings("school_ahead", _coherent(4))
    registrar.register_embeddings("road_hump", _coherent(4, axis=2))

    registrar.unregister("school_ahead")

    assert PrototypeRegistry.load(path).labels == ("road_hump",)


def test_persistence_can_be_deferred(tmp_path: Path) -> None:
    path = tmp_path / "registry.npz"
    registrar = _registrar(registry_path=path)

    result = registrar.register_embeddings("school_ahead", _coherent(4), persist=False)

    assert result.registry_path is None
    assert not path.exists()


def test_registry_may_not_be_written_inside_a_protected_dataset_root(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "data" / "raw"
    dataset_root.mkdir(parents=True)
    policy = RegistrationPolicy(protected_roots=(dataset_root,))

    with pytest.raises(RegistrationError, match="protected dataset root"):
        IncrementalRegistrar(
            PrototypeRegistry(embedding_dim=EMBEDDING_DIM),
            registry_path=dataset_root / "registry.npz",
            policy=policy,
        )


def test_registry_outside_a_protected_root_is_allowed(tmp_path: Path) -> None:
    dataset_root = tmp_path / "data" / "raw"
    dataset_root.mkdir(parents=True)
    policy = RegistrationPolicy(protected_roots=(dataset_root,))

    registrar = IncrementalRegistrar(
        PrototypeRegistry(embedding_dim=EMBEDDING_DIM),
        registry_path=tmp_path / "artifacts" / "registry.npz",
        policy=policy,
    )

    assert registrar.registry_path is not None
