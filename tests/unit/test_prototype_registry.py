"""Unit tests for the incremental prototype registry."""

import json
from pathlib import Path

import numpy as np
import pytest

from models.prototype_registry import PrototypeMatch, PrototypeRegistry


def test_normalize_embeddings_handles_vector_and_matrix() -> None:
    vector = PrototypeRegistry.normalize_embeddings([3.0, 4.0])
    matrix = PrototypeRegistry.normalize_embeddings([[3.0, 4.0], [0.0, 2.0]])

    np.testing.assert_allclose(vector, [0.6, 0.8])
    np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), [1.0, 1.0])
    assert vector.dtype == np.float32


@pytest.mark.parametrize(
    "embeddings",
    [[], [0.0, 0.0], [1.0, np.nan], [[[1.0, 2.0]]]],
)
def test_normalize_embeddings_rejects_invalid_values(embeddings: object) -> None:
    with pytest.raises(ValueError):
        PrototypeRegistry.normalize_embeddings(embeddings)


def test_calculate_prototype_normalizes_references_before_averaging() -> None:
    prototype = PrototypeRegistry.calculate_prototype([[10.0, 0.0], [0.0, 2.0]])

    expected = np.asarray([1.0, 1.0], dtype=np.float32) / np.sqrt(2.0)
    np.testing.assert_allclose(prototype, expected)
    np.testing.assert_allclose(np.linalg.norm(prototype), 1.0)


def test_calculate_prototype_rejects_cancelling_references() -> None:
    with pytest.raises(ValueError, match="cancel out"):
        PrototypeRegistry.calculate_prototype([[1.0, 0.0], [-1.0, 0.0]])


def test_add_get_overwrite_and_remove_class() -> None:
    registry = PrototypeRegistry(embedding_dim=2)
    original = registry.add_class("new-stop", [[2.0, 0.0]], metadata={"shots": 1})

    assert len(registry) == 1
    assert "new-stop" in registry
    assert registry.labels == ("new-stop",)
    assert registry.get_metadata("new-stop") == {"shots": 1}
    original[0] = 0.0
    np.testing.assert_allclose(registry.get_prototype("new-stop"), [1.0, 0.0])

    with pytest.raises(ValueError, match="already registered"):
        registry.add_class("new-stop", [[0.0, 1.0]])

    registry.add_class("new-stop", [[0.0, 1.0]], overwrite=True)
    np.testing.assert_allclose(registry.get_prototype("new-stop"), [0.0, 1.0])
    registry.remove_class("new-stop")
    assert len(registry) == 0
    with pytest.raises(KeyError):
        registry.remove_class("new-stop")


def test_add_class_rejects_dimension_mismatch_and_unsafe_metadata() -> None:
    registry = PrototypeRegistry(embedding_dim=3)

    with pytest.raises(ValueError, match="dimension"):
        registry.add_class("wrong-width", [1.0, 0.0])
    with pytest.raises(ValueError, match="JSON-serializable"):
        registry.add_class("bad-metadata", [1.0, 0.0, 0.0], metadata={"x": {1}})
    with pytest.raises(ValueError, match="finite"):
        registry.add_class("non-finite", [1.0, 0.0, 0.0], metadata={"x": np.nan})


@pytest.mark.parametrize("label", ["", "   ", "bad\nlabel", "x" * 257])
def test_add_class_rejects_invalid_labels(label: str) -> None:
    with pytest.raises(ValueError):
        PrototypeRegistry().add_class(label, [1.0, 0.0])


def test_search_orders_cosine_matches_and_applies_threshold() -> None:
    registry = PrototypeRegistry(embedding_dim=2)
    registry.add_class("east", [1.0, 0.0], metadata={"id": 1})
    registry.add_class("north", [0.0, 1.0], metadata={"id": 2})
    registry.add_class("west", [-1.0, 0.0], metadata={"id": 3})

    matches = registry.search([0.9, 0.1], top_k=3, min_similarity=0.0)

    assert [match.label for match in matches] == ["east", "north"]
    assert all(isinstance(match, PrototypeMatch) for match in matches)
    assert matches[0].similarity > matches[1].similarity
    assert matches[0].metadata == {"id": 1}


def test_search_empty_registry_and_validation() -> None:
    registry = PrototypeRegistry(embedding_dim=2)

    assert registry.search([1.0, 0.0]) == []
    with pytest.raises(ValueError, match="top_k"):
        registry.search([1.0, 0.0], top_k=0)
    with pytest.raises(ValueError, match="min_similarity"):
        registry.search([1.0, 0.0], min_similarity=1.1)
    with pytest.raises(ValueError, match="exactly one"):
        registry.search([[1.0, 0.0]])


def test_save_and_load_round_trip_without_pickle(tmp_path: Path) -> None:
    registry = PrototypeRegistry(embedding_dim=3)
    registry.add_class(
        "pedestrian-crossing",
        [[1.0, 2.0, 0.0], [2.0, 1.0, 0.0]],
        metadata={"display_name": "Pedestrian Crossing", "shots": 2},
    )
    registry.add_class(
        "भार-सीमा",
        [[0.0, 0.0, 4.0]],
        metadata={"source": "user registration"},
    )
    path = tmp_path / "nested" / "registry.npz"

    registry.save(path)
    loaded = PrototypeRegistry.load(path)

    assert loaded.embedding_dim == 3
    assert loaded.labels == registry.labels
    for label in registry.labels:
        np.testing.assert_allclose(
            loaded.get_prototype(label), registry.get_prototype(label)
        )
        assert loaded.get_metadata(label) == registry.get_metadata(label)
    with np.load(path, allow_pickle=False) as archive:
        assert archive["labels"].dtype.kind in {"U", "S"}
        assert archive["prototypes"].dtype == np.float32
        json.loads(str(archive["metadata_json"].item()))


def test_empty_registry_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "empty.npz"
    PrototypeRegistry(embedding_dim=8).save(path)

    loaded = PrototypeRegistry.load(path)

    assert loaded.embedding_dim == 8
    assert len(loaded) == 0


def test_load_rejects_missing_arrays_and_object_labels(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.npz"
    np.savez(missing_path, labels=np.asarray(["x"]))
    with pytest.raises(ValueError, match="Could not load"):
        PrototypeRegistry.load(missing_path)

    object_path = tmp_path / "object.npz"
    np.savez_compressed(
        object_path,
        schema_version=np.asarray(1),
        embedding_dim=np.asarray(2),
        labels=np.asarray(["x"], dtype=object),
        prototypes=np.asarray([[1.0, 0.0]], dtype=np.float32),
        metadata_json=np.asarray("[{}]"),
    )
    with pytest.raises(ValueError, match="Could not load"):
        PrototypeRegistry.load(object_path)
