"""Local regression tests for the measured frozen Baseline V2 checkpoint."""

import csv
from pathlib import Path

import numpy as np
import pytest
import torch

from inference.frozen_embedding import FrozenEmbeddingPipeline
from inference.open_set import IncrementalClassRegistrar
from models.prototype_registry import PrototypeRegistry

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = (
    ROOT
    / "outputs"
    / "checkpoints"
    / "20260901_000220_558663_v2_mobilenetv3_small_100"
    / "best.pt"
)
EXPECTED_SHA256 = "0f990f21c7f844f5611e91f867740b7f980e851426681c69deb2fefadbea8ff4"


def _reference_paths(count: int = 5) -> list[Path]:
    manifest = ROOT / "outputs" / "manifests" / "v2_train.csv"
    dataset_root = ROOT / "data" / "raw" / "indian_traffic_vqa"
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [dataset_root / row["image_path"] for row in rows[:count]]


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="Local V2 checkpoint is ignored")
def test_frozen_v2_embeddings_and_registration_do_not_change_weights(
    tmp_path: Path,
) -> None:
    pipeline = FrozenEmbeddingPipeline.from_checkpoint(
        CHECKPOINT, expected_sha256=EXPECTED_SHA256
    )
    paths = _reference_paths()
    if not all(path.exists() for path in paths):
        pytest.skip("Local V2 training images are unavailable")
    tensors = pipeline.preprocess_paths(paths)
    before = pipeline.model_state_sha256()
    first = pipeline.infer_preprocessed(tensors, batch_size=5)
    repeated = pipeline.infer_preprocessed(tensors, batch_size=5)
    partitioned = pipeline.infer_preprocessed(tensors, batch_size=2)

    assert pipeline.all_parameters_frozen
    assert pipeline.identity.embedding_dim == 1024
    assert first.embeddings.shape == (5, 1024)
    assert np.all(np.isfinite(first.embeddings))
    assert np.allclose(np.linalg.norm(first.embeddings, axis=1), 1.0, atol=1e-6)
    assert np.array_equal(first.embeddings, repeated.embeddings)
    assert np.allclose(first.embeddings, partitioned.embeddings, atol=1e-6)
    assert pipeline.model_state_sha256() == before

    registry = PrototypeRegistry(embedding_dim=1024)
    registrar = IncrementalClassRegistrar(pipeline, registry)
    base_before = first.base_probabilities.copy()
    for shots in (1, 3, 5):
        label = f"temporary_{shots}_shot"
        prototype = registrar.register_preprocessed(label, tensors[:shots])
        assert np.linalg.norm(prototype) == pytest.approx(1.0, abs=1e-6)
        assert registry.get_metadata(label)["shot_count"] == shots
    assert pipeline.model_state_sha256() == before
    base_after = pipeline.infer_preprocessed(tensors, batch_size=5).base_probabilities
    assert np.array_equal(base_before, base_after)

    with pytest.raises(ValueError, match="already registered"):
        registrar.register_preprocessed("temporary_1_shot", tensors[:1])
    registry.remove_class("temporary_1_shot")
    assert "temporary_1_shot" not in registry
    path = tmp_path / "registry.npz"
    registry.save(path)
    loaded = PrototypeRegistry.load(path)
    assert loaded.labels == registry.labels
    for label in registry.labels:
        assert np.array_equal(
            loaded.get_prototype(label), registry.get_prototype(label)
        )
        assert loaded.get_metadata(label) == registry.get_metadata(label)
    assert pipeline.model_state_sha256() == before
