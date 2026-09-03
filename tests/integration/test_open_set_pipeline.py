"""End-to-end open-set recognizer smoke test on an untrained synthetic model.

These tests verify wiring and invariants, not accuracy. The model is randomly
initialized with ``pretrained=False``, so no verdict here says anything about
how well the system recognizes real road signs.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip(
    "torch", reason="PyTorch is required for open-set pipeline testing"
)
pytest.importorskip(
    "torchvision", reason="torchvision is required for open-set pipeline testing"
)
pytest.importorskip("timm", reason="timm is required for open-set pipeline testing")

from inference.decision import OpenSetThresholds, Strategy, Verdict
from inference.pipeline import InferenceError, OpenSetRecognizer
from inference.registration import RegistrationError, RegistrationPolicy
from models.classifier import RoadSignClassifier
from training.checkpoint import CheckpointManager, CheckpointMetadata

BASE_CLASSES = ("give_way", "no_entry", "road_hump")


def _write_checkpoint(directory: Path) -> Path:
    """Save a real checkpoint for a small randomly initialized classifier."""
    model = RoadSignClassifier(num_classes=len(BASE_CLASSES), pretrained=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    manager = CheckpointManager(directory)
    return manager.save(
        "best.pt",
        model=model,
        optimizer=optimizer,
        scheduler=None,
        epoch=0,
        best_validation_metric=None,
        metadata=CheckpointMetadata(
            class_mapping={label: index for index, label in enumerate(BASE_CLASSES)},
            model_config={"backbone": "mobilenetv3_small_100", "dropout": 0.2},
            preprocessing_config={
                "image_size": 64,
                "normalization_mean": [0.485, 0.456, 0.406],
                "normalization_std": [0.229, 0.224, 0.225],
            },
            random_seed=42,
            training_config={"epochs": 1},
            project_metadata={"name": "test"},
        ),
    )


def _images(count: int, *, value: int = 120, size: int = 48) -> list[np.ndarray]:
    """Build deterministic uint8 RGB images."""
    generator = np.random.default_rng(value)
    return [
        np.clip(
            np.full((size, size, 3), value, dtype=np.int16)
            + generator.integers(-6, 7, size=(size, size, 3)),
            0,
            255,
        ).astype(np.uint8)
        for _ in range(count)
    ]


@pytest.fixture
def recognizer(tmp_path: Path) -> OpenSetRecognizer:
    checkpoint = _write_checkpoint(tmp_path / "checkpoints")
    return OpenSetRecognizer.from_checkpoint(
        checkpoint,
        registry_path=tmp_path / "artifacts" / "registry.npz",
        device="cpu",
        registration_policy=RegistrationPolicy(min_coherence=-1.0),
    )


def test_recognizer_rebuilds_from_a_checkpoint(recognizer: OpenSetRecognizer) -> None:
    info = recognizer.info()

    assert recognizer.class_names == BASE_CLASSES
    assert info.base_class_count == 3
    assert info.embedding_dim > 0
    assert info.registered_class_count == 0
    # Placeholder thresholds must never be reported as measured.
    assert info.thresholds_calibrated is False


def test_backbone_is_frozen_after_loading(recognizer: OpenSetRecognizer) -> None:
    assert all(not p.requires_grad for p in recognizer.model.parameters())
    assert recognizer.model.training is False


def test_prediction_returns_one_decision_per_image(
    recognizer: OpenSetRecognizer,
) -> None:
    decisions = recognizer.predict_images(_images(3))

    assert len(decisions) == 3
    for decision in decisions:
        assert decision.verdict in set(Verdict)
        assert decision.base.available is True
        assert decision.base.label in BASE_CLASSES
        assert 0.0 <= decision.base.confidence <= 1.0


def test_everything_is_unknown_before_anything_is_registered(
    recognizer: OpenSetRecognizer,
) -> None:
    # An untrained head is not confident, and nothing is registered, so the only
    # honest answer is "unknown".
    strict = OpenSetThresholds(base_confidence_threshold=0.99)
    recognizer.thresholds = strict

    decision = recognizer.predict_images(_images(1))[0]

    assert decision.verdict is Verdict.UNKNOWN
    assert "no incremental class is registered" in decision.reason


def test_registering_a_new_sign_does_not_change_any_model_weight(
    recognizer: OpenSetRecognizer,
) -> None:
    # This is the project's central invariant: adding a class must not touch the
    # base model. Compare every parameter tensor before and after.
    before = {
        name: parameter.detach().clone()
        for name, parameter in recognizer.model.state_dict().items()
    }

    recognizer.register_sign("school_ahead", _images(4, value=200))

    after = recognizer.model.state_dict()
    assert set(before) == set(after)
    for name, tensor in before.items():
        assert torch.equal(tensor, after[name]), f"weight {name} changed"


def test_registering_a_new_sign_does_not_grow_the_classifier_head(
    recognizer: OpenSetRecognizer,
) -> None:
    recognizer.register_sign("school_ahead", _images(4, value=200))

    assert recognizer.model.num_classes == len(BASE_CLASSES)
    assert recognizer.model.classifier.out_features == len(BASE_CLASSES)
    assert recognizer.class_names == BASE_CLASSES


def test_a_registered_sign_becomes_recognizable(
    recognizer: OpenSetRecognizer,
) -> None:
    reference = _images(4, value=200)
    recognizer.register_sign("school_ahead", reference)
    # Force the registry to be the deciding evidence.
    recognizer.thresholds = OpenSetThresholds(
        base_confidence_threshold=0.99,
        prototype_similarity_threshold=0.5,
    )

    decision = recognizer.predict_images([reference[0]])[0]

    assert decision.verdict is Verdict.REGISTERED_CLASS
    assert decision.label == "school_ahead"
    assert decision.prototype.similarity >= 0.5


def test_registration_persists_across_a_reload(
    tmp_path: Path, recognizer: OpenSetRecognizer
) -> None:
    recognizer.register_sign("school_ahead", _images(4, value=200))
    checkpoint = tmp_path / "checkpoints" / "best.pt"

    reloaded = OpenSetRecognizer.from_checkpoint(
        checkpoint,
        registry_path=tmp_path / "artifacts" / "registry.npz",
        device="cpu",
    )

    assert reloaded.registered_labels == ("school_ahead",)
    assert reloaded.info().registered_class_count == 1


def test_unregistering_removes_the_class_everywhere(
    tmp_path: Path, recognizer: OpenSetRecognizer
) -> None:
    recognizer.register_sign("school_ahead", _images(4, value=200))
    recognizer.unregister_sign("school_ahead")

    reloaded = OpenSetRecognizer.from_checkpoint(
        tmp_path / "checkpoints" / "best.pt",
        registry_path=tmp_path / "artifacts" / "registry.npz",
        device="cpu",
    )

    assert reloaded.registered_labels == ()


def test_embeddings_are_normalized_and_shaped_per_image(
    recognizer: OpenSetRecognizer,
) -> None:
    embeddings = recognizer.embed_images(_images(3))

    assert embeddings.shape == (3, recognizer.info().embedding_dim)
    np.testing.assert_allclose(
        np.linalg.norm(embeddings, axis=1), np.ones(3), atol=1e-5
    )


def test_embedding_is_deterministic(recognizer: OpenSetRecognizer) -> None:
    images = _images(2)

    np.testing.assert_allclose(
        recognizer.embed_images(images), recognizer.embed_images(images)
    )


def test_prototype_priority_lets_a_registered_sign_outrank_the_base_head(
    recognizer: OpenSetRecognizer,
) -> None:
    reference = _images(4, value=200)
    recognizer.register_sign("school_ahead", reference)
    recognizer.thresholds = OpenSetThresholds(
        base_confidence_threshold=0.0,  # base would always win under classifier_first
        prototype_similarity_threshold=0.5,
        strategy=Strategy.PROTOTYPE_PRIORITY,
    )

    decision = recognizer.predict_images([reference[0]])[0]

    assert decision.verdict is Verdict.REGISTERED_CLASS


def test_registering_too_few_references_is_refused(
    recognizer: OpenSetRecognizer,
) -> None:
    with pytest.raises(RegistrationError, match="at least"):
        recognizer.register_sign("school_ahead", _images(2, value=200))


def test_predicting_without_images_is_refused(recognizer: OpenSetRecognizer) -> None:
    with pytest.raises(InferenceError, match="At least one image"):
        recognizer.predict_images([])


def test_a_checkpoint_with_gapped_class_indices_is_refused(tmp_path: Path) -> None:
    model = RoadSignClassifier(num_classes=2, pretrained=False)
    manager = CheckpointManager(tmp_path / "bad")
    path = manager.save(
        "best.pt",
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        scheduler=None,
        epoch=0,
        best_validation_metric=None,
        metadata=CheckpointMetadata(
            class_mapping={"give_way": 0, "no_entry": 5},  # not contiguous
            model_config={"backbone": "mobilenetv3_small_100", "dropout": 0.2},
            preprocessing_config={"image_size": 64},
            random_seed=42,
            training_config={},
            project_metadata={},
        ),
    )

    with pytest.raises(InferenceError, match="contiguous"):
        OpenSetRecognizer.from_checkpoint(path, device="cpu")


def test_a_missing_checkpoint_is_reported(tmp_path: Path) -> None:
    with pytest.raises(InferenceError, match="Could not read checkpoint"):
        OpenSetRecognizer.from_checkpoint(tmp_path / "absent.pt", device="cpu")


def test_a_missing_registry_file_yields_an_empty_registry(tmp_path: Path) -> None:
    checkpoint = _write_checkpoint(tmp_path / "checkpoints")

    recognizer: Any = OpenSetRecognizer.from_checkpoint(
        checkpoint, registry_path=tmp_path / "not_created_yet.npz", device="cpu"
    )

    assert recognizer.registered_labels == ()
