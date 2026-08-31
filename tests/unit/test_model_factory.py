"""Tests for model class-count compatibility."""

import pytest

pytest.importorskip("torch", reason="PyTorch is required for model factory tests")
pytest.importorskip("timm", reason="timm is required for model factory tests")

from models.factory import ModelConfig, ModelConfigurationError, build_classifier


def test_auto_class_count_matches_dataset() -> None:
    model = build_classifier(ModelConfig(pretrained=False), {"stop": 0, "yield": 1})
    assert model.num_classes == 2


def test_explicit_incompatible_class_count_is_rejected() -> None:
    with pytest.raises(ModelConfigurationError, match="conflicts"):
        build_classifier(
            ModelConfig(num_classes=3, pretrained=False),
            {"stop": 0, "yield": 1},
        )
