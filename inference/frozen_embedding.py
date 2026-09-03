"""Frozen checkpoint-backed embedding extraction for incremental recognition."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
from torch.nn import functional as F

from models.classifier import RoadSignClassifier
from models.factory import ModelConfig, build_classifier
from training.checkpoint import load_checkpoint, read_checkpoint_payload
from training.transforms import TransformConfig, build_evaluation_transform
from utils.image_validation import decode_image

FloatArray = npt.NDArray[np.float32]


class FrozenEmbeddingError(RuntimeError):
    """Raised when a frozen checkpoint or embedding request is invalid."""


@dataclass(frozen=True, slots=True)
class FrozenCheckpointIdentity:
    """Auditable identity and configuration of the frozen V2 checkpoint."""

    path: str
    sha256: str
    epoch: int
    best_validation_metric: float
    embedding_dim: int
    class_mapping: dict[str, int]
    model_config: dict[str, Any]
    preprocessing_config: dict[str, Any]


@dataclass(frozen=True, slots=True)
class FrozenBatchOutput:
    """Normalized embeddings and unchanged base-class probabilities."""

    embeddings: FloatArray
    base_probabilities: FloatArray


class FrozenEmbeddingPipeline:
    """Load a classifier checkpoint and expose deterministic inference only.

    All model parameters have ``requires_grad=False`` and the model is forced into
    evaluation mode before every forward pass. The evaluation transform is rebuilt
    from checkpoint metadata; training augmentation fields are retained for audit
    identity but never applied during extraction.
    """

    def __init__(
        self,
        *,
        model: RoadSignClassifier,
        transform: Any,
        identity: FrozenCheckpointIdentity,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.transform = transform
        self.identity = identity
        self.device = device
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.eval()
        if self.model.feature_extractor.embedding_dim != identity.embedding_dim:
            raise FrozenEmbeddingError(
                "Model embedding dimension differs from identity"
            )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device = "cpu",
        expected_sha256: str | None = None,
        expected_embedding_dim: int = 1024,
    ) -> FrozenEmbeddingPipeline:
        """Load, validate, freeze, and identify a trained classifier checkpoint."""
        path = Path(checkpoint_path).expanduser().resolve()
        sha256 = _file_sha256(path)
        if expected_sha256 is not None and sha256.lower() != expected_sha256.lower():
            raise FrozenEmbeddingError(
                f"Checkpoint SHA-256 mismatch: expected {expected_sha256}, got {sha256}"
            )
        selected_device = torch.device(device)
        payload = read_checkpoint_payload(path, map_location=selected_device)
        class_mapping = _class_mapping(payload.get("class_mapping"))
        raw_model_config = _string_mapping(payload.get("model_config"), "model_config")
        raw_preprocessing = _string_mapping(
            payload.get("preprocessing_config"), "preprocessing_config"
        )
        try:
            recorded_model_config = ModelConfig(**raw_model_config)
            transform_config = _transform_config(raw_preprocessing)
        except (TypeError, ValueError) as error:
            raise FrozenEmbeddingError(
                "Checkpoint model or preprocessing configuration is incompatible"
            ) from error
        # Checkpoint weights replace initialization, so construction must never
        # download pretrained weights.
        model = build_classifier(
            replace(recorded_model_config, pretrained=False), class_mapping
        )
        load_checkpoint(
            path,
            model=model,
            expected_class_mapping=class_mapping,
            expected_model_config=raw_model_config,
            map_location=selected_device,
        )
        embedding_dim = model.feature_extractor.embedding_dim
        if embedding_dim != expected_embedding_dim:
            raise FrozenEmbeddingError(
                f"Expected embedding dimension {expected_embedding_dim}, got {embedding_dim}"
            )
        epoch = payload.get("epoch")
        best_metric = payload.get("best_validation_metric")
        if not isinstance(epoch, int) or not isinstance(best_metric, (int, float)):
            raise FrozenEmbeddingError("Checkpoint epoch or best metric is invalid")
        identity = FrozenCheckpointIdentity(
            path=str(path),
            sha256=sha256,
            epoch=epoch,
            best_validation_metric=float(best_metric),
            embedding_dim=embedding_dim,
            class_mapping=class_mapping,
            model_config=raw_model_config,
            preprocessing_config=raw_preprocessing,
        )
        return cls(
            model=model,
            transform=build_evaluation_transform(transform_config),
            identity=identity,
            device=selected_device,
        )

    @property
    def all_parameters_frozen(self) -> bool:
        """Return whether every feature-extractor and classifier parameter is frozen."""
        return all(not parameter.requires_grad for parameter in self.model.parameters())

    def model_state_sha256(self) -> str:
        """Hash tensor names, shapes, dtypes, and values in the current model state."""
        digest = hashlib.sha256()
        for name, tensor in self.model.state_dict().items():
            value = tensor.detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(value.numpy().tobytes())
        return digest.hexdigest()

    def preprocess_arrays(self, images: Sequence[npt.NDArray[np.uint8]]) -> Tensor:
        """Apply exact deterministic checkpoint evaluation preprocessing."""
        if not images:
            raise ValueError("images must contain at least one image")
        tensors = [self.transform(image) for image in images]
        if any(not isinstance(tensor, Tensor) for tensor in tensors):
            raise FrozenEmbeddingError("Evaluation transform did not return tensors")
        return torch.stack(tensors)

    def preprocess_paths(self, paths: Sequence[str | Path]) -> Tensor:
        """Decode image paths and apply checkpoint evaluation preprocessing."""
        if not paths:
            raise ValueError("paths must contain at least one image path")
        images = [decode_image(path, convert_to_rgb=True) for path in paths]
        return self.preprocess_arrays(images)

    def infer_preprocessed(
        self, images: Tensor, *, batch_size: int = 32
    ) -> FrozenBatchOutput:
        """Extract normalized embeddings and base probabilities without mutation."""
        if images.ndim != 4 or images.shape[0] == 0:
            raise ValueError("images must be a non-empty NCHW tensor")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        before = self.model_state_sha256()
        embedding_chunks: list[Tensor] = []
        probability_chunks: list[Tensor] = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, int(images.shape[0]), batch_size):
                batch = images[start : start + batch_size].to(self.device)
                embeddings = F.normalize(
                    self.model.extract_embeddings(batch), p=2, dim=1
                )
                logits = self.model.classifier(embeddings)
                embedding_chunks.append(embeddings.cpu())
                probability_chunks.append(torch.softmax(logits, dim=1).cpu())
        if self.model_state_sha256() != before:
            raise FrozenEmbeddingError("Frozen model state changed during inference")
        embeddings_array = np.asarray(
            torch.cat(embedding_chunks).numpy(), dtype=np.float32
        )
        probabilities_array = np.asarray(
            torch.cat(probability_chunks).numpy(), dtype=np.float32
        )
        if embeddings_array.shape[1] != self.identity.embedding_dim:
            raise FrozenEmbeddingError("Extracted embedding dimension is invalid")
        if not np.all(np.isfinite(embeddings_array)):
            raise FrozenEmbeddingError("Extracted embeddings contain non-finite values")
        return FrozenBatchOutput(
            embeddings=embeddings_array,
            base_probabilities=probabilities_array,
        )

    def extract_paths(
        self, paths: Sequence[str | Path], *, batch_size: int = 32
    ) -> FloatArray:
        """Return normalized embeddings for decoded image paths."""
        return self.infer_preprocessed(
            self.preprocess_paths(paths), batch_size=batch_size
        ).embeddings

    def extract_arrays(
        self,
        images: Sequence[npt.NDArray[np.uint8]],
        *,
        batch_size: int = 32,
    ) -> FloatArray:
        """Return normalized embeddings for RGB image arrays."""
        return self.infer_preprocessed(
            self.preprocess_arrays(images), batch_size=batch_size
        ).embeddings


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise FrozenEmbeddingError(f"Could not read checkpoint {path}") from error
    return digest.hexdigest()


def _class_mapping(value: Any) -> dict[str, int]:
    if not isinstance(value, dict) or any(
        not isinstance(label, str)
        or isinstance(index, bool)
        or not isinstance(index, int)
        for label, index in value.items()
    ):
        raise FrozenEmbeddingError("Checkpoint class_mapping is invalid")
    mapping = dict(value)
    if sorted(mapping.values()) != list(range(len(mapping))):
        raise FrozenEmbeddingError("Checkpoint class_mapping must be contiguous")
    return mapping


def _string_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise FrozenEmbeddingError(f"Checkpoint {name} is invalid")
    return dict(value)


def _transform_config(value: dict[str, Any]) -> TransformConfig:
    copied = dict(value)
    for key in ("normalization_mean", "normalization_std"):
        raw = copied.get(key)
        if isinstance(raw, list):
            copied[key] = tuple(float(item) for item in raw)
    config = TransformConfig(**copied)
    config.validate()
    return config
