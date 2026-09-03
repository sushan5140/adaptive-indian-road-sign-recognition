"""End-to-end open-set recognizer: base classifier plus prototype registry.

This ties the three existing pieces together into the behaviour the project
exists to demonstrate. One forward pass through the frozen backbone yields both
the closed-set probabilities and the embedding; the embedding is searched
against the incremental registry; :mod:`inference.decision` arbitrates.

The same recognizer also performs registration, so the embedding used to
register a sign is produced by exactly the same preprocessing and backbone as
the embedding used to recognize it later. Registering never triggers a backward
pass and never touches the base dataset or the classifier head.

Torch is imported lazily. Importing this module in an environment without the
deep-learning stack raises only when a model is actually constructed.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt

from inference.decision import (
    OpenSetDecision,
    OpenSetThresholds,
    build_base_evidence,
    build_prototype_evidence,
    decide,
)
from inference.registration import (
    IncrementalRegistrar,
    RegistrationPolicy,
    RegistrationResult,
)
from models.prototype_registry import PrototypeRegistry
from training.checkpoint import CheckpointError, read_checkpoint_payload
from training.transforms import TransformConfig, build_evaluation_transform
from utils.dependencies import DependencyUnavailableError
from utils.image_validation import decode_image

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

    from models.classifier import RoadSignClassifier


class InferenceError(RuntimeError):
    """Raised when the recognizer cannot be built or cannot run."""


@dataclass(frozen=True, slots=True)
class RecognizerInfo:
    """Non-sensitive description of a constructed recognizer."""

    backbone: str
    base_class_count: int
    embedding_dim: int
    device: str
    registered_class_count: int
    thresholds_calibrated: bool
    checkpoint_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view."""
        return {
            "backbone": self.backbone,
            "base_class_count": self.base_class_count,
            "embedding_dim": self.embedding_dim,
            "device": self.device,
            "registered_class_count": self.registered_class_count,
            "thresholds_calibrated": self.thresholds_calibrated,
            "checkpoint_sha256": self.checkpoint_sha256,
        }


class OpenSetRecognizer:
    """Recognize base classes, registered classes, and unknown signs.

    Args:
        model: A trained :class:`models.classifier.RoadSignClassifier` in eval
            mode.
        class_names: Base labels in classifier index order.
        registry: The incremental prototype registry to search.
        thresholds: Decision thresholds. Uncalibrated defaults when omitted.
        transform_config: Preprocessing that must match training.
        device: The torch device the model lives on.
        registrar: Optional preconfigured registrar. One is created against
            ``registry`` when omitted.
    """

    def __init__(
        self,
        model: RoadSignClassifier,
        class_names: Sequence[str],
        *,
        registry: PrototypeRegistry | None = None,
        thresholds: OpenSetThresholds | None = None,
        transform_config: TransformConfig | None = None,
        device: torch.device | None = None,
        registry_path: str | Path | None = None,
        registration_policy: RegistrationPolicy | None = None,
        checkpoint_sha256: str | None = None,
    ) -> None:
        torch = _require_torch()
        labels = tuple(str(name) for name in class_names)
        if not labels:
            raise InferenceError("class_names must not be empty")
        if len(set(labels)) != len(labels):
            raise InferenceError("class_names must not contain duplicates")

        self.model = model
        self.class_names = labels
        self.registry = registry if registry is not None else PrototypeRegistry()
        self.thresholds = thresholds or OpenSetThresholds()
        self.transform_config = transform_config or TransformConfig()
        self._device = device or torch.device("cpu")
        self._transform = build_evaluation_transform(self.transform_config)
        self.checkpoint_sha256 = checkpoint_sha256

        self.model.to(self._device)
        self.model.eval()

        # The registrar always receives the base labels and a fingerprint hook,
        # so a label collision or an accidental weight change is caught at
        # runtime rather than only in tests.
        self.registrar = IncrementalRegistrar(
            self.registry,
            registry_path=registry_path,
            policy=registration_policy,
            base_labels=labels,
            model_fingerprint=self.model_state_sha256,
        )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        registry_path: str | Path | None = None,
        thresholds: OpenSetThresholds | None = None,
        device: str = "auto",
        registration_policy: RegistrationPolicy | None = None,
        expected_sha256: str | None = None,
    ) -> OpenSetRecognizer:
        """Rebuild a recognizer from a training checkpoint and saved registry.

        The class mapping, backbone name, and preprocessing configuration all
        travel inside the checkpoint, so the recognizer cannot be assembled with
        a label ordering or an input normalization that disagrees with training.

        Args:
            checkpoint_path: Path to a ``best.pt`` or ``last.pt`` checkpoint.
            registry_path: Optional NPZ registry to load. A missing file yields
                an empty registry, which is the correct state before any sign
                has been registered.
            thresholds: Decision thresholds. Uncalibrated defaults when omitted.
            device: ``auto``, ``cpu``, or ``cuda``.
            registration_policy: Bounds applied to later registrations.

        Returns:
            A ready recognizer in eval mode.

        Raises:
            InferenceError: If the checkpoint is unreadable, its metadata is
                inconsistent, or the registry cannot be loaded.
        """
        from models.classifier import RoadSignClassifier
        from utils.device import DeviceSelectionError, select_device

        torch = _require_torch()
        checkpoint_sha256 = _file_sha256(Path(checkpoint_path).expanduser())
        if (
            expected_sha256 is not None
            and checkpoint_sha256.lower() != expected_sha256.lower()
        ):
            raise InferenceError(
                f"Checkpoint SHA-256 mismatch: expected {expected_sha256}, "
                f"got {checkpoint_sha256}"
            )
        try:
            payload = read_checkpoint_payload(checkpoint_path)
        except CheckpointError as error:
            raise InferenceError(f"Could not read checkpoint: {error}") from error

        class_names = _ordered_class_names(payload.get("class_mapping"))
        model_config = payload.get("model_config") or {}
        preprocessing = payload.get("preprocessing_config") or {}
        if not isinstance(model_config, Mapping) or not isinstance(
            preprocessing, Mapping
        ):
            raise InferenceError("Checkpoint model/preprocessing config must be maps")

        try:
            selected_device = select_device(device)
        except DeviceSelectionError as error:
            raise InferenceError(str(error)) from error

        model = RoadSignClassifier(
            num_classes=len(class_names),
            backbone_name=str(model_config.get("backbone", "mobilenetv3_small_100")),
            # Weights come from the checkpoint; never re-download here.
            pretrained=False,
            dropout=float(model_config.get("dropout", 0.2)),
        )
        try:
            model.load_state_dict(payload["model_state_dict"], strict=True)
        except (KeyError, RuntimeError, ValueError) as error:
            raise InferenceError(
                f"Checkpoint weights do not match the described architecture: {error}"
            ) from error

        registry = _load_registry(registry_path)
        recognizer = cls(
            model,
            class_names,
            registry=registry,
            thresholds=thresholds,
            transform_config=_transform_config_from(preprocessing),
            device=selected_device,
            registry_path=registry_path,
            registration_policy=registration_policy,
            checkpoint_sha256=checkpoint_sha256,
        )
        # Freeze explicitly: registration must be structurally incapable of
        # updating backbone weights, not merely conventionally careful.
        for parameter in recognizer.model.parameters():
            parameter.requires_grad_(False)
        del torch
        return recognizer

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def info(self) -> RecognizerInfo:
        """Describe this recognizer for a health endpoint or the UI."""
        return RecognizerInfo(
            backbone=getattr(
                self.model.feature_extractor.backbone,
                "default_cfg",
                {},
            ).get("architecture", "unknown"),
            base_class_count=len(self.class_names),
            embedding_dim=int(self.model.feature_extractor.embedding_dim),
            device=str(self._device),
            registered_class_count=len(self.registry),
            thresholds_calibrated=self.thresholds.calibrated,
            checkpoint_sha256=self.checkpoint_sha256,
        )

    def model_state_sha256(self) -> str:
        """Digest tensor names, dtypes, shapes and values of the model state.

        This is the fingerprint the registrar samples before and after each
        registration, so an accidental weight change is detected immediately
        rather than being discovered later as drifted behaviour.
        """
        digest = hashlib.sha256()
        for name, tensor in self.model.state_dict().items():
            value = tensor.detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(value.numpy().tobytes())
        return digest.hexdigest()

    @property
    def registered_labels(self) -> tuple[str, ...]:
        """Labels currently held in the incremental registry."""
        return self.registry.labels

    # ------------------------------------------------------------------
    # Embedding and prediction
    # ------------------------------------------------------------------
    def embed_images(self, images: Sequence[np.ndarray]) -> npt.NDArray[np.float32]:
        """Embed decoded RGB images with the frozen backbone.

        Args:
            images: Decoded ``HWC`` RGB arrays, as returned by
                :func:`utils.image_validation.decode_image`.

        Returns:
            A ``(len(images), embedding_dim)`` float32 array of L2-normalized
            embeddings on the CPU.

        Raises:
            InferenceError: If ``images`` is empty or preprocessing fails.
        """
        torch = _require_torch()
        if not images:
            raise InferenceError("At least one image is required")
        batch = self._preprocess(images)
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                embeddings = self.model.extract_embeddings(batch)
        finally:
            self.model.train(was_training)
        return np.asarray(embeddings.detach().cpu().numpy(), dtype=np.float32)

    def predict_images(
        self, images: Sequence[np.ndarray], *, top_k: int = 3
    ) -> list[OpenSetDecision]:
        """Classify decoded images as base, registered, or unknown.

        One forward pass produces both the closed-set probabilities and the
        embedding, so the two evidence sources always describe the same tensor.

        Args:
            images: Decoded ``HWC`` RGB arrays.
            top_k: How many ranked base classes and prototype matches to report.

        Returns:
            One decision per input image, in input order.

        Raises:
            InferenceError: If ``images`` is empty or inference fails.
        """
        torch = _require_torch()
        if not images:
            raise InferenceError("At least one image is required")
        batch = self._preprocess(images)
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                embeddings = self.model.extract_embeddings(batch)
                logits = self.model.classifier(embeddings)
                probabilities = torch.softmax(logits, dim=-1)
        finally:
            self.model.train(was_training)

        probability_rows = probabilities.detach().cpu().numpy().astype(np.float32)
        embedding_rows = embeddings.detach().cpu().numpy().astype(np.float32)

        decisions: list[OpenSetDecision] = []
        for probability_row, embedding_row in zip(
            probability_rows, embedding_rows, strict=True
        ):
            matches = (
                self.registry.search(embedding_row, top_k=top_k)
                if len(self.registry)
                else []
            )
            decisions.append(
                decide(
                    base=build_base_evidence(
                        probability_row.tolist(),
                        self.class_names,
                        top_k=top_k,
                    ),
                    prototype=build_prototype_evidence(matches),
                    thresholds=self.thresholds,
                )
            )
        return decisions

    def predict_paths(
        self, paths: Sequence[str | Path], *, top_k: int = 3
    ) -> list[OpenSetDecision]:
        """Decode image files and classify them.

        Args:
            paths: Image file paths.
            top_k: How many ranked entries to report per evidence source.

        Returns:
            One decision per path, in input order.

        Raises:
            InferenceError: If ``paths`` is empty or an image cannot be decoded.
        """
        if not paths:
            raise InferenceError("At least one image path is required")
        return self.predict_images([self._decode(path) for path in paths], top_k=top_k)

    # ------------------------------------------------------------------
    # Incremental registration
    # ------------------------------------------------------------------
    def register_sign(
        self,
        label: str,
        images: Sequence[np.ndarray],
        *,
        metadata: Mapping[str, Any] | None = None,
        overwrite: bool = False,
        persist: bool = True,
    ) -> RegistrationResult:
        """Register a new sign from a few reference images.

        No gradient is computed, the classifier head is unchanged, and the base
        dataset is untouched.

        Args:
            label: Stable, human-readable class name.
            images: Decoded reference images of the one new sign.
            metadata: JSON-serializable provenance to store.
            overwrite: Whether an existing label may be replaced.
            persist: Whether to write the registry afterwards.

        Returns:
            The measured :class:`RegistrationResult`.

        Raises:
            RegistrationError: If any registration rule fails.
        """
        provenance = dict(metadata or {})
        if self.checkpoint_sha256 is not None:
            provenance.setdefault("checkpoint_sha256", self.checkpoint_sha256)
        return self.registrar.register_images(
            label,
            images,
            self.embed_images,
            metadata=provenance,
            overwrite=overwrite,
            persist=persist,
        )

    def register_sign_from_paths(
        self,
        label: str,
        paths: Sequence[str | Path],
        *,
        metadata: Mapping[str, Any] | None = None,
        overwrite: bool = False,
        persist: bool = True,
    ) -> RegistrationResult:
        """Register a new sign from reference image files.

        Args:
            label: Stable, human-readable class name.
            paths: Reference image paths.
            metadata: JSON-serializable provenance to store.
            overwrite: Whether an existing label may be replaced.
            persist: Whether to write the registry afterwards.

        Returns:
            The measured :class:`RegistrationResult`.
        """
        enriched = dict(metadata or {})
        enriched.setdefault("reference_filenames", [Path(p).name for p in paths])
        return self.register_sign(
            label,
            [self._decode(path) for path in paths],
            metadata=enriched,
            overwrite=overwrite,
            persist=persist,
        )

    def unregister_sign(self, label: str, *, persist: bool = True) -> None:
        """Remove a previously registered incremental class."""
        self.registrar.unregister(label, persist=persist)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _preprocess(self, images: Sequence[np.ndarray]) -> torch.Tensor:
        """Apply deterministic evaluation preprocessing and stack a batch."""
        torch_module = _require_torch()
        tensors = []
        for index, image in enumerate(images):
            if not isinstance(image, np.ndarray) or image.size == 0:
                raise InferenceError(f"Image at position {index} is not a valid array")
            try:
                tensors.append(self._transform(image))
            except (ValueError, TypeError, RuntimeError) as error:
                raise InferenceError(
                    f"Could not preprocess image at position {index}: {error}"
                ) from error
        return cast("torch.Tensor", torch_module.stack(tensors).to(self._device))

    @staticmethod
    def _decode(path: str | Path) -> np.ndarray:
        """Decode one image file as RGB, translating errors."""
        from utils.image_validation import ImageValidationError

        try:
            return decode_image(path, convert_to_rgb=True)
        except ImageValidationError as error:
            raise InferenceError(f"Could not read image {path}: {error}") from error


def _file_sha256(path: Path) -> str:
    """Hash a checkpoint file so a prototype can name the model that made it."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise InferenceError(f"Could not read checkpoint {path}: {error}") from error
    return digest.hexdigest()


def _ordered_class_names(class_mapping: Any) -> tuple[str, ...]:
    """Return base labels ordered by their classifier index."""
    if not isinstance(class_mapping, Mapping) or not class_mapping:
        raise InferenceError("Checkpoint class_mapping must be a non-empty mapping")
    try:
        pairs = sorted(
            ((str(label), int(index)) for label, index in class_mapping.items()),
            key=lambda item: item[1],
        )
    except (TypeError, ValueError) as error:
        raise InferenceError(
            "Checkpoint class_mapping must map labels to ints"
        ) from error
    indices = [index for _, index in pairs]
    if indices != list(range(len(indices))):
        raise InferenceError(
            "Checkpoint class_mapping indices must be contiguous and zero-based"
        )
    return tuple(label for label, _ in pairs)


def _load_registry(registry_path: str | Path | None) -> PrototypeRegistry:
    """Load a registry, tolerating a not-yet-created file."""
    if registry_path is None:
        return PrototypeRegistry()
    path = Path(registry_path).expanduser()
    if not path.exists():
        return PrototypeRegistry()
    try:
        return PrototypeRegistry.load(path)
    except ValueError as error:
        raise InferenceError(f"Could not load prototype registry: {error}") from error


def _transform_config_from(preprocessing: Mapping[str, Any]) -> TransformConfig:
    """Rebuild evaluation preprocessing from checkpoint metadata."""
    mean = preprocessing.get("normalization_mean")
    std = preprocessing.get("normalization_std")
    defaults = TransformConfig()
    return TransformConfig(
        image_size=int(preprocessing.get("image_size", defaults.image_size)),
        # Evaluation preprocessing is deterministic; augmentation is irrelevant.
        horizontal_flip_probability=0.0,
        max_rotation_degrees=0.0,
        brightness=0.0,
        contrast=0.0,
        normalization_mean=(
            tuple(float(value) for value in mean)  # type: ignore[arg-type]
            if isinstance(mean, (list, tuple)) and len(mean) == 3
            else defaults.normalization_mean
        ),
        normalization_std=(
            tuple(float(value) for value in std)  # type: ignore[arg-type]
            if isinstance(std, (list, tuple)) and len(std) == 3
            else defaults.normalization_std
        ),
    )


def _require_torch() -> Any:
    """Import torch lazily with a clear error when it is unavailable."""
    try:
        import torch
    except ImportError as error:
        raise DependencyUnavailableError(
            "PyTorch is required for open-set inference"
        ) from error
    return torch
