"""Persistent, pickle-free registry of incremental class prototypes."""

from __future__ import annotations

import copy
import json
import os
import tempfile
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class PrototypeMatch:
    """A cosine-similarity match returned by a prototype search."""

    label: str
    similarity: float
    metadata: dict[str, Any]


class PrototypeRegistry:
    """Manage normalized prototypes for few-shot incremental classes.

    Prototypes are computed by normalizing each reference embedding, averaging the
    references, and normalizing the mean. Persistence uses non-object NumPy arrays
    plus JSON metadata and always loads with ``allow_pickle=False``.

    Args:
        embedding_dim: Optional expected embedding width. It is inferred when the
            first class is added if omitted.
    """

    SCHEMA_VERSION: Final[int] = 1
    _MAX_LABEL_LENGTH: Final[int] = 256

    def __init__(self, embedding_dim: int | None = None) -> None:
        if embedding_dim is not None and embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive when provided")
        self._embedding_dim = embedding_dim
        self._prototypes: dict[str, FloatArray] = {}
        self._metadata: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    @property
    def embedding_dim(self) -> int | None:
        """Return the required embedding width, if known."""
        return self._embedding_dim

    @property
    def labels(self) -> tuple[str, ...]:
        """Return registered labels in insertion order."""
        with self._lock:
            return tuple(self._prototypes)

    def __len__(self) -> int:
        """Return the number of registered incremental classes."""
        with self._lock:
            return len(self._prototypes)

    def __contains__(self, label: object) -> bool:
        """Return whether a label is registered."""
        with self._lock:
            return label in self._prototypes

    @staticmethod
    def normalize_embeddings(embeddings: npt.ArrayLike) -> FloatArray:
        """Convert embeddings to finite row-wise L2-normalized float32 values.

        A one-dimensional input is returned as one normalized vector. A
        two-dimensional input is normalized row by row.

        Raises:
            ValueError: If the input is empty, non-finite, has an unsupported
                shape, or contains a zero-norm embedding.
        """
        array = np.asarray(embeddings, dtype=np.float32)
        if array.ndim not in (1, 2):
            raise ValueError("embeddings must be a one- or two-dimensional array")
        if array.size == 0 or array.shape[-1] == 0:
            raise ValueError("embeddings must not be empty")
        if not np.all(np.isfinite(array)):
            raise ValueError("embeddings must contain only finite values")

        norms = np.linalg.norm(array, axis=-1, keepdims=True)
        if np.any(norms <= np.finfo(np.float32).eps):
            raise ValueError("embeddings must have non-zero L2 norm")
        normalized = array / norms
        return np.asarray(normalized, dtype=np.float32)

    @classmethod
    def calculate_prototype(cls, embeddings: npt.ArrayLike) -> FloatArray:
        """Calculate a normalized class prototype from one or more embeddings."""
        array = np.asarray(embeddings, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2 or array.shape[0] == 0:
            raise ValueError("embeddings must contain at least one embedding")
        normalized = cls.normalize_embeddings(array)
        mean = normalized.mean(axis=0)
        try:
            return cls.normalize_embeddings(mean)
        except ValueError as error:
            raise ValueError(
                "reference embeddings cancel out and do not define a prototype"
            ) from error

    def add_class(
        self,
        label: str,
        embeddings: npt.ArrayLike,
        *,
        metadata: Mapping[str, Any] | None = None,
        overwrite: bool = False,
    ) -> FloatArray:
        """Register a class prototype calculated from reference embeddings.

        Args:
            label: Stable, human-readable incremental class label.
            embeddings: One embedding or an array of reference embeddings.
            metadata: JSON-serializable descriptive values.
            overwrite: Whether an existing label may be replaced.

        Returns:
            A defensive copy of the normalized prototype.
        """
        validated_label = self._validate_label(label)
        validated_metadata = self._validate_metadata(metadata)
        prototype = self.calculate_prototype(embeddings)

        with self._lock:
            if validated_label in self._prototypes and not overwrite:
                raise ValueError(f"Class {validated_label!r} is already registered")
            if self._embedding_dim is None:
                self._embedding_dim = int(prototype.shape[0])
            elif prototype.shape[0] != self._embedding_dim:
                raise ValueError(
                    f"Expected embedding dimension {self._embedding_dim}, "
                    f"got {prototype.shape[0]}"
                )
            self._prototypes[validated_label] = prototype.copy()
            self._metadata[validated_label] = validated_metadata
        return prototype.copy()

    def remove_class(self, label: str) -> None:
        """Remove an incremental class.

        Raises:
            KeyError: If ``label`` is not registered.
        """
        with self._lock:
            if label not in self._prototypes:
                raise KeyError(f"Class {label!r} is not registered")
            del self._prototypes[label]
            del self._metadata[label]

    def get_prototype(self, label: str) -> FloatArray:
        """Return a defensive copy of a registered class prototype."""
        with self._lock:
            try:
                return self._prototypes[label].copy()
            except KeyError as error:
                raise KeyError(f"Class {label!r} is not registered") from error

    def get_metadata(self, label: str) -> dict[str, Any]:
        """Return a deep copy of a registered class's metadata."""
        with self._lock:
            try:
                return copy.deepcopy(self._metadata[label])
            except KeyError as error:
                raise KeyError(f"Class {label!r} is not registered") from error

    def search(
        self,
        embedding: npt.ArrayLike,
        *,
        top_k: int = 1,
        min_similarity: float | None = None,
    ) -> list[PrototypeMatch]:
        """Find the most similar prototypes using cosine similarity."""
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if min_similarity is not None and not -1.0 <= min_similarity <= 1.0:
            raise ValueError("min_similarity must be in the range [-1, 1]")
        query = self.normalize_embeddings(embedding)
        if query.ndim != 1:
            raise ValueError("search expects exactly one embedding")

        with self._lock:
            if not self._prototypes:
                return []
            if query.shape[0] != self._embedding_dim:
                raise ValueError(
                    f"Expected embedding dimension {self._embedding_dim}, "
                    f"got {query.shape[0]}"
                )
            labels = list(self._prototypes)
            matrix = np.stack([self._prototypes[label] for label in labels])
            similarities = matrix @ query
            ranked_indices = np.argsort(-similarities, kind="stable")[:top_k]
            matches = []
            for index in ranked_indices:
                similarity = float(similarities[index])
                if min_similarity is None or similarity >= min_similarity:
                    label = labels[index]
                    matches.append(
                        PrototypeMatch(
                            label=label,
                            similarity=similarity,
                            metadata=copy.deepcopy(self._metadata[label]),
                        )
                    )
            return matches

    def save(self, path: str | Path) -> None:
        """Atomically save the registry as a compressed, pickle-free NPZ file."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            labels = list(self._prototypes)
            dimension = self._embedding_dim or 0
            prototypes = (
                np.stack([self._prototypes[label] for label in labels])
                if labels
                else np.empty((0, dimension), dtype=np.float32)
            )
            metadata = [self._metadata[label] for label in labels]
            metadata_json = json.dumps(
                metadata,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".npz",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
            np.savez_compressed(
                temporary_path,
                schema_version=np.asarray(self.SCHEMA_VERSION, dtype=np.int64),
                embedding_dim=np.asarray(dimension, dtype=np.int64),
                labels=np.asarray(labels, dtype=np.str_),
                prototypes=prototypes.astype(np.float32, copy=False),
                metadata_json=np.asarray(metadata_json, dtype=np.str_),
            )
            os.replace(temporary_path, destination)
        except (OSError, ValueError) as error:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise OSError(
                f"Could not save prototype registry to {destination}"
            ) from error

    @classmethod
    def load(cls, path: str | Path) -> PrototypeRegistry:
        """Load and validate a registry from a pickle-free NPZ file."""
        source = Path(path)
        try:
            with np.load(source, allow_pickle=False) as archive:
                required = {
                    "schema_version",
                    "embedding_dim",
                    "labels",
                    "prototypes",
                    "metadata_json",
                }
                if not required.issubset(archive.files):
                    missing = sorted(required.difference(archive.files))
                    raise ValueError(f"Registry is missing arrays: {missing}")
                schema_version = int(archive["schema_version"].item())
                dimension = int(archive["embedding_dim"].item())
                labels_array = archive["labels"]
                prototypes = np.asarray(archive["prototypes"], dtype=np.float32)
                metadata_json = str(archive["metadata_json"].item())
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise ValueError(
                f"Could not load prototype registry from {source}"
            ) from error

        if schema_version != cls.SCHEMA_VERSION:
            raise ValueError(f"Unsupported registry schema version {schema_version}")
        if labels_array.ndim != 1 or labels_array.dtype.kind not in {"U", "S"}:
            raise ValueError("Registry labels must be a one-dimensional string array")
        labels = [str(label) for label in labels_array.tolist()]
        if dimension < 0 or prototypes.shape != (len(labels), dimension):
            raise ValueError("Registry prototype dimensions are inconsistent")
        if not np.all(np.isfinite(prototypes)):
            raise ValueError("Registry prototypes contain non-finite values")
        try:
            parsed_metadata = json.loads(metadata_json)
        except (json.JSONDecodeError, TypeError) as error:
            raise ValueError("Registry metadata is not valid JSON") from error
        if not isinstance(parsed_metadata, list) or len(parsed_metadata) != len(labels):
            raise ValueError("Registry metadata does not match its labels")

        registry = cls(embedding_dim=dimension or None)
        for index, label in enumerate(labels):
            metadata_item = parsed_metadata[index]
            if not isinstance(metadata_item, dict):
                raise ValueError("Each registry metadata entry must be an object")
            cls._validate_label(label)
            if label in registry:
                raise ValueError(f"Registry contains duplicate label {label!r}")
            registry.add_class(label, prototypes[index], metadata=metadata_item)
        return registry

    @classmethod
    def _validate_label(cls, label: str) -> str:
        if not isinstance(label, str):
            raise TypeError("label must be a string")
        if not label.strip():
            raise ValueError("label must not be empty or whitespace")
        if len(label) > cls._MAX_LABEL_LENGTH:
            raise ValueError(
                f"label must be at most {cls._MAX_LABEL_LENGTH} characters"
            )
        if any(ord(character) < 32 for character in label):
            raise ValueError("label must not contain control characters")
        return label

    @staticmethod
    def _validate_metadata(
        metadata: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        value = {} if metadata is None else dict(metadata)
        try:
            encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
            decoded = json.loads(encoded)
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("metadata must be JSON-serializable and finite") from error
        if not isinstance(decoded, dict):
            raise ValueError("metadata must be a JSON object")
        return decoded
