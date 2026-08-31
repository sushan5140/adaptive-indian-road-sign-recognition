"""Configurable PyTorch-compatible road-sign dataset adapter."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from data.manifest import (
    DatasetRecord,
    ManifestError,
    discover_directory_records,
    load_class_mapping,
    load_manifest,
    validate_class_mapping,
)
from utils.image_validation import (
    DEFAULT_IMAGE_EXTENSIONS,
    ImageValidationError,
    decode_image,
    extended_length_path,
    inspect_image,
    normalize_extensions,
)

if TYPE_CHECKING:
    from torch.utils.data import Dataset as _TorchDataset
else:
    try:
        from torch.utils.data import Dataset as _TorchDataset
    except ImportError:  # pragma: no cover - only without the declared dependency
        _TorchDataset = object


class DatasetConfigurationError(ValueError):
    """Raised when dataset configuration or discovered samples are invalid."""


@dataclass(frozen=True, slots=True)
class RoadSignDatasetConfig:
    """Configuration for :class:`RoadSignDataset`."""

    root: str | Path
    mode: str = "auto"
    manifest_path: str | Path | None = None
    image_path_column: str = "image_path"
    label_column: str = "label"
    split_column: str | None = "split"
    split: str | None = "train"
    allowed_extensions: Sequence[str] = DEFAULT_IMAGE_EXTENSIONS
    convert_to_rgb: bool = True
    corrupt_image_policy: str = "error"
    unknown_label_policy: str = "error"
    return_metadata: bool = False
    class_mapping_path: str | Path | None = None


class RoadSignDataset(_TorchDataset[Any]):
    """Read road-sign samples without changing source files.

    When PyTorch is installed this class subclasses ``torch.utils.data.Dataset``.
    Images are returned as NumPy arrays unless the caller's transform converts
    them to another representation.
    """

    SUPPORTED_MODES = {
        "auto",
        "directory",
        "split_directory",
        "csv_manifest",
        "json_manifest",
        "manifest",
    }

    def __init__(
        self,
        config: RoadSignDatasetConfig,
        *,
        transform: Callable[[Any], Any] | None = None,
        class_mapping: Mapping[str, int] | None = None,
    ) -> None:
        self.config = config
        self.transform = transform
        self.root = extended_length_path(config.root)
        if not self.root.exists() or not self.root.is_dir():
            raise DatasetConfigurationError(
                f"Dataset root does not exist or is not a directory: {self.root}"
            )
        mode = config.mode.strip().lower()
        if mode not in self.SUPPORTED_MODES:
            raise DatasetConfigurationError(
                f"Unsupported dataset mode {config.mode!r}; "
                f"expected one of {sorted(self.SUPPORTED_MODES)}"
            )
        if config.corrupt_image_policy not in {"error", "skip"}:
            raise DatasetConfigurationError(
                "corrupt_image_policy must be 'error' or 'skip'"
            )
        if config.unknown_label_policy not in {"error", "skip"}:
            raise DatasetConfigurationError(
                "unknown_label_policy must be 'error' or 'skip'"
            )
        self.allowed_extensions = normalize_extensions(list(config.allowed_extensions))
        resolved_mode = self._resolve_mode(mode)
        records = self._load_records(resolved_mode)
        if not records:
            raise DatasetConfigurationError(
                f"No samples found for mode {resolved_mode!r} and split {config.split!r}"
            )

        explicit_mapping = self._resolve_mapping(class_mapping)
        if explicit_mapping is None:
            labels = sorted({record.label for record in records})
            self.class_to_index = {label: index for index, label in enumerate(labels)}
        else:
            self.class_to_index = explicit_mapping
        self.index_to_class = {
            index: label for label, index in self.class_to_index.items()
        }

        mapped_records: list[DatasetRecord] = []
        for record in records:
            if record.label not in self.class_to_index:
                context = self._record_context(record)
                if config.unknown_label_policy == "skip":
                    continue
                raise DatasetConfigurationError(
                    f"Unknown label {record.label!r} is absent from the explicit "
                    f"mapping ({context})"
                )
            try:
                inspect_image(record.image_path)
            except ImageValidationError as error:
                if config.corrupt_image_policy == "skip":
                    continue
                raise DatasetConfigurationError(
                    f"Invalid image ({self._record_context(record)}): {error}"
                ) from error
            mapped_records.append(record)
        if not mapped_records:
            raise DatasetConfigurationError(
                "No usable samples remain after applying dataset policies"
            )
        self.records = tuple(mapped_records)

    def __len__(self) -> int:
        """Return the number of usable samples."""
        return len(self.records)

    def __getitem__(
        self, index: int
    ) -> tuple[Any, int] | tuple[Any, int, dict[str, Any]]:
        """Load one image and return its deterministic class index."""
        record = self.records[index]
        try:
            image: Any = decode_image(
                record.image_path,
                convert_to_rgb=self.config.convert_to_rgb,
            )
        except ImageValidationError as error:
            raise DatasetConfigurationError(
                f"Image became unreadable ({self._record_context(record)}): {error}"
            ) from error
        if self.transform is not None:
            image = self.transform(image)
        class_index = self.class_to_index[record.label]
        if not self.config.return_metadata:
            return image, class_index
        metadata: dict[str, Any] = {
            "image_path": str(record.image_path),
            "relative_image_path": record.relative_path(self.root),
            "label": record.label,
            "split": record.split or "",
            "row_number": record.row_number if record.row_number is not None else -1,
            "source": record.source or "",
            "review_id": record.review_id or "",
        }
        return image, class_index, metadata

    def _resolve_mode(self, mode: str) -> str:
        if mode != "auto":
            return mode
        if self.config.manifest_path is not None:
            suffix = Path(self.config.manifest_path).suffix.lower()
            if suffix == ".csv":
                return "csv_manifest"
            if suffix == ".json":
                return "json_manifest"
            raise DatasetConfigurationError(
                f"Cannot infer manifest mode from extension {suffix!r}"
            )
        split_names = {
            child.name.lower() for child in self.root.iterdir() if child.is_dir()
        }
        if split_names.intersection(
            {"train", "training", "val", "valid", "validation", "test", "testing"}
        ):
            return "split_directory"
        if any(child.is_dir() for child in self.root.iterdir()):
            return "directory"
        raise DatasetConfigurationError(
            "Could not infer dataset mode; configure a manifest or class directories"
        )

    def _load_records(self, mode: str) -> list[DatasetRecord]:
        if mode in {"directory", "split_directory"}:
            requested_split = self.config.split if mode == "split_directory" else None
            try:
                records, empty_directories, unsupported_files = (
                    discover_directory_records(
                        self.root,
                        split=requested_split,
                        allowed_extensions=self.allowed_extensions,
                    )
                )
            except ManifestError as error:
                raise DatasetConfigurationError(str(error)) from error
            if unsupported_files:
                formatted = ", ".join(str(path) for path in unsupported_files)
                raise DatasetConfigurationError(
                    f"Unsupported files found in class directories: {formatted}"
                )
            if empty_directories:
                formatted = ", ".join(str(path) for path in empty_directories)
                raise DatasetConfigurationError(f"Empty class directories: {formatted}")
            return records

        if mode not in {"manifest", "csv_manifest", "json_manifest"}:
            raise DatasetConfigurationError(f"Unsupported resolved mode {mode!r}")
        if self.config.manifest_path is None:
            raise DatasetConfigurationError(f"Mode {mode!r} requires manifest_path")
        manifest_path = Path(self.config.manifest_path).expanduser()
        if not manifest_path.is_absolute():
            manifest_path = self.root / manifest_path
        expected_suffix = {
            "csv_manifest": ".csv",
            "json_manifest": ".json",
        }.get(mode)
        if expected_suffix and manifest_path.suffix.lower() != expected_suffix:
            raise DatasetConfigurationError(
                f"Mode {mode!r} requires a {expected_suffix} manifest"
            )
        try:
            return load_manifest(
                manifest_path,
                dataset_root=self.root,
                image_path_column=self.config.image_path_column,
                label_column=self.config.label_column,
                split_column=self.config.split_column,
                requested_split=self.config.split,
                allowed_extensions=self.allowed_extensions,
                require_files=True,
            )
        except ManifestError as error:
            raise DatasetConfigurationError(str(error)) from error

    def _resolve_mapping(
        self, mapping: Mapping[str, int] | None
    ) -> dict[str, int] | None:
        if mapping is not None and self.config.class_mapping_path is not None:
            raise DatasetConfigurationError(
                "Provide class_mapping or class_mapping_path, not both"
            )
        try:
            if mapping is not None:
                return validate_class_mapping(mapping)
            if self.config.class_mapping_path is not None:
                return load_class_mapping(self.config.class_mapping_path)
        except ManifestError as error:
            raise DatasetConfigurationError(str(error)) from error
        return None

    @staticmethod
    def _record_context(record: DatasetRecord) -> str:
        row_context = (
            f", row {record.row_number}" if record.row_number is not None else ""
        )
        return f"file {record.image_path}{row_context}"
