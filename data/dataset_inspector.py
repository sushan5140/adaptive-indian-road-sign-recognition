"""Read-only inspection for common road-sign dataset layouts."""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from data.manifest import (
    DatasetRecord,
    ManifestError,
    canonical_split,
    load_manifest,
)
from utils.image_validation import (
    DEFAULT_IMAGE_EXTENSIONS,
    ImageValidationError,
    inspect_image,
    normalize_extensions,
)

ANNOTATION_EXTENSIONS = {".csv", ".json", ".xml", ".yaml", ".yml", ".txt"}
KNOWN_SPLITS = {"train", "validation", "test"}


class DatasetInspectionError(ValueError):
    """Raised when inspection cannot start because configuration is invalid."""


@dataclass(frozen=True, slots=True)
class DimensionSummary:
    """Minimum, maximum, and median decoded image dimensions."""

    minimum: tuple[int, int] | None
    maximum: tuple[int, int] | None
    median: tuple[float, float] | None


@dataclass(frozen=True, slots=True)
class DatasetInspectionReport:
    """Serializable results of a read-only dataset inspection."""

    dataset_root: str
    detected_mode: str
    directory_structure: tuple[str, ...]
    supported_image_extensions: tuple[str, ...]
    total_image_count: int
    verified_image_count: int
    number_of_classes: int
    class_names: tuple[str, ...]
    samples_per_class: dict[str, int]
    possible_splits: tuple[str, ...]
    unreadable_files: tuple[str, ...]
    duplicate_file_paths: tuple[str, ...]
    empty_class_directories: tuple[str, ...]
    unsupported_files: tuple[str, ...]
    class_imbalance_exists: bool
    image_dimensions: DimensionSummary
    colour_modes: dict[str, int]
    possible_annotation_files: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of this report."""
        return asdict(self)


class DatasetInspector:
    """Inspect dataset layout and image health without changing source files."""

    SUPPORTED_MODES = {
        "auto",
        "directory",
        "split_directory",
        "flat",
        "csv_manifest",
        "json_manifest",
        "manifest",
    }

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        mode: str = "auto",
        manifest_path: str | Path | None = None,
        image_path_column: str = "image_path",
        label_column: str = "label",
        split_column: str | None = "split",
        allowed_extensions: Sequence[str] = DEFAULT_IMAGE_EXTENSIONS,
        verify_images: bool = True,
        max_images: int | None = None,
        imbalance_ratio: float = 1.5,
    ) -> None:
        self.root = Path(dataset_root).expanduser().resolve()
        self.mode = mode.strip().lower()
        self.manifest_path = manifest_path
        self.image_path_column = image_path_column
        self.label_column = label_column
        self.split_column = split_column
        self.allowed_extensions = normalize_extensions(list(allowed_extensions))
        self.verify_images = verify_images
        self.max_images = max_images
        self.imbalance_ratio = imbalance_ratio
        if self.mode not in self.SUPPORTED_MODES:
            raise DatasetInspectionError(
                f"Unsupported inspection mode {mode!r}; "
                f"expected one of {sorted(self.SUPPORTED_MODES)}"
            )
        if max_images is not None and max_images <= 0:
            raise DatasetInspectionError("max_images must be positive when provided")
        if imbalance_ratio < 1.0:
            raise DatasetInspectionError("imbalance_ratio must be at least 1")

    def inspect(self) -> DatasetInspectionReport:
        """Inspect configured paths and return a complete read-only report."""
        if not self.root.exists() or not self.root.is_dir():
            raise DatasetInspectionError(
                f"Dataset root does not exist or is not a directory: {self.root}"
            )
        mode = self._detect_mode()
        all_files = self._visible_files()
        annotation_files = sorted(
            file_path
            for file_path in all_files
            if file_path.suffix.lower() in ANNOTATION_EXTENSIONS
        )
        unsupported_files = sorted(
            file_path
            for file_path in all_files
            if file_path.suffix.lower() not in self.allowed_extensions
            and file_path.suffix.lower() not in ANNOTATION_EXTENSIONS
        )
        warnings: list[str] = []
        if mode in {"csv_manifest", "json_manifest", "manifest"}:
            records = self._manifest_records(mode)
            empty_class_directories: list[Path] = []
        else:
            records, empty_class_directories, structural_warnings = (
                self._filesystem_records(mode)
            )
            warnings.extend(structural_warnings)

        path_counts = Counter(record.image_path.resolve() for record in records)
        duplicate_paths = sorted(
            path for path, count in path_counts.items() if count > 1
        )
        unique_records = list(
            {record.image_path.resolve(): record for record in records}.values()
        )
        class_counts = Counter(record.label for record in records if record.label)
        for empty_directory in empty_class_directories:
            class_counts.setdefault(empty_directory.name, 0)
        class_names = tuple(sorted(class_counts))
        splits = tuple(
            name
            for name in ("train", "validation", "test")
            if any(record.split == name for record in records)
        )
        other_splits = sorted(
            {
                record.split
                for record in records
                if record.split is not None and record.split not in KNOWN_SPLITS
            }
        )
        splits += tuple(other_splits)

        unreadable: list[Path] = []
        widths: list[int] = []
        heights: list[int] = []
        colour_modes: Counter[str] = Counter()
        candidates = unique_records
        if self.max_images is not None:
            candidates = candidates[: self.max_images]
            if len(unique_records) > len(candidates):
                warnings.append(
                    f"Image verification limited to {len(candidates)} of "
                    f"{len(unique_records)} unique image paths"
                )
        verified_count = 0
        for record in candidates:
            if not record.image_path.exists() or not record.image_path.is_file():
                unreadable.append(record.image_path)
                continue
            if not self.verify_images:
                continue
            verified_count += 1
            try:
                info = inspect_image(record.image_path)
            except ImageValidationError:
                unreadable.append(record.image_path)
                continue
            widths.append(info.width)
            heights.append(info.height)
            colour_modes[info.colour_mode] += 1
        if not self.verify_images:
            warnings.append(
                "Image decoding was disabled; dimensions and modes are unknown"
            )

        class_imbalance = self._has_class_imbalance(
            class_counts, empty_class_directories
        )
        return DatasetInspectionReport(
            dataset_root=str(self.root),
            detected_mode=mode,
            directory_structure=self._directory_structure(),
            supported_image_extensions=self.allowed_extensions,
            total_image_count=len(records),
            verified_image_count=verified_count,
            number_of_classes=len(class_names),
            class_names=class_names,
            samples_per_class=dict(sorted(class_counts.items())),
            possible_splits=splits,
            unreadable_files=tuple(str(path) for path in sorted(set(unreadable))),
            duplicate_file_paths=tuple(str(path) for path in duplicate_paths),
            empty_class_directories=tuple(
                str(path) for path in sorted(empty_class_directories)
            ),
            unsupported_files=tuple(str(path) for path in unsupported_files),
            class_imbalance_exists=class_imbalance,
            image_dimensions=self._dimension_summary(widths, heights),
            colour_modes=dict(sorted(colour_modes.items())),
            possible_annotation_files=tuple(str(path) for path in annotation_files),
            warnings=tuple(warnings),
        )

    def _detect_mode(self) -> str:
        if self.mode != "auto":
            return self.mode
        if self.manifest_path is not None:
            suffix = Path(self.manifest_path).suffix.lower()
            if suffix == ".csv":
                return "csv_manifest"
            if suffix == ".json":
                return "json_manifest"
            raise DatasetInspectionError(
                f"Cannot infer manifest mode from extension {suffix!r}"
            )
        children = [
            child for child in self.root.iterdir() if not child.name.startswith(".")
        ]
        if any(
            child.is_dir() and canonical_split(child.name) in KNOWN_SPLITS
            for child in children
        ):
            return "split_directory"
        if any(child.is_dir() for child in children):
            return "directory"
        if any(
            child.is_file() and child.suffix.lower() in self.allowed_extensions
            for child in children
        ):
            return "flat"
        return "flat"

    def _manifest_records(self, mode: str) -> list[DatasetRecord]:
        if self.manifest_path is None:
            raise DatasetInspectionError(f"Mode {mode!r} requires manifest_path")
        path = Path(self.manifest_path).expanduser()
        if not path.is_absolute():
            path = self.root / path
        expected_suffix = {
            "csv_manifest": ".csv",
            "json_manifest": ".json",
        }.get(mode)
        if expected_suffix and path.suffix.lower() != expected_suffix:
            raise DatasetInspectionError(
                f"Mode {mode!r} requires a {expected_suffix} manifest"
            )
        try:
            return load_manifest(
                path,
                dataset_root=self.root,
                image_path_column=self.image_path_column,
                label_column=self.label_column,
                split_column=self.split_column,
                requested_split=None,
                allowed_extensions=self.allowed_extensions,
                require_files=False,
                reject_duplicate_paths=False,
                allow_unsupported_extensions=True,
            )
        except ManifestError as error:
            raise DatasetInspectionError(str(error)) from error

    def _filesystem_records(
        self, mode: str
    ) -> tuple[list[DatasetRecord], list[Path], list[str]]:
        image_files = sorted(
            file_path
            for file_path in self._visible_files()
            if file_path.suffix.lower() in self.allowed_extensions
        )
        records: list[DatasetRecord] = []
        warnings: list[str] = []
        for image_path in image_files:
            relative_parts = image_path.relative_to(self.root).parts
            label = ""
            split: str | None = None
            if mode == "directory" and len(relative_parts) >= 2:
                label = relative_parts[0]
            elif mode == "split_directory" and len(relative_parts) >= 3:
                split = canonical_split(relative_parts[0])
                label = relative_parts[1]
            elif mode != "flat":
                warnings.append(
                    f"Could not infer a class label from {image_path.relative_to(self.root)}"
                )
            records.append(
                DatasetRecord(
                    image_path=image_path.resolve(),
                    label=label,
                    split=split,
                    source="directory",
                )
            )
        empty_directories = self._find_empty_class_directories(mode)
        return records, empty_directories, warnings

    def _find_empty_class_directories(self, mode: str) -> list[Path]:
        if mode == "directory":
            candidates = [
                child
                for child in self.root.iterdir()
                if child.is_dir() and not child.name.startswith(".")
            ]
        elif mode == "split_directory":
            candidates = []
            for split_directory in self.root.iterdir():
                if (
                    split_directory.is_dir()
                    and canonical_split(split_directory.name) in KNOWN_SPLITS
                ):
                    candidates.extend(
                        child
                        for child in split_directory.iterdir()
                        if child.is_dir() and not child.name.startswith(".")
                    )
        else:
            return []
        return [
            directory.resolve()
            for directory in candidates
            if not any(
                file_path.is_file()
                and file_path.suffix.lower() in self.allowed_extensions
                for file_path in directory.rglob("*")
            )
        ]

    def _visible_files(self) -> list[Path]:
        return sorted(
            path
            for path in self.root.rglob("*")
            if path.is_file()
            and not any(
                part.startswith(".") for part in path.relative_to(self.root).parts
            )
        )

    def _directory_structure(self) -> tuple[str, ...]:
        directories = ["."]
        directories.extend(
            path.relative_to(self.root).as_posix()
            for path in sorted(self.root.rglob("*"))
            if path.is_dir()
            and not any(
                part.startswith(".") for part in path.relative_to(self.root).parts
            )
        )
        return tuple(directories)

    def _has_class_imbalance(
        self, counts: Counter[str], empty_directories: list[Path]
    ) -> bool:
        if empty_directories and counts:
            return True
        positive_counts = [count for count in counts.values() if count > 0]
        if len(positive_counts) < 2:
            return False
        return max(positive_counts) / min(positive_counts) > self.imbalance_ratio

    @staticmethod
    def _dimension_summary(widths: list[int], heights: list[int]) -> DimensionSummary:
        if not widths:
            return DimensionSummary(minimum=None, maximum=None, median=None)
        return DimensionSummary(
            minimum=(min(widths), min(heights)),
            maximum=(max(widths), max(heights)),
            median=(
                float(statistics.median(widths)),
                float(statistics.median(heights)),
            ),
        )
