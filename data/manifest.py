"""Manifest parsing and filesystem record discovery."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from utils.image_validation import (
    DEFAULT_IMAGE_EXTENSIONS,
    ImageValidationError,
    normalize_extensions,
    resolve_path_within_root,
    validate_image_path,
)

SPLIT_ALIASES: dict[str, str] = {
    "train": "train",
    "training": "train",
    "val": "validation",
    "valid": "validation",
    "validation": "validation",
    "test": "test",
    "testing": "test",
}


class ManifestError(ValueError):
    """Raised when a dataset manifest is malformed or unsafe."""


@dataclass(frozen=True, slots=True)
class DatasetRecord:
    """A validated reference to one labelled image sample."""

    image_path: Path
    label: str
    split: str | None = None
    row_number: int | None = None
    source: str | None = None
    review_id: str | None = None

    def relative_path(self, root: str | Path) -> str:
        """Return this image path relative to a dataset root using POSIX separators."""
        return self.image_path.resolve().relative_to(Path(root).resolve()).as_posix()


def canonical_split(value: str | None) -> str | None:
    """Normalize common split names while preserving unknown explicit names."""
    if value is None or not value.strip():
        return None
    normalized = value.strip().lower()
    return SPLIT_ALIASES.get(normalized, normalized)


def load_manifest(
    manifest_path: str | Path,
    *,
    dataset_root: str | Path,
    image_path_column: str = "image_path",
    label_column: str = "label",
    split_column: str | None = "split",
    requested_split: str | None = None,
    allowed_extensions: Sequence[str] = DEFAULT_IMAGE_EXTENSIONS,
    require_files: bool = True,
    reject_duplicate_paths: bool = True,
    allow_unsupported_extensions: bool = False,
) -> list[DatasetRecord]:
    """Load CSV or JSON dataset records with row-aware validation.

    JSON manifests may be a list of objects or an object containing a ``samples``
    list. Image paths may be absolute or relative, but must resolve inside the
    configured dataset root.
    """
    path = Path(manifest_path).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise ManifestError(f"Manifest file does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        rows = _read_csv_rows(path)
    elif suffix == ".json":
        rows = _read_json_rows(path)
    else:
        raise ManifestError(f"Unsupported manifest extension {suffix!r}: {path}")

    required_columns = {image_path_column, label_column}
    requested = canonical_split(requested_split)
    records: list[DatasetRecord] = []
    seen_paths: dict[Path, int] = {}
    normalized_extensions = normalize_extensions(list(allowed_extensions))
    for row_number, row in rows:
        missing_columns = sorted(required_columns.difference(row))
        if missing_columns:
            raise ManifestError(
                f"Manifest {path}, row {row_number}: missing columns {missing_columns}"
            )
        raw_image_path = row.get(image_path_column)
        raw_label = row.get(label_column)
        if not isinstance(raw_image_path, str) or not raw_image_path.strip():
            raise ManifestError(
                f"Manifest {path}, row {row_number}: image path must be non-empty"
            )
        if not isinstance(raw_label, str) or not raw_label.strip():
            raise ManifestError(
                f"Manifest {path}, row {row_number}: label must be non-empty"
            )
        if raw_label != raw_label.strip():
            raise ManifestError(
                f"Manifest {path}, row {row_number}: label must not have "
                "leading or trailing whitespace"
            )
        raw_split = row.get(split_column) if split_column else None
        if raw_split is not None and not isinstance(raw_split, str):
            raise ManifestError(
                f"Manifest {path}, row {row_number}: split must be a string"
            )
        split = canonical_split(raw_split)
        raw_review_id = row.get("review_id")
        if raw_review_id is not None and not isinstance(raw_review_id, str):
            raise ManifestError(
                f"Manifest {path}, row {row_number}: review_id must be a string"
            )
        review_id = raw_review_id.strip() if raw_review_id else None
        if requested is not None and split != requested:
            continue
        try:
            if allow_unsupported_extensions:
                image_path = resolve_path_within_root(dataset_root, raw_image_path)
                if require_files and (
                    not image_path.exists() or not image_path.is_file()
                ):
                    raise ImageValidationError(
                        f"Image file does not exist: {image_path}"
                    )
            else:
                image_path = validate_image_path(
                    raw_image_path,
                    root=dataset_root,
                    allowed_extensions=normalized_extensions,
                    require_exists=require_files,
                )
        except ImageValidationError as error:
            raise ManifestError(
                f"Manifest {path}, row {row_number}: {error}"
            ) from error
        if image_path in seen_paths and reject_duplicate_paths:
            first_row = seen_paths[image_path]
            raise ManifestError(
                f"Manifest {path}, row {row_number}: duplicate image path; "
                f"first seen at row {first_row}: {image_path}"
            )
        seen_paths.setdefault(image_path, row_number)
        records.append(
            DatasetRecord(
                image_path=image_path,
                label=raw_label,
                split=split,
                row_number=row_number,
                source=str(path),
                review_id=review_id,
            )
        )
    return records


def discover_directory_records(
    dataset_root: str | Path,
    *,
    split: str | None = None,
    allowed_extensions: Sequence[str] = DEFAULT_IMAGE_EXTENSIONS,
) -> tuple[list[DatasetRecord], list[Path], list[Path]]:
    """Discover labelled records from class or split/class directories.

    Returns records, empty class directories, and unsupported files. Hidden files
    and directories are ignored.
    """
    root = Path(dataset_root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise ManifestError(
            f"Dataset root does not exist or is not a directory: {root}"
        )
    canonical_requested_split = canonical_split(split)
    scan_root = root
    discovered_split: str | None = None
    split_directories: dict[str | None, Path] = {}
    for child in root.iterdir():
        canonical_name = canonical_split(child.name)
        if (
            child.is_dir()
            and not child.name.startswith(".")
            and canonical_name in {"train", "validation", "test"}
        ):
            if canonical_name in split_directories:
                raise ManifestError(
                    f"Multiple directories map to split {canonical_name!r}: "
                    f"{split_directories[canonical_name]} and {child}"
                )
            split_directories[canonical_name] = child
    if split_directories:
        if canonical_requested_split is None:
            raise ManifestError(
                "Dataset has split directories; a requested split is required"
            )
        if canonical_requested_split not in split_directories:
            raise ManifestError(
                f"Requested split {canonical_requested_split!r} was not found in {root}"
            )
        scan_root = split_directories[canonical_requested_split]
        discovered_split = canonical_requested_split
    elif canonical_requested_split is not None:
        discovered_split = canonical_requested_split

    extensions = normalize_extensions(list(allowed_extensions))
    class_directories = sorted(
        (
            child
            for child in scan_root.iterdir()
            if child.is_dir() and not child.name.startswith(".")
        ),
        key=lambda child: child.name,
    )
    if not class_directories:
        raise ManifestError(f"No class directories found in {scan_root}")

    records: list[DatasetRecord] = []
    empty_directories: list[Path] = []
    unsupported_files: list[Path] = []
    for class_directory in class_directories:
        class_records: list[DatasetRecord] = []
        for file_path in sorted(class_directory.rglob("*")):
            if not file_path.is_file() or any(
                part.startswith(".") for part in file_path.relative_to(root).parts
            ):
                continue
            if file_path.suffix.lower() not in extensions:
                unsupported_files.append(file_path)
                continue
            class_records.append(
                DatasetRecord(
                    image_path=file_path.resolve(),
                    label=class_directory.name,
                    split=discovered_split,
                    source="directory",
                )
            )
        if not class_records:
            empty_directories.append(class_directory.resolve())
        records.extend(class_records)
    return records, empty_directories, unsupported_files


def load_class_mapping(path: str | Path) -> dict[str, int]:
    """Load a JSON label-to-index mapping and reject duplicate labels or indices."""
    mapping_path = Path(path).expanduser().resolve()
    try:
        pairs = json.loads(
            mapping_path.read_text(encoding="utf-8"), object_pairs_hook=list
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestError(f"Could not load class mapping {mapping_path}") from error
    if not isinstance(pairs, list) or any(
        not isinstance(pair, tuple) or len(pair) != 2 for pair in pairs
    ):
        raise ManifestError("Class mapping must be a JSON object")
    labels = [pair[0] for pair in pairs]
    if len(labels) != len(set(labels)):
        raise ManifestError("Class mapping contains duplicate labels")
    mapping = dict(pairs)
    _validate_class_mapping(mapping)
    return mapping


def validate_class_mapping(mapping: Mapping[str, int]) -> dict[str, int]:
    """Validate and copy an explicit label-to-index mapping."""
    copied = dict(mapping)
    _validate_class_mapping(copied)
    return copied


def _validate_class_mapping(mapping: Mapping[str, Any]) -> None:
    if not mapping:
        raise ManifestError("Class mapping must not be empty")
    if any(not isinstance(label, str) or not label.strip() for label in mapping):
        raise ManifestError("Class mapping labels must be non-empty strings")
    indices = list(mapping.values())
    if any(isinstance(index, bool) or not isinstance(index, int) for index in indices):
        raise ManifestError("Class mapping indices must be integers")
    if len(indices) != len(set(indices)):
        raise ManifestError("Class mapping contains duplicate indices")
    if sorted(indices) != list(range(len(indices))):
        raise ManifestError("Class mapping indices must be contiguous from zero")


def _read_csv_rows(path: Path) -> list[tuple[int, Mapping[str, Any]]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ManifestError(f"CSV manifest has no header: {path}")
            return [(row_number, row) for row_number, row in enumerate(reader, start=2)]
    except (OSError, csv.Error) as error:
        raise ManifestError(f"Could not read CSV manifest {path}") from error


def _read_json_rows(path: Path) -> list[tuple[int, Mapping[str, Any]]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestError(f"Could not read JSON manifest {path}") from error
    if isinstance(payload, dict):
        payload = payload.get("samples")
    if not isinstance(payload, list):
        raise ManifestError(
            f"JSON manifest must be a list or contain a samples list: {path}"
        )
    rows: list[tuple[int, Mapping[str, Any]]] = []
    for row_number, row in enumerate(payload, start=1):
        if not isinstance(row, dict):
            raise ManifestError(
                f"JSON manifest {path}, item {row_number}: expected an object"
            )
        rows.append((row_number, row))
    return rows
