"""Deterministic stratified splitting without copying source images."""

from __future__ import annotations

import csv
import math
import os
import random
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from data.manifest import DatasetRecord
from utils.image_validation import resolve_path_within_root

SPLIT_ORDER = ("train", "validation", "test")


class SplitError(ValueError):
    """Raised when deterministic split generation cannot proceed safely."""


@dataclass(frozen=True, slots=True)
class SplitResult:
    """Generated split records plus stratification warnings."""

    splits: Mapping[str, tuple[DatasetRecord, ...]]
    warnings: tuple[str, ...]


def stratified_split(
    records: Iterable[DatasetRecord],
    *,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> SplitResult:
    """Split labelled records deterministically while preserving classes when possible."""
    ratios = (train_ratio, validation_ratio, test_ratio)
    if any(not math.isfinite(ratio) or ratio < 0.0 for ratio in ratios):
        raise SplitError("Split ratios must be finite and non-negative")
    if not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise SplitError("Split ratios must sum to 1")
    nonzero_split_count = sum(ratio > 0.0 for ratio in ratios)
    if nonzero_split_count == 0:
        raise SplitError("At least one split ratio must be positive")

    records_list = list(records)
    if not records_list:
        raise SplitError("At least one record is required")
    seen_paths: set[Path] = set()
    by_label: dict[str, list[DatasetRecord]] = defaultdict(list)
    for record in records_list:
        resolved_path = record.image_path.resolve()
        if resolved_path in seen_paths:
            raise SplitError(f"Duplicate image path cannot be split: {resolved_path}")
        seen_paths.add(resolved_path)
        if not record.label:
            raise SplitError(f"Record has an empty label: {resolved_path}")
        by_label[record.label].append(record)

    generator = random.Random(random_seed)
    split_records: dict[str, list[DatasetRecord]] = {name: [] for name in SPLIT_ORDER}
    warnings: list[str] = []
    for label in sorted(by_label):
        class_records = sorted(by_label[label], key=lambda item: str(item.image_path))
        generator.shuffle(class_records)
        if len(class_records) < nonzero_split_count:
            warnings.append(
                f"Class {label!r} has {len(class_records)} sample(s), too few to "
                f"populate all {nonzero_split_count} non-empty splits"
            )
        counts = _allocate_counts(len(class_records), ratios)
        start = 0
        for split_name, count in zip(SPLIT_ORDER, counts, strict=True):
            for record in class_records[start : start + count]:
                split_records[split_name].append(
                    DatasetRecord(
                        image_path=record.image_path,
                        label=record.label,
                        split=split_name,
                        row_number=record.row_number,
                        source=record.source,
                    )
                )
            start += count

    for split_name in SPLIT_ORDER:
        split_records[split_name].sort(key=lambda item: str(item.image_path))
    _validate_no_overlap(split_records)
    return SplitResult(
        splits={name: tuple(split_records[name]) for name in SPLIT_ORDER},
        warnings=tuple(warnings),
    )


def save_split_manifest(
    result: SplitResult,
    *,
    dataset_root: str | Path,
    output_dir: str | Path,
    filename: str = "generated_splits.csv",
) -> Path:
    """Write relative paths to a generated CSV outside the source dataset."""
    root = Path(dataset_root).expanduser().resolve()
    destination_directory = Path(output_dir).expanduser().resolve()
    if destination_directory == root or destination_directory.is_relative_to(root):
        raise SplitError(
            f"Generated manifests must be outside dataset root {root}: "
            f"{destination_directory}"
        )
    if Path(filename).name != filename or Path(filename).suffix.lower() != ".csv":
        raise SplitError("filename must be a plain .csv filename")
    destination_directory.mkdir(parents=True, exist_ok=True)
    destination = destination_directory / filename

    rows: list[tuple[str, str, str]] = []
    for split_name in SPLIT_ORDER:
        for record in result.splits.get(split_name, ()):
            safe_path = resolve_path_within_root(root, record.image_path)
            if not safe_path.exists() or not safe_path.is_file():
                raise SplitError(f"Cannot write missing image to manifest: {safe_path}")
            rows.append(
                (safe_path.relative_to(root).as_posix(), record.label, split_name)
            )

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=destination_directory,
            prefix=f".{filename}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            writer = csv.writer(handle)
            writer.writerow(("image_path", "label", "split"))
            writer.writerows(rows)
        os.replace(temporary_path, destination)
    except OSError as error:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise SplitError(f"Could not save generated manifest {destination}") from error
    return destination


def _allocate_counts(total: int, ratios: tuple[float, float, float]) -> tuple[int, ...]:
    positive_indices = [index for index, ratio in enumerate(ratios) if ratio > 0.0]
    if total >= len(positive_indices):
        minimums = [
            1 if index in positive_indices else 0 for index in range(len(ratios))
        ]
        remaining = total - sum(minimums)
        if remaining == 0:
            return tuple(minimums)
        residual_weights = [
            max(0.0, total * ratio - minimums[index])
            for index, ratio in enumerate(ratios)
        ]
        weight_sum = sum(residual_weights)
        if weight_sum == 0.0:
            residual_weights = list(ratios)
            weight_sum = sum(residual_weights)
        exact = [remaining * weight / weight_sum for weight in residual_weights]
        counts = [
            minimum + math.floor(value)
            for minimum, value in zip(minimums, exact, strict=True)
        ]
        unassigned = total - sum(counts)
        priority = sorted(
            range(len(ratios)),
            key=lambda index: (-(exact[index] - math.floor(exact[index])), index),
        )
        for index in priority[:unassigned]:
            counts[index] += 1
        return tuple(counts)

    exact = [total * ratio for ratio in ratios]
    counts = [math.floor(value) for value in exact]
    remaining = total - sum(counts)
    priority = sorted(
        range(len(ratios)),
        key=lambda index: (-(exact[index] - counts[index]), index),
    )
    for index in priority[:remaining]:
        counts[index] += 1
    return tuple(counts)


def _validate_no_overlap(records: Mapping[str, list[DatasetRecord]]) -> None:
    owners: dict[Path, str] = {}
    for split_name, split_records in records.items():
        for record in split_records:
            path = record.image_path.resolve()
            if path in owners:
                raise SplitError(
                    f"Image path overlaps {owners[path]!r} and {split_name!r}: {path}"
                )
            owners[path] = split_name
