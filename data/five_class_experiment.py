"""Strict construction of the relocked five-class baseline manifests."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path


class FiveClassExperimentError(ValueError):
    """Raised when five-class artifacts cannot be derived unambiguously."""


@dataclass(frozen=True, slots=True)
class FiveClassSpec:
    """One fixed model index and its original Dataset A identity."""

    model_index: int
    class_name: str
    dataset_a_class_id: str
    expected_training_count: int
    expected_external_count: int


FIVE_CLASS_SPECS = (
    FiveClassSpec(0, "give_way", "0", 201, 18),
    FiveClassSpec(1, "no_entry", "1", 201, 14),
    FiveClassSpec(2, "no_right_turn", "16", 201, 11),
    FiveClassSpec(3, "road_hump", "30", 201, 53),
    FiveClassSpec(4, "filling_station", "55", 616, 21),
)
SPEC_BY_CLASS = {spec.class_name: spec for spec in FIVE_CLASS_SPECS}
SPEC_BY_DATASET_A_ID = {spec.dataset_a_class_id: spec for spec in FIVE_CLASS_SPECS}
CLASS_MAPPING = {spec.class_name: spec.model_index for spec in FIVE_CLASS_SPECS}


def build_five_class_training_pool(
    six_class_pool: str | Path,
) -> list[dict[str, str]]:
    """Filter the audited six-class pool to the five fixed Dataset A classes."""
    rows, columns = _read_csv(Path(six_class_pool))
    required = {
        "image_path",
        "class_name",
        "dataset_a_class_id",
        "source_template_id",
        "augmentation_family_id",
        "exact_content_sha256",
    }
    _require_columns(columns, required, "Dataset A pool")
    selected: list[dict[str, str]] = []
    for row in rows:
        class_id = row["dataset_a_class_id"]
        if class_id == "41":
            continue
        spec = SPEC_BY_DATASET_A_ID.get(class_id)
        if spec is None:
            raise FiveClassExperimentError(
                f"Unexpected Dataset A class ID in source pool: {class_id!r}"
            )
        if row["class_name"] != spec.class_name:
            raise FiveClassExperimentError(
                f"Dataset A class {class_id} has label {row['class_name']!r}; "
                f"expected {spec.class_name!r}"
            )
        selected.append(dict(row))
    _validate_counts(
        selected,
        key="class_name",
        expected={
            spec.class_name: spec.expected_training_count for spec in FIVE_CLASS_SPECS
        },
        description="Dataset A training",
    )
    if len({row["image_path"] for row in selected}) != len(selected):
        raise FiveClassExperimentError(
            "Dataset A training pool contains duplicate paths"
        )
    selected.sort(key=lambda row: (CLASS_MAPPING[row["class_name"]], row["image_path"]))
    return selected


def build_five_class_external_test(
    reviewed_external_manifest: str | Path,
) -> list[dict[str, str]]:
    """Validate the approved external rows against the five-class vocabulary."""
    rows, columns = _read_csv(Path(reviewed_external_manifest))
    required = {
        "image_path",
        "class_name",
        "dataset_a_class_id",
        "source_image_id",
        "perceptual_group_id",
        "review_id",
    }
    _require_columns(columns, required, "Dataset B external manifest")
    selected: list[dict[str, str]] = []
    for row in rows:
        label = row["class_name"]
        if label == "y_junction" or row["dataset_a_class_id"] == "41":
            raise FiveClassExperimentError(
                "Reviewed external manifest still contains a Y-junction row"
            )
        spec = SPEC_BY_CLASS.get(label)
        if spec is None:
            raise FiveClassExperimentError(
                f"External manifest contains non-baseline class {label!r}"
            )
        if row["dataset_a_class_id"] != spec.dataset_a_class_id:
            raise FiveClassExperimentError(
                f"External label {label!r} maps to Dataset A ID "
                f"{row['dataset_a_class_id']!r}, expected {spec.dataset_a_class_id!r}"
            )
        selected.append(dict(row))
    _validate_counts(
        selected,
        key="class_name",
        expected={
            spec.class_name: spec.expected_external_count for spec in FIVE_CLASS_SPECS
        },
        description="Dataset B external-test",
    )
    for field in ("image_path", "source_image_id", "review_id"):
        if len({row[field] for row in selected}) != len(selected):
            raise FiveClassExperimentError(
                f"Dataset B external manifest contains duplicate {field} values"
            )
    selected.sort(
        key=lambda row: (CLASS_MAPPING[row["class_name"]], row["source_image_id"])
    )
    return selected


def write_five_class_artifacts(
    training_rows: list[dict[str, str]],
    external_rows: list[dict[str, str]],
    *,
    training_manifest: str | Path,
    class_mapping_path: str | Path,
    external_manifest: str | Path,
) -> tuple[Path, Path, Path]:
    """Write all relocked manifests while refusing to overwrite prior results."""
    destinations = tuple(
        Path(value).expanduser().resolve()
        for value in (training_manifest, class_mapping_path, external_manifest)
    )
    existing = [str(path) for path in destinations if path.exists()]
    if existing:
        raise FiveClassExperimentError(
            f"Refusing to overwrite five-class artifacts: {existing}"
        )
    _write_csv_atomic(destinations[0], training_rows, tuple(training_rows[0]))
    _write_text_atomic(
        destinations[1],
        json.dumps(CLASS_MAPPING, ensure_ascii=False, indent=2) + "\n",
    )
    _write_csv_atomic(destinations[2], external_rows, tuple(external_rows[0]))
    return destinations[0], destinations[1], destinations[2]


def _validate_counts(
    rows: list[dict[str, str]],
    *,
    key: str,
    expected: dict[str, int],
    description: str,
) -> None:
    actual = {label: 0 for label in expected}
    for row in rows:
        label = row[key]
        actual[label] = actual.get(label, 0) + 1
    if actual != expected:
        raise FiveClassExperimentError(
            f"{description} counts differ; expected={expected}, actual={actual}"
        )


def _read_csv(path: Path) -> tuple[list[dict[str, str]], tuple[str, ...]]:
    try:
        with (
            path.expanduser()
            .resolve()
            .open("r", encoding="utf-8-sig", newline="") as handle
        ):
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise FiveClassExperimentError(f"CSV has no header: {path}")
            return [dict(row) for row in reader], tuple(reader.fieldnames)
    except (OSError, csv.Error) as error:
        raise FiveClassExperimentError(f"Could not read CSV: {path}") from error


def _require_columns(
    columns: tuple[str, ...], required: set[str], description: str
) -> None:
    missing = sorted(required.difference(columns))
    if missing:
        raise FiveClassExperimentError(
            f"{description} is missing required columns: {missing}"
        )


def _write_csv_atomic(
    path: Path, rows: list[dict[str, str]], fieldnames: tuple[str, ...]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    except OSError:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
        os.replace(temporary, path)
    except OSError:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
