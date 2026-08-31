"""Deterministic preparation of the reviewed six-class baseline experiment."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps

from utils.image_validation import extended_length_path


class ExperimentPreparationError(ValueError):
    """Raised when experiment artifacts cannot be derived without ambiguity."""


@dataclass(frozen=True, slots=True)
class SixClassSpec:
    """Exact semantic alignment between one Dataset B class and Dataset A."""

    dataset_b_class: str
    dataset_a_class_id: str
    dataset_a_class_name: str
    source_template_id: str


SIX_CLASS_SPECS = (
    SixClassSpec("filling_station", "55", "Filling station", "98"),
    SixClassSpec("give_way", "0", "Give way", "43"),
    SixClassSpec("no_entry", "1", "No entry", "44"),
    SixClassSpec("no_right_turn", "16", "No right turn", "59"),
    SixClassSpec("road_hump", "30", "Road hump", "73"),
    SixClassSpec("y_junction", "41", "Y-junction", "84"),
)
SPECS_BY_B_CLASS = {spec.dataset_b_class: spec for spec in SIX_CLASS_SPECS}
SPECS_BY_A_ID = {spec.dataset_a_class_id: spec for spec in SIX_CLASS_SPECS}

REVIEW_COLUMNS = (
    "review_id",
    "image_path",
    "source_image_id",
    "proposed_class",
    "dataset_a_class_id",
    "dataset_a_class_name",
    "source_question",
    "source_answer",
    "perceptual_group_id",
    "review_status",
    "review_notes",
    "review_label",
)


def build_review_rows(classification_manifest: str | Path) -> list[dict[str, str]]:
    """Build stable pending-review rows for the six exact-overlap classes."""
    rows = _read_csv(Path(classification_manifest), "Dataset B classification manifest")
    selected: list[dict[str, str]] = []
    seen_images: set[str] = set()
    for row in rows:
        class_name = _required(row, "class_name")
        if class_name not in SPECS_BY_B_CLASS:
            continue
        image_id = _required(row, "source_image_id")
        if image_id in seen_images:
            raise ExperimentPreparationError(
                f"Duplicate Dataset B source image in classification manifest: {image_id}"
            )
        seen_images.add(image_id)
        spec = SPECS_BY_B_CLASS[class_name]
        selected.append(
            {
                "image_path": _required(row, "image_path"),
                "source_image_id": image_id,
                "proposed_class": class_name,
                "dataset_a_class_id": spec.dataset_a_class_id,
                "dataset_a_class_name": spec.dataset_a_class_name,
                "source_question": _required(row, "source_question"),
                "source_answer": _required(row, "source_answer"),
                "perceptual_group_id": _required(row, "leakage_group_id"),
            }
        )
    selected.sort(key=lambda row: (row["proposed_class"], row["source_image_id"]))
    review_rows = []
    for index, row in enumerate(selected, start=1):
        review_rows.append(
            {
                "review_id": f"B6-{index:04d}",
                **row,
                "review_status": "pending",
                "review_notes": "",
                "review_label": "",
            }
        )
    return review_rows


def write_review_manifest(rows: list[dict[str, str]], destination: str | Path) -> Path:
    """Write a new editable review manifest without overwriting human work."""
    path = Path(destination).expanduser().resolve()
    if path.exists():
        raise ExperimentPreparationError(
            f"Refusing to overwrite review manifest: {path}"
        )
    _write_csv(path, rows, REVIEW_COLUMNS)
    return path


def create_contact_sheets(
    rows: list[dict[str, str]],
    *,
    dataset_root: str | Path,
    output_directory: str | Path,
    items_per_page: int = 12,
) -> tuple[Path, ...]:
    """Create non-cropped thumbnail sheets grouped by proposed class."""
    if items_per_page <= 0:
        raise ExperimentPreparationError("items_per_page must be positive")
    root = extended_length_path(dataset_root)
    output = Path(output_directory).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ExperimentPreparationError(
            f"Refusing to overwrite non-empty contact-sheet directory: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_class[row["proposed_class"]].append(row)
    generated: list[Path] = []
    for class_name in sorted(by_class):
        class_rows = by_class[class_name]
        for page_index, start in enumerate(
            range(0, len(class_rows), items_per_page), start=1
        ):
            page_rows = class_rows[start : start + items_per_page]
            destination = output / f"{class_name}_{page_index:02d}.jpg"
            _render_contact_sheet(page_rows, root, destination)
            generated.append(destination)
    return tuple(generated)


def build_dataset_a_training_pool(
    dataset_root: str | Path,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Reference all six Dataset A classes and report source-family dependence."""
    root = extended_length_path(dataset_root)
    if not root.is_dir():
        raise ExperimentPreparationError(f"Dataset A image root is missing: {root}")
    pool: list[dict[str, str]] = []
    class_reports: list[dict[str, Any]] = []
    global_hashes: dict[str, list[str]] = defaultdict(list)
    for spec in SIX_CLASS_SPECS:
        class_directory = root / spec.dataset_a_class_id
        if not class_directory.is_dir():
            raise ExperimentPreparationError(
                f"Dataset A class directory is missing: {class_directory}"
            )
        paths = sorted(
            path for path in class_directory.iterdir() if path.suffix == ".png"
        )
        generation_counts: Counter[int] = Counter()
        class_hashes: Counter[str] = Counter()
        for path in paths:
            template_id, generation = derive_template_family(path.name)
            if template_id != spec.source_template_id:
                raise ExperimentPreparationError(
                    f"Unexpected source template for {path.name}: {template_id}"
                )
            digest = _sha256(path)
            relative_path = path.relative_to(root).as_posix()
            generation_counts[generation] += 1
            class_hashes[digest] += 1
            global_hashes[digest].append(relative_path)
            pool.append(
                {
                    "image_path": relative_path,
                    "class_name": spec.dataset_b_class,
                    "split": "train",
                    "dataset_a_class_id": spec.dataset_a_class_id,
                    "dataset_a_class_name": spec.dataset_a_class_name,
                    "source_template_id": template_id,
                    "augmentation_family_id": (
                        f"dataset_a_{spec.dataset_a_class_id}_template_{template_id}"
                    ),
                    "augmentation_generation": str(generation),
                    "exact_content_sha256": digest,
                }
            )
        class_reports.append(
            {
                "dataset_a_class_id": spec.dataset_a_class_id,
                "dataset_a_class_name": spec.dataset_a_class_name,
                "class_name": spec.dataset_b_class,
                "sample_count": len(paths),
                "source_template_count": 1,
                "source_template_id": spec.source_template_id,
                "augmentation_generation_counts": dict(
                    sorted(generation_counts.items())
                ),
                "exact_unique_content_count": len(class_hashes),
                "exact_duplicate_group_count": sum(
                    count > 1 for count in class_hashes.values()
                ),
                "exact_duplicate_excess": sum(
                    count - 1 for count in class_hashes.values() if count > 1
                ),
            }
        )
    pool.sort(key=lambda row: (int(row["dataset_a_class_id"]), row["image_path"]))
    cross_class_duplicate_groups = []
    for digest, hash_paths in global_hashes.items():
        class_ids = {path.split("/", maxsplit=1)[0] for path in hash_paths}
        if len(class_ids) > 1:
            cross_class_duplicate_groups.append(
                {
                    "sha256": digest,
                    "class_ids": sorted(class_ids),
                    "paths": sorted(hash_paths),
                }
            )
    report = {
        "dataset_root": str(root),
        "total_samples": len(pool),
        "class_reports": class_reports,
        "source_groups_per_class": 1,
        "independent_validation_possible": False,
        "grouping_rule": (
            "All filenames resolve to one numeric source template per class; exact hashes "
            "and augmentation generation are recorded, but generations are dependent."
        ),
        "cross_class_exact_duplicate_groups": cross_class_duplicate_groups,
        "model_selection_protocol": "fixed_epochs_no_validation",
    }
    return pool, report


def write_dataset_a_pool(
    rows: list[dict[str, str]],
    report: dict[str, Any],
    *,
    manifest_path: str | Path,
    report_path: str | Path,
) -> tuple[Path, Path]:
    """Write Dataset A references and the grouping evidence report."""
    manifest = Path(manifest_path).expanduser().resolve()
    grouping = Path(report_path).expanduser().resolve()
    if manifest.exists() or grouping.exists():
        raise ExperimentPreparationError(
            "Refusing to overwrite existing Dataset A pool or grouping report"
        )
    _write_csv(manifest, rows, tuple(rows[0]))
    grouping.parent.mkdir(parents=True, exist_ok=True)
    grouping.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest, grouping


def derive_template_family(filename: str) -> tuple[str, int]:
    """Derive the original numeric template and augmentation-generation depth."""
    original_ids = re.findall(r"_original_(\d+)", filename)
    if original_ids:
        if len(set(original_ids)) != 1:
            raise ExperimentPreparationError(
                f"Filename contains conflicting source-template IDs: {filename}"
            )
        return original_ids[0], len(original_ids)
    match = re.fullmatch(r"(\d+)\.png", filename)
    if match is None:
        raise ExperimentPreparationError(
            f"Cannot derive Dataset A source template from filename: {filename}"
        )
    return match.group(1), 0


def _render_contact_sheet(
    rows: list[dict[str, str]], root: Path, destination: Path
) -> None:
    columns, rows_per_page = 3, 4
    cell_width, cell_height = 360, 420
    title_height = 48
    canvas = Image.new(
        "RGB",
        (columns * cell_width, title_height + rows_per_page * cell_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = _font(16)
    title_font = _font(22)
    class_name = rows[0]["proposed_class"]
    draw.text(
        (16, 12),
        f"Dataset B manual review: {class_name}",
        fill="black",
        font=title_font,
    )
    for index, row in enumerate(rows):
        column, line = index % columns, index // columns
        left = column * cell_width
        top = title_height + line * cell_height
        image_path = (root / row["image_path"]).resolve()
        if not image_path.is_relative_to(root) or not image_path.is_file():
            raise ExperimentPreparationError(
                f"Review image is missing or unsafe: {image_path}"
            )
        with Image.open(image_path) as source:
            thumbnail = ImageOps.contain(
                source.convert("RGB"), (320, 320), Image.Resampling.LANCZOS
            )
        image_left = left + (cell_width - thumbnail.width) // 2
        canvas.paste(thumbnail, (image_left, top + 8))
        text_top = top + 334
        lines = (
            f"{row['review_id']} | {row['proposed_class']}",
            f"source: {row['source_image_id']}",
            f"group: {row['perceptual_group_id']}",
        )
        for offset, text in enumerate(lines):
            draw.text(
                (left + 12, text_top + offset * 22), text, fill="black", font=font
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="JPEG", quality=90, optimize=True)


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _read_csv(path: Path, description: str) -> list[dict[str, str]]:
    try:
        with (
            path.expanduser().resolve().open(encoding="utf-8-sig", newline="") as handle
        ):
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ExperimentPreparationError(f"{description} has no header: {path}")
            return [dict(row) for row in reader]
    except (OSError, csv.Error) as error:
        raise ExperimentPreparationError(
            f"Could not read {description}: {path}"
        ) from error


def _write_csv(
    path: Path, rows: list[dict[str, str]], fieldnames: tuple[str, ...]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _required(row: dict[str, str], column: str) -> str:
    value = row.get(column, "").strip()
    if not value:
        raise ExperimentPreparationError(f"Required column {column!r} is blank")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
