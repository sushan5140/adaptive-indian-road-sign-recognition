"""Human-review-first audit utilities for independently acquired unseen signs."""

from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from PIL import Image, UnidentifiedImageError

REVIEW_COLUMNS = (
    "review_id",
    "class_label",
    "relative_image_path",
    "source_filename",
    "source_identifier",
    "source_url",
    "license",
    "attribution",
    "width",
    "height",
    "channels",
    "sha256",
    "perceptual_hash",
    "exact_duplicate_group_id",
    "perceptual_group_id",
    "source_group_id",
    "independence_group_id",
    "exact_duplicate",
    "likely_near_duplicate",
    "cross_label_group_conflict",
    "technical_status",
    "audit_flags",
    "review_status",
    "review_label",
    "rejection_reason",
    "review_notes",
)
EDITABLE_COLUMNS = (
    "review_status",
    "review_label",
    "rejection_reason",
    "review_notes",
)
PROTECTED_COLUMNS = tuple(
    column for column in REVIEW_COLUMNS if column not in EDITABLE_COLUMNS
)
METADATA_COLUMNS = (
    "relative_path",
    "source_identifier",
    "source_url",
    "license",
    "attribution",
)
ALLOWED_REVIEW_STATUSES = {"pending", "approved", "rejected", "relabel"}


class UnseenAuditError(ValueError):
    """Raised when unseen-class intake data cannot be audited safely."""


@dataclass(frozen=True, slots=True)
class UnseenAuditConfig:
    """Configuration for a read-only unseen-class image audit."""

    raw_root: Path
    class_names: tuple[str, ...]
    source_metadata_csv: Path
    existing_review_csv: Path | None = None
    allowed_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")
    near_duplicate_hamming_distance: int = 4
    minimum_independent_groups: int = 15
    target_photos_minimum: int = 30
    target_photos_maximum: int = 50

    def __post_init__(self) -> None:
        if not self.class_names or len(set(self.class_names)) != len(self.class_names):
            raise UnseenAuditError("class_names must be non-empty and unique")
        if not 0 <= self.near_duplicate_hamming_distance <= 64:
            raise UnseenAuditError("near-duplicate Hamming distance must be in [0, 64]")
        if self.minimum_independent_groups < 15:
            raise UnseenAuditError("minimum_independent_groups cannot be below 15")
        if not 1 <= self.target_photos_minimum <= self.target_photos_maximum:
            raise UnseenAuditError("photo target range is invalid")


@dataclass(frozen=True, slots=True)
class UnseenAuditResult:
    """Review rows, duplicate evidence, and measured intake summary."""

    review_rows: tuple[dict[str, str], ...]
    near_duplicate_pairs: tuple[dict[str, str], ...]
    summary: dict[str, Any]


class UnseenDatasetAuditor:
    """Audit quarantined real photographs without approving or moving them."""

    def __init__(self, config: UnseenAuditConfig) -> None:
        self.config = config

    def audit(self) -> UnseenAuditResult:
        """Inspect images, group dependence, and preserve validated human decisions."""
        self._validate_roots()
        metadata = _read_metadata(self.config.source_metadata_csv)
        inventory = self._inspect_images(metadata)
        discovered_paths = {row["relative_image_path"] for row in inventory}
        unused_metadata = sorted(set(metadata).difference(discovered_paths))
        if unused_metadata:
            raise UnseenAuditError(
                f"Source metadata references missing images: {unused_metadata}"
            )

        exact_members: dict[str, list[str]] = defaultdict(list)
        for row in inventory:
            exact_members[row["sha256"]].append(row["review_id"])
        exact_group_by_id: dict[str, str] = {}
        for digest, review_ids in exact_members.items():
            if len(review_ids) > 1:
                group_id = f"EXACT-{digest[:12].upper()}"
                exact_group_by_id.update(
                    {review_id: group_id for review_id in review_ids}
                )

        readable = [row for row in inventory if row["perceptual_hash"]]
        near_pairs: list[dict[str, str]] = []
        perceptual_parent = {row["review_id"]: row["review_id"] for row in inventory}
        for review_ids in exact_members.values():
            for review_id in review_ids[1:]:
                _union(perceptual_parent, review_ids[0], review_id)
        for index, left in enumerate(readable):
            for right in readable[index + 1 :]:
                if left["sha256"] == right["sha256"]:
                    _union(perceptual_parent, left["review_id"], right["review_id"])
                    continue
                distance = (
                    int(left["perceptual_hash"], 16) ^ int(right["perceptual_hash"], 16)
                ).bit_count()
                if distance <= self.config.near_duplicate_hamming_distance:
                    _union(perceptual_parent, left["review_id"], right["review_id"])
                    near_pairs.append(
                        {
                            "left_review_id": left["review_id"],
                            "right_review_id": right["review_id"],
                            "hamming_distance": str(distance),
                            "cross_label": str(
                                left["class_label"] != right["class_label"]
                            ).lower(),
                        }
                    )

        perceptual_groups = _stable_group_ids(perceptual_parent, "PER")
        independence_parent = dict(perceptual_parent)
        source_members: dict[str, list[str]] = defaultdict(list)
        for row in inventory:
            if row["source_identifier"]:
                source_members[row["source_identifier"]].append(row["review_id"])
        for members in source_members.values():
            for member in members[1:]:
                _union(independence_parent, members[0], member)
        independence_groups = _stable_group_ids(independence_parent, "IND")

        labels_by_independence_group: dict[str, set[str]] = defaultdict(set)
        for row in inventory:
            labels_by_independence_group[independence_groups[row["review_id"]]].add(
                row["class_label"]
            )
        conflicting_groups = {
            group_id
            for group_id, labels in labels_by_independence_group.items()
            if len(labels) > 1
        }
        near_ids = {
            pair[key]
            for pair in near_pairs
            for key in ("left_review_id", "right_review_id")
        }
        rows: list[dict[str, str]] = []
        for row in inventory:
            review_id = row["review_id"]
            source_id = row["source_identifier"]
            source_group = (
                f"SRC-{hashlib.sha256(source_id.encode()).hexdigest()[:12].upper()}"
                if source_id
                else ""
            )
            group_id = independence_groups[review_id]
            flags: list[str] = []
            if row["technical_status"] != "readable":
                flags.append("corrupt_or_unsupported_image")
            if not row["source_identifier"]:
                flags.append("missing_source_identifier")
            if not row["license"]:
                flags.append("missing_license")
            if review_id in exact_group_by_id:
                flags.append("exact_duplicate")
            if review_id in near_ids:
                flags.append("likely_near_duplicate")
            if group_id in conflicting_groups:
                flags.append("cross_label_group_conflict")
            rows.append(
                {
                    **row,
                    "exact_duplicate_group_id": exact_group_by_id.get(review_id, ""),
                    "perceptual_group_id": perceptual_groups[review_id],
                    "source_group_id": source_group,
                    "independence_group_id": group_id,
                    "exact_duplicate": str(review_id in exact_group_by_id).lower(),
                    "likely_near_duplicate": str(review_id in near_ids).lower(),
                    "cross_label_group_conflict": str(
                        group_id in conflicting_groups
                    ).lower(),
                    "audit_flags": ";".join(flags),
                    "review_status": "pending",
                    "review_label": "",
                    "rejection_reason": "",
                    "review_notes": "",
                }
            )
        rows.sort(key=lambda row: row["review_id"])
        rows = _preserve_existing_decisions(rows, self.config.existing_review_csv)
        statuses = Counter(row["review_status"] for row in rows)
        per_class: dict[str, dict[str, Any]] = {}
        for class_name in sorted(self.config.class_names):
            class_rows = [row for row in rows if row["class_label"] == class_name]
            approved = [row for row in class_rows if row["review_status"] == "approved"]
            approved_groups = {row["independence_group_id"] for row in approved}
            per_class[class_name] = {
                "raw_photos": len(class_rows),
                "independent_groups": len(
                    {row["independence_group_id"] for row in class_rows}
                ),
                "pending": sum(row["review_status"] == "pending" for row in class_rows),
                "approved": len(approved),
                "approved_independent_groups": len(approved_groups),
                "minimum_groups_met": (
                    len(approved_groups) >= self.config.minimum_independent_groups
                ),
                "target_photo_range_met": (
                    self.config.target_photos_minimum
                    <= len(approved)
                    <= self.config.target_photos_maximum
                ),
            }
        summary: dict[str, Any] = {
            "raw_photo_count": len(rows),
            "review_status_counts": {
                status: statuses[status]
                for status in ("approved", "rejected", "relabel", "pending")
            },
            "readable_count": sum(
                row["technical_status"] == "readable" for row in rows
            ),
            "corrupt_or_unsupported_count": sum(
                row["technical_status"] != "readable" for row in rows
            ),
            "exact_duplicate_group_count": len(set(exact_group_by_id.values())),
            "near_duplicate_pair_count": len(near_pairs),
            "cross_label_group_conflict_count": len(conflicting_groups),
            "independent_group_count": len(set(independence_groups.values())),
            "minimum_independent_groups_per_class": (
                self.config.minimum_independent_groups
            ),
            "target_photos_per_class": [
                self.config.target_photos_minimum,
                self.config.target_photos_maximum,
            ],
            "human_review_required": True,
            "automatic_approval_performed": False,
            "partition_created": False,
            "training_or_evaluation_performed": False,
            "per_class": per_class,
        }
        return UnseenAuditResult(tuple(rows), tuple(near_pairs), summary)

    def _validate_roots(self) -> None:
        if not self.config.raw_root.is_dir():
            raise UnseenAuditError(
                f"Raw unseen root is missing: {self.config.raw_root}"
            )
        actual = {path.name for path in self.config.raw_root.iterdir() if path.is_dir()}
        unexpected = sorted(actual.difference(self.config.class_names))
        missing = sorted(set(self.config.class_names).difference(actual))
        if unexpected or missing:
            raise UnseenAuditError(
                f"Unseen class folders differ from config; missing={missing}, "
                f"unexpected={unexpected}"
            )

    def _inspect_images(
        self, metadata: dict[str, dict[str, str]]
    ) -> list[dict[str, str]]:
        inventory: list[dict[str, str]] = []
        for class_name in sorted(self.config.class_names):
            class_root = self.config.raw_root / class_name
            for path in sorted(
                class_root.iterdir(), key=lambda value: value.name.casefold()
            ):
                if not path.is_file() or path.suffix.casefold() not in {
                    extension.casefold() for extension in self.config.allowed_extensions
                }:
                    continue
                relative_path = path.relative_to(self.config.raw_root).as_posix()
                review_id = _review_id(class_name, relative_path)
                width = height = channels = ""
                perceptual_hash = ""
                technical_status = "readable"
                try:
                    with Image.open(path) as image:
                        image.load()
                        width, height = (str(value) for value in image.size)
                        channels = str(len(image.getbands()))
                        perceptual_hash = f"{_difference_hash(image):016x}"
                except (OSError, UnidentifiedImageError, ValueError):
                    technical_status = "corrupt_or_unsupported"
                source = metadata.get(relative_path, {})
                inventory.append(
                    {
                        "review_id": review_id,
                        "class_label": class_name,
                        "relative_image_path": relative_path,
                        "source_filename": path.name,
                        "source_identifier": source.get("source_identifier", ""),
                        "source_url": source.get("source_url", ""),
                        "license": source.get("license", ""),
                        "attribution": source.get("attribution", ""),
                        "width": width,
                        "height": height,
                        "channels": channels,
                        "sha256": _file_hash(path),
                        "perceptual_hash": perceptual_hash,
                        "technical_status": technical_status,
                    }
                )
        if len({row["review_id"] for row in inventory}) != len(inventory):
            raise UnseenAuditError("Stable review ID collision detected")
        return inventory


def _read_metadata(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        raise UnseenAuditError(f"Source metadata CSV is missing: {path}")
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise UnseenAuditError(f"Source metadata CSV has no header: {path}")
            missing = sorted(set(METADATA_COLUMNS).difference(reader.fieldnames))
            if missing:
                raise UnseenAuditError(
                    f"Source metadata columns are missing: {missing}"
                )
            rows = [dict(row) for row in reader]
    except (OSError, csv.Error) as error:
        raise UnseenAuditError(f"Could not read source metadata CSV: {path}") from error
    result: dict[str, dict[str, str]] = {}
    for index, row in enumerate(rows, start=2):
        relative = row.get("relative_path", "").strip().replace("\\", "/")
        if not relative or relative.startswith("/") or ".." in Path(relative).parts:
            raise UnseenAuditError(f"Source metadata row {index} has unsafe path")
        if relative in result:
            raise UnseenAuditError(f"Duplicate source metadata path: {relative}")
        result[relative] = {key: row.get(key, "").strip() for key in METADATA_COLUMNS}
    return result


def _preserve_existing_decisions(
    generated: list[dict[str, str]], existing_path: Path | None
) -> list[dict[str, str]]:
    if existing_path is None or not existing_path.is_file():
        return generated
    with existing_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise UnseenAuditError(
                f"Existing review CSV has no header: {existing_path}"
            )
        missing = sorted(set(REVIEW_COLUMNS).difference(reader.fieldnames))
        if missing:
            raise UnseenAuditError(f"Existing review columns are missing: {missing}")
        existing_rows = [dict(row) for row in reader]
    existing_by_id: dict[str, dict[str, str]] = {}
    for row in existing_rows:
        review_id = row.get("review_id", "")
        if review_id in existing_by_id:
            raise UnseenAuditError(f"Duplicate existing review ID: {review_id}")
        existing_by_id[review_id] = row
    for row in generated:
        existing = existing_by_id.get(row["review_id"])
        if existing is None:
            continue
        for column in PROTECTED_COLUMNS:
            if existing.get(column, "") != row[column]:
                raise UnseenAuditError(
                    f"Protected field changed for {row['review_id']}: {column}"
                )
        status = existing.get("review_status", "").strip().casefold()
        if status not in ALLOWED_REVIEW_STATUSES:
            raise UnseenAuditError(
                f"Invalid review status for {row['review_id']}: {status!r}"
            )
        for column in EDITABLE_COLUMNS:
            row[column] = existing.get(column, "").strip()
        row["review_status"] = status
    return generated


def _review_id(class_name: str, relative_path: str) -> str:
    digest = hashlib.sha256(f"{class_name}\0{relative_path}".encode()).hexdigest()
    return f"UNSEEN-{digest[:12].upper()}"


def _difference_hash(image: Image.Image) -> int:
    grayscale = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = cast(list[int], list(grayscale.get_flattened_data()))
    value = 0
    for row in range(8):
        for column in range(8):
            value = (value << 1) | int(
                pixels[row * 9 + column] > pixels[row * 9 + column + 1]
            )
    return value


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find(parent: dict[str, str], value: str) -> str:
    while parent[value] != value:
        parent[value] = parent[parent[value]]
        value = parent[value]
    return value


def _union(parent: dict[str, str], left: str, right: str) -> None:
    left_root, right_root = _find(parent, left), _find(parent, right)
    if left_root != right_root:
        parent[max(left_root, right_root)] = min(left_root, right_root)


def _stable_group_ids(parent: dict[str, str], prefix: str) -> dict[str, str]:
    members: dict[str, list[str]] = defaultdict(list)
    for value in parent:
        members[_find(parent, value)].append(value)
    group_ids = {
        root: f"{prefix}-{min(values).removeprefix('UNSEEN-')}"
        for root, values in members.items()
    }
    return {value: group_ids[_find(parent, value)] for value in parent}
