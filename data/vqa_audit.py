"""Read-only quality audit and conservative label derivation for VQA data."""

from __future__ import annotations

import csv
import hashlib
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from PIL import Image, UnidentifiedImageError

DIRECT_IDENTITY_QUESTIONS = {
    "what does this traffic sign indicate",
    "what does the traffic sign indicate",
    "what does this sign indicate",
    "what does the symbol represent",
    "what is ahead according to the traffic sign",
    "what type of obstacle does this traffic sign warn about",
    "what does a red triangle traffic signal with a cross mean",
}

ANSWER_ALIASES = {
    "pedestrian crossing ahead": "pedestrian_crossing_ahead",
    "school": "school_ahead",
    "school zone": "school_ahead",
    "stop": "stop",
    "humps/speed breakers": "road_hump",
    "hump/speed breaker": "road_hump",
    "hump": "road_hump",
    "bump": "road_hump",
    "right curve ahead": "right_curve_ahead",
    "right bend": "right_curve_ahead",
    "left curve ahead": "left_curve_ahead",
    "give way to traffic on major road": "give_way",
    "petrol pump": "filling_station",
    "fuel station": "filling_station",
    "fuel": "filling_station",
    "y-intersection": "y_junction",
    "major road ahead": "major_road_ahead",
    "side road right": "side_road_right",
    "side road left": "side_road_left",
    "hairpin bend ahead": "hairpin_bend_ahead",
    "no entry": "no_entry",
    "one way": "one_way",
    "gap in median": "gap_in_median",
    "vehicles may pass on either side": "pass_either_side",
    "narrow bridge": "narrow_bridge",
    "no right turn": "no_right_turn",
    "no left turn": "no_left_turn",
    "truck lay-by": "truck_lay_by",
    "no overtaking": "no_overtaking",
    "t-intersection": "t_junction",
    "no parking": "no_parking",
    "staggered intersection": "staggered_junction",
    "overhead cables": "overhead_cables",
    "unguarded railway crossing": "unguarded_level_crossing",
    "steep ascent": "steep_ascent",
    "no horn": "horn_prohibited",
    "no stopping": "no_stopping",
    "cattle": "cattle",
    "bus stop": "bus_stop",
    "restaurant": "restaurant",
    "keep left": "keep_left",
    "keep right": "keep_right",
}

# Explicit expert mapping. Numeric Dataset A labels are not self-describing, and
# duplicated names in Dataset A remain listed rather than arbitrarily resolved.
DATASET_A_ALIGNMENT: dict[str, tuple[str, str, str]] = {
    "give_way": ("0", "exact", "Give way"),
    "no_entry": ("1", "exact", "No entry"),
    "one_way": ("2|3", "ambiguous", "Dataset A has two one-way orientations"),
    "no_left_turn": ("15", "exact", "No left turn"),
    "no_right_turn": ("16", "exact", "No right turn"),
    "no_overtaking": ("17", "exact", "No overtaking"),
    "horn_prohibited": ("20", "exact", "Horn prohibited"),
    "no_parking": ("21", "exact", "No parking"),
    "no_stopping": ("22", "exact", "No stopping"),
    "steep_ascent": ("26", "exact", "Steep ascent"),
    "narrow_bridge": ("28", "exact", "Narrow bridge"),
    "road_hump": ("30", "exact", "Road hump"),
    "cattle": ("34", "exact", "Cattle"),
    "side_road_right": ("36|37", "ambiguous", "Direction IDs require visual review"),
    "side_road_left": ("36|37", "ambiguous", "Direction IDs require visual review"),
    "t_junction": ("40", "exact", "T-junction"),
    "y_junction": ("41", "exact", "Y-junction"),
    "staggered_junction": (
        "42|43",
        "ambiguous",
        "Orientation IDs require visual review",
    ),
    "unguarded_level_crossing": ("46", "exact", "Unguarded level crossing"),
    "bus_stop": ("52", "exact", "Bus stop"),
    "filling_station": ("55", "exact", "Filling station"),
    "restaurant": ("57", "exact", "Restaurant"),
}

DATASET_A_CLASS_NAMES = {
    "0": "Give way",
    "1": "No entry",
    "2": "One-way traffic",
    "3": "One-way traffic",
    "15": "No left turn",
    "16": "No right turn",
    "17": "No overtaking",
    "20": "Horn prohibited",
    "21": "No parking",
    "22": "No stopping",
    "26": "Steep ascent",
    "28": "Narrow bridge",
    "30": "Road hump",
    "34": "Cattle",
    "36": "Side road junction",
    "37": "Side road junction",
    "40": "T-junction",
    "41": "Y-junction",
    "42": "Staggered side road junction",
    "43": "Staggered side road junction",
    "46": "Unguarded level crossing ahead",
    "52": "Bus stop",
    "55": "Filling station",
    "57": "Restaurant",
}


class VqaAuditError(ValueError):
    """Raised when VQA source data is missing or structurally invalid."""


@dataclass(frozen=True, slots=True)
class VqaAuditConfig:
    """Paths and conservative thresholds for a VQA dataset audit."""

    dataset_root: Path
    images_directory: Path
    compact_csv: Path
    full_csv: Path
    minimum_class_images: int = 10
    minimum_independent_groups: int = 5
    near_duplicate_hamming_distance: int = 4

    def __post_init__(self) -> None:
        if self.minimum_class_images < 1:
            raise VqaAuditError("minimum_class_images must be positive")
        if self.minimum_independent_groups < 1:
            raise VqaAuditError("minimum_independent_groups must be positive")
        if not 0 <= self.near_duplicate_hamming_distance <= 64:
            raise VqaAuditError("near-duplicate Hamming distance must be in [0, 64]")


@dataclass(frozen=True, slots=True)
class VqaAuditResult:
    """Serializable audit summary plus tabular rows for generated artifacts."""

    report: dict[str, Any]
    quality: dict[str, Any]
    image_inventory: tuple[dict[str, Any], ...]
    question_frequency: tuple[dict[str, Any], ...]
    answer_frequency: tuple[dict[str, Any], ...]
    question_answer_frequency: tuple[dict[str, Any], ...]
    exact_duplicate_groups: tuple[dict[str, Any], ...]
    near_duplicate_pairs: tuple[dict[str, Any], ...]
    candidate_manifest: tuple[dict[str, Any], ...]
    excluded_candidates: tuple[dict[str, Any], ...]
    class_distribution: tuple[dict[str, Any], ...]
    class_mapping: dict[str, int]
    semantic_alignment: tuple[dict[str, Any], ...]


class VqaDatasetAuditor:
    """Audit Indian Traffic VQA images and derive only defensible class labels."""

    def __init__(self, config: VqaAuditConfig) -> None:
        self.config = config

    def audit(self) -> VqaAuditResult:
        """Decode images, audit annotations, and build a conservative manifest."""
        self._validate_paths()
        compact_rows, compact_columns = _read_vqa_csv(self.config.compact_csv)
        full_rows, full_columns = _read_vqa_csv(self.config.full_csv)
        inventory, unreadable = self._inspect_images()
        image_names = {str(row["image_name"]) for row in inventory}
        image_hashes = {str(row["image_name"]): str(row["sha256"]) for row in inventory}
        exact_groups = _exact_duplicate_groups(image_hashes)
        near_pairs = _near_duplicate_pairs(
            inventory,
            self.config.near_duplicate_hamming_distance,
            image_hashes,
        )
        leakage_groups = _leakage_groups(image_names, exact_groups, near_pairs)
        annotation_quality = _annotation_quality(full_rows, image_names)
        candidates, excluded = _derive_candidates(
            full_rows, self.config.images_directory
        )
        direct_identity_image_count = len(candidates) + sum(
            row["reason"] != "no_direct_identity_question" for row in excluded
        )
        initially_supported_candidate_count = len(candidates)
        labels_by_group: dict[str, set[str]] = defaultdict(set)
        for row in candidates:
            labels_by_group[leakage_groups[str(row["image_id"])]].add(str(row["label"]))
        conflicting_groups = {
            group_id: labels
            for group_id, labels in labels_by_group.items()
            if len(labels) > 1
        }
        non_conflicting_candidates: list[dict[str, Any]] = []
        for row in candidates:
            group_id = leakage_groups[str(row["image_id"])]
            if group_id in conflicting_groups:
                excluded.append(
                    {
                        "image_id": row["image_id"],
                        "source_question": row["source_question"],
                        "source_answer": row["source_answer"],
                        "reason": "conflicting_labels_within_duplicate_group",
                        "proposed_label": row["label"],
                    }
                )
                continue
            non_conflicting_candidates.append(row)
        distribution = Counter(str(row["label"]) for row in non_conflicting_candidates)
        groups_by_class: dict[str, set[str]] = defaultdict(set)
        for row in non_conflicting_candidates:
            groups_by_class[str(row["label"])].add(leakage_groups[str(row["image_id"])])
        viable_labels = {
            label
            for label, count in distribution.items()
            if count >= self.config.minimum_class_images
            and len(groups_by_class[label]) >= self.config.minimum_independent_groups
        }
        selected: list[dict[str, Any]] = []
        for row in non_conflicting_candidates:
            label = str(row["label"])
            if label not in viable_labels:
                excluded.append(
                    {
                        "image_id": row["image_id"],
                        "source_question": row["source_question"],
                        "source_answer": row["source_answer"],
                        "reason": "class_below_viability_threshold",
                        "proposed_label": label,
                    }
                )
                continue
            selected.append(
                {
                    **row,
                    "leakage_group_id": leakage_groups[str(row["image_id"])],
                }
            )
        selected_distribution = Counter(str(row["label"]) for row in selected)
        class_mapping = {
            label: index for index, label in enumerate(sorted(selected_distribution))
        }
        source_variants: dict[str, set[str]] = defaultdict(set)
        for row in selected:
            source_variants[str(row["label"])].add(str(row["source_answer"]))
        conflicting_by_label = Counter(
            str(row["proposed_label"])
            for row in excluded
            if row["reason"]
            in {
                "conflicting_identity_labels",
                "conflicting_labels_within_duplicate_group",
            }
        )
        class_rows = tuple(
            {
                "label": label,
                "image_count": selected_distribution[label],
                "independent_group_count": len(groups_by_class[label]),
                "source_answer_variants": sorted(source_variants[label]),
                "ambiguous_examples_count": 0,
                "conflicting_examples_count": conflicting_by_label[label],
                "fewer_than_5_images": selected_distribution[label] < 5,
                "fewer_than_10_images": selected_distribution[label] < 10,
                "fewer_than_20_images": selected_distribution[label] < 20,
            }
            for label in sorted(selected_distribution)
        )
        semantic_alignment = tuple(
            _alignment_row(label) for label in sorted(selected_distribution)
        )
        question_counts = Counter(str(row["question"]) for row in full_rows)
        answer_counts = Counter(str(row["answer"]) for row in full_rows)
        qa_counts = Counter(
            (str(row["question"]), str(row["answer"])) for row in full_rows
        )
        compact_keys = {
            (row["image_name"], row["question"], row["answer"]) for row in compact_rows
        }
        full_keys = {
            (row["image_name"], row["question"], row["answer"]) for row in full_rows
        }
        compact_rows_not_in_full = [
            {
                "image_name": row["image_name"],
                "question": row["question"],
                "answer": row["answer"],
            }
            for row in compact_rows
            if (row["image_name"], row["question"], row["answer"]) not in full_keys
        ]
        report = {
            "dataset_root": str(self.config.dataset_root.resolve()),
            "image_directory": str(self.config.images_directory.resolve()),
            "images": {
                "discovered": len(inventory) + len(unreadable),
                "readable": len(inventory),
                "unreadable": unreadable,
                "formats": dict(Counter(str(row["format"]) for row in inventory)),
                "colour_modes": dict(Counter(str(row["mode"]) for row in inventory)),
                "dimensions": _dimension_summary(inventory),
            },
            "compact_csv": {
                "path": str(self.config.compact_csv.resolve()),
                "columns": compact_columns,
                "rows": len(compact_rows),
                "unique_images": len({row["image_name"] for row in compact_rows}),
                "is_subset_of_full_csv": compact_keys <= full_keys,
                "rows_not_in_full_csv": compact_rows_not_in_full,
            },
            "full_csv": {
                "path": str(self.config.full_csv.resolve()),
                "columns": full_columns,
                "rows": len(full_rows),
                "unique_images": len({row["image_name"] for row in full_rows}),
                "unique_questions": len(question_counts),
                "unique_answers": len(answer_counts),
                "questions_per_image": _count_distribution(full_rows),
            },
            "annotation_quality": annotation_quality,
            "exact_duplicate_group_count": len(exact_groups),
            "near_duplicate_pair_count": len(near_pairs),
            "byte_unique_image_count": len(set(image_hashes.values())),
            "perceptual_leakage_group_count": len(set(leakage_groups.values())),
            "direct_identity_question_images": direct_identity_image_count,
            "initially_supported_candidate_images": (
                initially_supported_candidate_count
            ),
            "usable_source_answer_variant_count": len(
                {str(row["source_answer"]) for row in candidates}
            ),
            "initially_supported_canonical_class_count": len(
                {str(row["label"]) for row in candidates}
            ),
            "conflicting_duplicate_groups": {
                group_id: sorted(labels)
                for group_id, labels in sorted(conflicting_groups.items())
            },
            "selected_classification_images": len(selected),
            "selected_class_count": len(class_mapping),
            "selected_class_statistics": _class_statistics(selected_distribution),
            "selection_thresholds": {
                "minimum_class_images": self.config.minimum_class_images,
                "minimum_independent_groups": self.config.minimum_independent_groups,
                "near_duplicate_hamming_distance": (
                    self.config.near_duplicate_hamming_distance
                ),
            },
        }
        quality = {
            "classification_manifest_is_derived_not_source_ground_truth": True,
            "vqa_rows_are_not_independent_samples": True,
            "one_photograph_appears_at_most_once_in_manifest": (
                len(selected) == len({row["image_id"] for row in selected})
            ),
            "split_created": False,
            "training_started": False,
            "recommended_plan": "B",
            "recommendation": (
                "After human review, train only semantically exact overlapping Dataset A "
                "classes and keep all corresponding Dataset B leakage groups as one "
                "external held-out evaluation set. Do not tune on Dataset B or claim "
                "that a Dataset A random holdout measures real-world generalization."
            ),
            "real_baseline_training_scientifically_justified_now": False,
            "training_blockers": [
                "The derived Dataset B class manifest still requires human visual review.",
                "Only exact semantic overlaps should enter the first external benchmark.",
                "A training-only model-selection protocol must be documented first.",
            ],
            "limitations": [
                "Labels are conservatively derived from direct VQA identity answers.",
                "Perceptual-hash near-duplicate pairs are screening candidates, not proof.",
                "No train/validation/test assignment is emitted until label review.",
                "Dataset A numeric orientations cannot always be aligned from names alone.",
            ],
        }
        return VqaAuditResult(
            report=report,
            quality=quality,
            image_inventory=tuple(inventory),
            question_frequency=tuple(
                {"question": value, "count": count}
                for value, count in question_counts.most_common()
            ),
            answer_frequency=tuple(
                {"answer": value, "count": count}
                for value, count in answer_counts.most_common()
            ),
            question_answer_frequency=tuple(
                {"question": question, "answer": answer, "count": count}
                for (question, answer), count in qa_counts.most_common()
            ),
            exact_duplicate_groups=tuple(exact_groups),
            near_duplicate_pairs=tuple(near_pairs),
            candidate_manifest=tuple(
                sorted(selected, key=lambda row: str(row["image_id"]))
            ),
            excluded_candidates=tuple(
                sorted(excluded, key=lambda row: str(row["image_id"]))
            ),
            class_distribution=class_rows,
            class_mapping=class_mapping,
            semantic_alignment=semantic_alignment,
        )

    def _validate_paths(self) -> None:
        for path, description in (
            (self.config.dataset_root, "dataset root"),
            (self.config.images_directory, "image directory"),
        ):
            if not path.is_dir():
                raise VqaAuditError(f"Missing {description}: {path}")
        for path, description in (
            (self.config.compact_csv, "compact CSV"),
            (self.config.full_csv, "full CSV"),
        ):
            if not path.is_file():
                raise VqaAuditError(f"Missing {description}: {path}")

    def _inspect_images(self) -> tuple[list[dict[str, Any]], list[str]]:
        inventory: list[dict[str, Any]] = []
        unreadable: list[str] = []
        for path in sorted(self.config.images_directory.iterdir()):
            if not path.is_file() or path.suffix.casefold() not in {
                ".jpg",
                ".jpeg",
                ".png",
            }:
                continue
            try:
                with Image.open(path) as image:
                    image.load()
                    width, height = image.size
                    image_format = image.format or "unknown"
                    mode = image.mode
                    difference_hash = _difference_hash(image)
            except (OSError, UnidentifiedImageError):
                unreadable.append(path.name)
                continue
            inventory.append(
                {
                    "image_name": path.name,
                    "relative_path": path.relative_to(
                        self.config.dataset_root
                    ).as_posix(),
                    "width": width,
                    "height": height,
                    "format": image_format,
                    "mode": mode,
                    "bytes": path.stat().st_size,
                    "sha256": _file_hash(path),
                    "difference_hash": f"{difference_hash:016x}",
                }
            )
        return inventory, unreadable


def result_to_dict(result: VqaAuditResult) -> dict[str, Any]:
    """Convert a VQA audit result to built-in serializable containers."""
    return asdict(result)


def _read_vqa_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise VqaAuditError(f"CSV has no header: {path}")
            original_columns = list(reader.fieldnames)
            rows = list(reader)
    except (OSError, csv.Error) as error:
        raise VqaAuditError(f"Could not read VQA CSV: {path}") from error
    canonical_columns = {_normalize_header(name): name for name in original_columns}
    required = {"id", "imagename", "question", "answer"}
    if not required <= canonical_columns.keys():
        raise VqaAuditError(
            f"CSV {path} requires id, image_name, question, and answer columns"
        )
    category_candidates = [
        name
        for key, name in canonical_columns.items()
        if "sign" in key and key not in required
    ]
    if len(category_candidates) != 1:
        raise VqaAuditError(
            f"CSV {path} requires exactly one traffic-sign category column"
        )
    normalized_rows: list[dict[str, str]] = []
    for row_number, row in enumerate(rows, start=2):
        normalized = {
            "id": _clean(row.get(canonical_columns["id"])),
            "image_name": _clean(row.get(canonical_columns["imagename"])),
            "question": _clean(row.get(canonical_columns["question"])),
            "answer": _clean(row.get(canonical_columns["answer"])),
            "category": _canonical_category(_clean(row.get(category_candidates[0]))),
        }
        if any(
            not normalized[key] for key in ("id", "image_name", "question", "answer")
        ):
            raise VqaAuditError(
                f"CSV {path}, row {row_number}: required value is blank"
            )
        normalized_rows.append(normalized)
    return normalized_rows, original_columns


def _derive_candidates(
    rows: list[dict[str, str]], images_directory: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_image: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_image[row["image_name"]].append(row)
    candidates: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for image_name, image_rows in sorted(by_image.items()):
        direct = [
            row
            for row in image_rows
            if _normalize_text(row["question"]) in DIRECT_IDENTITY_QUESTIONS
        ]
        if not direct:
            excluded.append(
                {
                    "image_id": image_name,
                    "source_question": "",
                    "source_answer": "",
                    "reason": "no_direct_identity_question",
                    "proposed_label": "",
                }
            )
            continue
        derived = {
            label
            for label in (_canonical_label(row["answer"], image_rows) for row in direct)
            if label is not None
        }
        if len(derived) != 1:
            excluded.append(
                {
                    "image_id": image_name,
                    "source_question": " | ".join(row["question"] for row in direct),
                    "source_answer": " | ".join(row["answer"] for row in direct),
                    "reason": (
                        "unsupported_or_ambiguous_identity_answer"
                        if not derived
                        else "conflicting_identity_labels"
                    ),
                    "proposed_label": " | ".join(sorted(derived)),
                }
            )
            continue
        source = direct[0]
        candidates.append(
            {
                "relative_image_path": f"{images_directory.name}/{image_name}",
                "label": next(iter(derived)),
                "category": source["category"],
                "source_question": source["question"],
                "source_answer": source["answer"],
                "image_id": image_name,
            }
        )
    return candidates, excluded


def _canonical_label(answer: str, image_rows: list[dict[str, str]]) -> str | None:
    normalized = _normalize_text(answer)
    if normalized in {"speed", "speed limit"}:
        for row in image_rows:
            if (
                _normalize_text(row["question"])
                == "what speed limit does this traffic sign indicate"
            ):
                match = re.search(r"\b(\d{1,3})\b", row["answer"])
                if match:
                    return f"maximum_speed_limit_{match.group(1)}_km_h"
        return None
    return ANSWER_ALIASES.get(normalized)


def _annotation_quality(
    rows: list[dict[str, str]], image_names: set[str]
) -> dict[str, Any]:
    row_keys = [
        tuple(
            row[key] for key in ("id", "image_name", "question", "answer", "category")
        )
        for row in rows
    ]
    by_image_question: dict[tuple[str, str], set[str]] = defaultdict(set)
    categories_by_image: dict[str, set[str]] = defaultdict(set)
    referenced = set()
    for row in rows:
        image_name = row["image_name"]
        referenced.add(image_name)
        by_image_question[(image_name, _normalize_text(row["question"]))].add(
            _normalize_text(row["answer"])
        )
        categories_by_image[image_name].add(row["category"])
    conflicts = [
        {"image_name": image, "question": question, "answers": sorted(answers)}
        for (image, question), answers in by_image_question.items()
        if len(answers) > 1
    ]
    return {
        "duplicate_rows": len(row_keys) - len(set(row_keys)),
        "missing_image_references": sorted(referenced - image_names),
        "unreferenced_images": sorted(image_names - referenced),
        "conflicting_image_question_answers": conflicts,
        "images_with_multiple_categories": sorted(
            image for image, values in categories_by_image.items() if len(values) > 1
        ),
        "category_frequency": dict(
            Counter(row["category"] for row in rows).most_common()
        ),
    }


def _exact_duplicate_groups(image_hashes: dict[str, str]) -> list[dict[str, Any]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for image_name, digest in image_hashes.items():
        grouped[digest].append(image_name)
    return [
        {"group_id": f"exact_{index:04d}", "sha256": digest, "images": sorted(names)}
        for index, (digest, names) in enumerate(
            sorted(
                (digest, names) for digest, names in grouped.items() if len(names) > 1
            ),
            start=1,
        )
    ]


def _near_duplicate_pairs(
    inventory: list[dict[str, Any]],
    maximum_distance: int,
    image_hashes: dict[str, str],
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    hashes = [
        (str(row["image_name"]), int(str(row["difference_hash"]), 16))
        for row in inventory
    ]
    for index, (left_name, left_hash) in enumerate(hashes):
        for right_name, right_hash in hashes[index + 1 :]:
            if image_hashes[left_name] == image_hashes[right_name]:
                continue
            distance = (left_hash ^ right_hash).bit_count()
            if distance <= maximum_distance:
                pairs.append(
                    {
                        "left_image": left_name,
                        "right_image": right_name,
                        "hamming_distance": distance,
                    }
                )
    return pairs


def _leakage_groups(
    image_names: set[str],
    exact_groups: list[dict[str, Any]],
    near_pairs: list[dict[str, Any]],
) -> dict[str, str]:
    parent = {name: name for name in image_names}

    def find(name: str) -> str:
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for group in exact_groups:
        names = [str(name) for name in group["images"]]
        for name in names[1:]:
            union(names[0], name)
    for pair in near_pairs:
        union(str(pair["left_image"]), str(pair["right_image"]))
    roots = sorted({find(name) for name in image_names})
    root_ids = {root: f"group_{index:04d}" for index, root in enumerate(roots, start=1)}
    return {name: root_ids[find(name)] for name in image_names}


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


def _count_distribution(rows: list[dict[str, str]]) -> dict[str, int]:
    counts = Counter(row["image_name"] for row in rows)
    return {
        str(question_count): image_count
        for question_count, image_count in sorted(Counter(counts.values()).items())
    }


def _dimension_summary(inventory: list[dict[str, Any]]) -> dict[str, Any]:
    if not inventory:
        return {"minimum": None, "maximum": None, "unique": {}}
    dimensions = Counter(f"{row['width']}x{row['height']}" for row in inventory)
    widths = [int(row["width"]) for row in inventory]
    heights = [int(row["height"]) for row in inventory]
    return {
        "minimum": [min(widths), min(heights)],
        "maximum": [max(widths), max(heights)],
        "unique": dict(sorted(dimensions.items())),
    }


def _alignment_row(label: str) -> dict[str, str]:
    class_ids, relation, note = DATASET_A_ALIGNMENT.get(
        label, ("", "unmatched", "No defensible Dataset A name match")
    )
    ids = class_ids.split("|") if class_ids else []
    names = " | ".join(DATASET_A_CLASS_NAMES[class_id] for class_id in ids)
    return {
        "dataset_a_id": class_ids,
        "dataset_a_name": names,
        "dataset_b_name": label,
        "alignment_status": "no_match" if relation == "unmatched" else relation,
        "notes": note,
    }


def _class_statistics(counts: Counter[str]) -> dict[str, Any]:
    if not counts:
        return {
            "median_images_per_class": None,
            "smallest_class": None,
            "largest_class": None,
        }
    ordered = sorted(counts.items(), key=lambda item: (item[1], item[0]))
    values = list(counts.values())
    return {
        "median_images_per_class": float(statistics.median(values)),
        "smallest_class": {"label": ordered[0][0], "images": ordered[0][1]},
        "largest_class": {"label": ordered[-1][0], "images": ordered[-1][1]},
    }


def _normalize_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _normalize_text(value: str) -> str:
    return " ".join(value.casefold().strip().rstrip("?.").split())


def _canonical_category(value: str) -> str:
    normalized = _normalize_text(value)
    aliases = {
        "cautionary": "Cautionary",
        "mandatory": "Mandatory",
        "informatory": "Informatory",
        "lnformatory": "Informatory",
    }
    return aliases.get(normalized, value.strip())


def _clean(value: str | None) -> str:
    return "" if value is None else " ".join(value.strip().split())
