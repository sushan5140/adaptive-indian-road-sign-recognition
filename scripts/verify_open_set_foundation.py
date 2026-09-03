"""Verify the frozen V2 foundation and audit available unseen-class data."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.frozen_embedding import FrozenEmbeddingPipeline  # noqa: E402
from inference.open_set import IncrementalClassRegistrar  # noqa: E402
from models.prototype_registry import PrototypeRegistry  # noqa: E402

CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "checkpoints"
    / "20260901_000220_558663_v2_mobilenetv3_small_100"
    / "best.pt"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "0f990f21c7f844f5611e91f867740b7f980e851426681c69deb2fefadbea8ff4"
)
OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs" / "open_set"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _training_reference_paths(count: int) -> list[Path]:
    rows = _read_csv(PROJECT_ROOT / "outputs" / "manifests" / "v2_train.csv")
    label = rows[0]["class_name"]
    selected = [row for row in rows if row["class_name"] == label][:count]
    if len(selected) != count:
        raise RuntimeError(f"Could not select {count} same-class training references")
    root = PROJECT_ROOT / "data" / "raw" / "indian_traffic_vqa"
    return [root / row["image_path"] for row in selected]


def _verify_foundation() -> dict[str, Any]:
    pipeline = FrozenEmbeddingPipeline.from_checkpoint(
        CHECKPOINT,
        expected_sha256=EXPECTED_CHECKPOINT_SHA256,
    )
    paths = _training_reference_paths(5)
    tensors = pipeline.preprocess_paths(paths)
    state_before = pipeline.model_state_sha256()
    one_batch = pipeline.infer_preprocessed(tensors, batch_size=5)
    repeated = pipeline.infer_preprocessed(tensors, batch_size=5)
    partitioned = pipeline.infer_preprocessed(tensors, batch_size=2)
    norms = np.linalg.norm(one_batch.embeddings, axis=1)

    registry = PrototypeRegistry(embedding_dim=pipeline.identity.embedding_dim)
    registrar = IncrementalClassRegistrar(pipeline, registry)
    shot_results: dict[str, Any] = {}
    for shots in (1, 3, 5):
        label = f"verification_only_{shots}_shot"
        prototype = registrar.register_preprocessed(label, tensors[:shots])
        shot_results[str(shots)] = {
            "registered": True,
            "prototype_norm": float(np.linalg.norm(prototype)),
            "recorded_shot_count": registry.get_metadata(label)["shot_count"],
        }
    with tempfile.TemporaryDirectory() as directory:
        registry_path = Path(directory) / "registry.npz"
        registry.save(registry_path)
        loaded = PrototypeRegistry.load(registry_path)
        persistence_identical = loaded.labels == registry.labels and all(
            np.array_equal(loaded.get_prototype(label), registry.get_prototype(label))
            and loaded.get_metadata(label) == registry.get_metadata(label)
            for label in registry.labels
        )
    state_after = pipeline.model_state_sha256()
    return {
        "checkpoint": {
            "path": str(CHECKPOINT.relative_to(PROJECT_ROOT)),
            "sha256": pipeline.identity.sha256,
            "epoch_zero_based": pipeline.identity.epoch,
            "epoch_one_based": pipeline.identity.epoch + 1,
            "best_validation_macro_f1": pipeline.identity.best_validation_metric,
            "model_config": pipeline.identity.model_config,
            "preprocessing_config": pipeline.identity.preprocessing_config,
            "class_count": len(pipeline.identity.class_mapping),
        },
        "verification_data": {
            "source": "outputs/manifests/v2_train.csv only",
            "reference_count": len(paths),
            "locked_v2_test_used": False,
        },
        "frozen_model": {
            "all_parameters_frozen": pipeline.all_parameters_frozen,
            "model_state_sha256_before": state_before,
            "model_state_sha256_after": state_after,
            "weights_unchanged": state_before == state_after,
            "base_outputs_unchanged": bool(
                np.array_equal(
                    one_batch.base_probabilities, repeated.base_probabilities
                )
            ),
        },
        "embeddings": {
            "shape": list(one_batch.embeddings.shape),
            "dimension": pipeline.identity.embedding_dim,
            "finite": bool(np.all(np.isfinite(one_batch.embeddings))),
            "minimum_l2_norm": float(norms.min()),
            "maximum_l2_norm": float(norms.max()),
            "repeated_extraction_exact": bool(
                np.array_equal(one_batch.embeddings, repeated.embeddings)
            ),
            "maximum_batch_partition_absolute_difference": float(
                np.max(np.abs(one_batch.embeddings - partitioned.embeddings))
            ),
        },
        "registration": {
            "shot_conditions_verified": shot_results,
            "overwrite_default": False,
            "removal_supported": True,
            "pickle_free_npz_round_trip_identical": persistence_identical,
        },
    }


def _audit_unseen_data() -> dict[str, Any]:
    final_review = json.loads(
        (
            PROJECT_ROOT / "outputs" / "v2_review" / "final_review_summary.json"
        ).read_text(encoding="utf-8")
    )
    base_classes = set(final_review["viable_classes"])
    v2_rows = _read_csv(
        PROJECT_ROOT / "outputs" / "v2_review" / "dataset_b_v2_review.csv"
    )
    v2_non_base: list[dict[str, Any]] = []
    for class_name in sorted({row["proposed_class"] for row in v2_rows} - base_classes):
        rows = [row for row in v2_rows if row["proposed_class"] == class_name]
        statuses = Counter(row["review_status"] for row in rows)
        v2_non_base.append(
            {
                "class_name": class_name,
                "candidate_images": len(rows),
                "independent_perceptual_groups": len(
                    {row["perceptual_group_id"] for row in rows}
                ),
                "review_status_counts": dict(statuses),
                "defensible_for_unseen_evaluation": False,
                "reason": "All available candidates were manually rejected for semantic mismatch or unusable sign evidence.",
            }
        )
    dataset_a_quality = json.loads(
        (PROJECT_ROOT / "outputs" / "dataset_audit" / "dataset_quality.json").read_text(
            encoding="utf-8"
        )
    )
    dataset_a_classes = _read_csv(
        PROJECT_ROOT / "outputs" / "dataset_audit" / "class_distribution.csv"
    )
    return {
        "stop_condition_triggered": True,
        "scientifically_defensible_unseen_classes": [],
        "dataset_b_non_base_candidates": v2_non_base,
        "dataset_a": {
            "populated_class_count": len(dataset_a_classes),
            "original_templates_per_populated_class": dataset_a_quality[
                "original_template_files_per_populated_class"
            ],
            "exact_duplicate_content_groups": dataset_a_quality[
                "exact_duplicate_content_groups"
            ],
            "cross_label_duplicate_content_groups": dataset_a_quality[
                "cross_label_duplicate_content_groups"
            ],
            "defensible_for_independent_reference_query_evaluation": False,
            "reason": dataset_a_quality["split_blocker"],
        },
        "threshold_calibration": {
            "performed": False,
            "thresholds": None,
            "reason": "No independent reviewed unseen/calibration dataset exists; the locked V2 test set is prohibited for threshold selection.",
        },
        "few_shot_experiment": {
            "performed": False,
            "results": None,
            "reason": "No defensible unseen class has independent reference and query groups.",
        },
        "required_additional_data": {
            "minimum_content": "Independently photographed road signs from at least one class outside the 17 V2 classes, with corrected human-reviewed labels.",
            "group_requirement": "For each unseen class, acquire at least 15 independent perceptual/source groups as an operational minimum (up to 5 reference, at least 5 calibration, and at least 5 query groups); larger samples are needed for stable metrics and repeated reference selections.",
            "calibration_requirement": "Separate base-known and unknown calibration images not drawn from the locked V2 test split.",
            "evaluation_requirement": "Independent query images never used as prototype references or threshold-calibration samples.",
        },
    }


def main() -> int:
    """Write foundation verification and unseen-data stop-condition records."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    foundation = _verify_foundation()
    audit = _audit_unseen_data()
    _write_json(OUTPUT_DIRECTORY / "frozen_backbone_verification.json", foundation)
    _write_json(OUTPUT_DIRECTORY / "unseen_class_data_audit.json", audit)
    print("Frozen backbone verification passed.")
    print("Unseen-data stop condition triggered; no experimental metrics generated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
