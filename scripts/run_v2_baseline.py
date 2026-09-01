"""Run the locked Baseline V2 training, one-time test, and embedding diagnostic."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from data.road_sign_dataset import RoadSignDataset, RoadSignDatasetConfig  # noqa: E402
from evaluation.evaluator import Evaluator, save_evaluation_outputs  # noqa: E402
from models.factory import ModelConfig, build_classifier  # noqa: E402
from training.checkpoint import (  # noqa: E402
    CheckpointManager,
    CheckpointMetadata,
    load_checkpoint,
    read_checkpoint_payload,
)
from training.class_weights import normalized_inverse_frequency_weights  # noqa: E402
from training.factories import (  # noqa: E402
    LossConfig,
    OptimizerConfig,
    SchedulerConfig,
    build_optimizer,
    build_scheduler,
)
from training.run import (  # noqa: E402
    collect_environment_details,
    create_run_directory,
    create_run_id,
    save_run_metadata,
)
from training.trainer import Trainer  # noqa: E402
from training.transforms import (  # noqa: E402
    TransformConfig,
    build_evaluation_transform,
    build_train_transform,
)
from utils.config import load_yaml_config  # noqa: E402
from utils.reproducibility import seed_dataloader_worker, seed_everything  # noqa: E402

CONFIG_PATH = PROJECT_ROOT / "configs" / "v2_baseline.yaml"
DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "indian_traffic_vqa"
MANIFEST_DIR = PROJECT_ROOT / "outputs" / "manifests"
REVIEW_PATH = PROJECT_ROOT / "outputs" / "v2_review" / "dataset_b_v2_review.csv"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "v2_results"
EMBEDDING_DIR = PROJECT_ROOT / "outputs" / "embedding_analysis"
EXPECTED_SIZES = {"train": 287, "validation": 62, "test": 63}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _preflight(
    mapping: dict[str, int],
) -> tuple[dict[str, list[dict[str, str]]], dict[str, Any]]:
    review = _read_csv(REVIEW_PATH)
    status_counts = Counter(row["review_status"].strip().lower() for row in review)
    if len(review) != 530 or status_counts != Counter(
        {"approved": 412, "rejected": 118}
    ):
        raise RuntimeError(
            f"Locked review state differs: rows={len(review)}, statuses={dict(status_counts)}"
        )
    splits = {
        name: _read_csv(MANIFEST_DIR / f"v2_{name}.csv")
        for name in ("train", "validation", "test")
    }
    if {name: len(rows) for name, rows in splits.items()} != EXPECTED_SIZES:
        raise RuntimeError("Locked V2 split sizes differ from 287/62/63")
    if len(mapping) != 17 or sorted(mapping.values()) != list(range(17)):
        raise RuntimeError("V2 mapping is not a contiguous 17-class mapping")
    approved_ids = {
        row["review_id"] for row in review if row["review_status"] == "approved"
    }
    owners: dict[str, dict[str, str]] = {
        key: {} for key in ("review_id", "source_image_id", "perceptual_group_id")
    }
    labels: set[str] = set()
    for split, rows in splits.items():
        for row in rows:
            labels.add(row["class_name"])
            if row["review_id"] not in approved_ids:
                raise RuntimeError(f"Non-approved row in {split}: {row['review_id']}")
            if row["split"] != split:
                raise RuntimeError(f"Split marker mismatch: {row['review_id']}")
            for key in owners:
                value = row[key]
                previous = owners[key].get(value)
                if previous is not None and previous != split:
                    raise RuntimeError(
                        f"{key} leakage between {previous} and {split}: {value}"
                    )
                owners[key][value] = split
    if labels != set(mapping):
        raise RuntimeError("Split labels differ from the locked class mapping")
    paths = [
        CONFIG_PATH,
        REVIEW_PATH,
        *(MANIFEST_DIR / f"v2_{name}.csv" for name in splits),
        MANIFEST_DIR / "v2_class_mapping.json",
        MANIFEST_DIR / "v2_split_summary.json",
    ]
    report = {
        "passed": True,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "review_status_counts": dict(status_counts),
        "split_sizes": {name: len(rows) for name, rows in splits.items()},
        "class_count": len(mapping),
        "leakage": {
            "review_id": False,
            "source_image_id": False,
            "perceptual_group_id": False,
        },
        "locked_file_sha256": {
            str(path.relative_to(PROJECT_ROOT)): _sha256(path) for path in paths
        },
        "test_image_policy": "Manifest metadata validated pre-training; test images were not decoded or inferred until best.pt selection completed.",
    }
    return splits, report


def _dataset(
    manifest: Path,
    split: str,
    mapping: dict[str, int],
    transform: Any,
    *,
    metadata: bool = False,
) -> RoadSignDataset:
    return RoadSignDataset(
        RoadSignDatasetConfig(
            root=DATASET_ROOT,
            mode="csv_manifest",
            manifest_path=manifest,
            image_path_column="image_path",
            label_column="class_name",
            split_column="split",
            split=split,
            return_metadata=metadata,
        ),
        transform=transform,
        class_mapping=mapping,
    )


def _loader(dataset: RoadSignDataset, *, shuffle: bool, seed: int) -> DataLoader[Any]:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=32,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
        worker_init_fn=seed_dataloader_worker,
        generator=generator,
    )


def _sanity_check(
    model_config: ModelConfig,
    mapping: dict[str, int],
    train_loader: DataLoader[Any],
    weights: torch.Tensor,
) -> dict[str, Any]:
    model = build_classifier(model_config, mapping).to("cpu")
    optimizer = build_optimizer(
        model.parameters(), OptimizerConfig(learning_rate=0.0005, weight_decay=0.0001)
    )
    loss_function = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    images, targets = next(iter(train_loader))[:2]
    image_shape = list(images.shape)
    label_min = int(targets.min().item())
    label_max = int(targets.max().item())
    if image_shape != [images.shape[0], 3, 224, 224]:
        raise RuntimeError(f"Unexpected real-batch image shape: {image_shape}")
    if label_min < 0 or label_max > 16:
        raise RuntimeError(f"Real-batch labels outside 0..16: {label_min}..{label_max}")
    optimizer.zero_grad(set_to_none=True)
    logits = model(images)
    loss = loss_function(logits, targets)
    loss.backward()
    optimizer.step()
    if logits.shape != (images.shape[0], 17) or not torch.isfinite(loss):
        raise RuntimeError("Real-batch optimizer sanity check failed")
    return {
        "passed": True,
        "batch_size": int(images.shape[0]),
        "image_tensor_shape": image_shape,
        "label_min": label_min,
        "label_max": label_max,
        "labels_within_0_to_16": True,
        "logits_shape": list(logits.shape),
        "loss": float(loss.detach()),
        "loss_finite": True,
        "backward_pass": True,
        "optimizer_step": True,
    }


def _embedding_diagnostic(
    model: Any, loader: DataLoader[Any], mapping: dict[str, int]
) -> dict[str, Any]:
    model.eval()
    embeddings: list[np.ndarray] = []
    labels: list[int] = []
    review_ids: list[str] = []
    with torch.no_grad():
        for images, targets, metadata in loader:
            embeddings.append(model.extract_embeddings(images).cpu().numpy())
            labels.extend(int(value) for value in targets.tolist())
            review_ids.extend(str(value) for value in metadata["review_id"])
    matrix = np.concatenate(embeddings, axis=0)
    label_array = np.asarray(labels)
    within: list[float] = []
    between: list[float] = []
    for left in range(len(matrix)):
        similarities = matrix[left + 1 :] @ matrix[left]
        same = label_array[left + 1 :] == label_array[left]
        within.extend(float(value) for value in similarities[same])
        between.extend(float(value) for value in similarities[~same])
    centroids = np.stack(
        [matrix[label_array == index].mean(axis=0) for index in range(len(mapping))]
    )
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    nearest = (matrix @ centroids.T).argmax(axis=1)
    return {
        "sample_count": len(matrix),
        "embedding_dimension": int(matrix.shape[1]),
        "normalized_embedding_norm_mean": float(np.linalg.norm(matrix, axis=1).mean()),
        "mean_within_class_cosine_similarity": float(np.mean(within)),
        "mean_between_class_cosine_similarity": float(np.mean(between)),
        "nearest_centroid_closed_set_accuracy": float(np.mean(nearest == label_array)),
        "nearest_centroid_protocol": "Diagnostic resubstitution: centroids are calculated from the same locked test embeddings, including each query; not an unbiased performance estimate.",
        "review_ids": review_ids,
    }


def main() -> int:
    """Execute the complete locked Baseline V2 protocol."""
    config = load_yaml_config(CONFIG_PATH)
    mapping = json.loads(
        (MANIFEST_DIR / "v2_class_mapping.json").read_text(encoding="utf-8")
    )
    splits, preflight = _preflight(mapping)
    RESULTS_DIR.mkdir(parents=True, exist_ok=False)
    (RESULTS_DIR / "pretraining_verification.json").write_text(
        json.dumps(preflight, indent=2), encoding="utf-8"
    )
    counts = Counter(row["class_name"] for row in splits["train"])
    weight_map = normalized_inverse_frequency_weights(counts)
    class_weight_report = {
        "formula": "N / (C * n_c)",
        "source": "outputs/manifests/v2_train.csv only",
        "total_samples": sum(counts.values()),
        "class_count": len(counts),
        "counts": dict(sorted(counts.items())),
        "weights": weight_map,
    }
    (RESULTS_DIR / "class_weights.json").write_text(
        json.dumps(class_weight_report, indent=2), encoding="utf-8"
    )
    seed_everything(42, True)
    transform_config = TransformConfig(
        image_size=224,
        horizontal_flip_probability=0.0,
        max_rotation_degrees=7.0,
        brightness=0.20,
        contrast=0.20,
    )
    train_dataset = _dataset(
        MANIFEST_DIR / "v2_train.csv",
        "train",
        mapping,
        build_train_transform(transform_config),
    )
    validation_dataset = _dataset(
        MANIFEST_DIR / "v2_validation.csv",
        "validation",
        mapping,
        build_evaluation_transform(transform_config),
    )
    weights = torch.tensor(
        [
            weight_map[label]
            for label, _ in sorted(mapping.items(), key=lambda item: item[1])
        ],
        dtype=torch.float32,
    )
    model_config = ModelConfig(
        backbone="mobilenetv3_small_100", pretrained=True, num_classes=17, dropout=0.2
    )
    sanity = _sanity_check(
        model_config, mapping, _loader(train_dataset, shuffle=True, seed=42), weights
    )
    (RESULTS_DIR / "real_batch_sanity.json").write_text(
        json.dumps(sanity, indent=2), encoding="utf-8"
    )
    seed_everything(42, True)
    model = build_classifier(model_config, mapping)
    optimizer = build_optimizer(
        model.parameters(), OptimizerConfig(learning_rate=0.0005, weight_decay=0.0001)
    )
    scheduler = build_scheduler(
        optimizer,
        SchedulerConfig(name="cosine", minimum_learning_rate=0.000001),
        epochs=30,
    )
    loss_function = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    run_id = create_run_id("v2_mobilenetv3_small_100")
    run_directory = create_run_directory(PROJECT_ROOT / "outputs" / "runs", run_id)
    checkpoint_manager = CheckpointManager(
        PROJECT_ROOT / "outputs" / "checkpoints" / run_id, dataset_root=DATASET_ROOT
    )
    environment = collect_environment_details(
        selected_device="cpu", dataset_sizes=EXPECTED_SIZES, class_count=17
    )
    environment.update(
        {"deterministic_algorithms": True, "test_images_loaded_before_selection": False}
    )
    save_run_metadata(run_directory, resolved_config=config, environment=environment)
    metadata = CheckpointMetadata(
        class_mapping=mapping,
        model_config=asdict(model_config),
        preprocessing_config=asdict(transform_config),
        random_seed=42,
        training_config={**config["training"], "class_weights": class_weight_report},
        project_metadata=config["project"],
    )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_function=loss_function,
        device=torch.device("cpu"),
        scheduler=scheduler,
        progress_callback=print,
    )
    fit = trainer.fit(
        _loader(train_dataset, shuffle=True, seed=42),
        _loader(validation_dataset, shuffle=False, seed=43),
        epochs=30,
        selection_metric="validation_macro_f1",
        checkpoint_manager=checkpoint_manager,
        checkpoint_metadata=metadata,
    )
    fit.history.save(run_directory)
    if (
        fit.best_epoch is None
        or not (checkpoint_manager.directory / "best.pt").exists()
    ):
        raise RuntimeError("Training completed without a selected best checkpoint")
    # The locked test images are first decoded only after all 30 epochs and selection.
    test_dataset = _dataset(
        MANIFEST_DIR / "v2_test.csv",
        "test",
        mapping,
        build_evaluation_transform(transform_config),
        metadata=True,
    )
    test_loader = _loader(test_dataset, shuffle=False, seed=44)
    best_path = checkpoint_manager.directory / "best.pt"
    load_checkpoint(
        best_path,
        model=model,
        expected_class_mapping=mapping,
        expected_model_config=asdict(model_config),
        map_location="cpu",
    )
    result = Evaluator(
        model=model, device=torch.device("cpu"), class_mapping=mapping
    ).evaluate(test_loader)
    save_evaluation_outputs(result, RESULTS_DIR, dataset_root=DATASET_ROOT)
    payload = read_checkpoint_payload(best_path)
    matrix = result.metrics.confusion_matrix
    ordered_labels = [
        label for label, _ in sorted(mapping.items(), key=lambda item: item[1])
    ]
    confusions = sorted(
        (
            {
                "true_label": ordered_labels[i],
                "predicted_label": ordered_labels[j],
                "count": matrix[i][j],
            }
            for i in range(17)
            for j in range(17)
            if i != j and matrix[i][j]
        ),
        key=lambda item: (-item["count"], item["true_label"], item["predicted_label"]),
    )
    summary = {
        "run_id": run_id,
        "best_epoch_zero_based": fit.best_epoch,
        "best_epoch_one_based": fit.best_epoch + 1,
        "best_validation_macro_f1": fit.best_validation_macro_f1,
        "best_validation_accuracy": fit.best_validation_accuracy,
        "best_checkpoint_epoch": int(payload["epoch"]),
        "test_metrics": result.metrics.to_dict(),
        "correct": sum(item.correct for item in result.predictions),
        "incorrect": sum(not item.correct for item in result.predictions),
        "most_common_confusions": confusions[:10],
        "checkpoint_sha256": _sha256(best_path),
    }
    (RESULTS_DIR / "v2_experiment_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)
    embedding = _embedding_diagnostic(model, test_loader, mapping)
    (EMBEDDING_DIR / "v2_embedding_summary.json").write_text(
        json.dumps(embedding, indent=2), encoding="utf-8"
    )
    v1_metrics_path = (
        PROJECT_ROOT
        / "outputs"
        / "evaluations"
        / "20260831_162238_646352_mobilenetv3_small_100"
        / "metrics.json"
    )
    v1 = json.loads(v1_metrics_path.read_text(encoding="utf-8"))
    comparison = f"""# Baseline V1 vs Baseline V2\n\n| Item | V1 | V2 |\n|---|---:|---:|\n| Classes | 5 | 17 |\n| Test samples | {sum(row['support'] for row in v1['per_class'])} | {len(result.predictions)} |\n| Top-1 accuracy | {v1['top1_accuracy']:.6f} | {result.metrics.top1_accuracy:.6f} |\n| Macro F1 | {v1['macro_f1']:.6f} | {result.metrics.macro_f1:.6f} |\n\nThe datasets and class sets differ, so this is descriptive rather than a controlled head-to-head comparison. V2 used validation macro F1 for checkpoint selection and evaluated the locked test split once afterward.\n"""
    (RESULTS_DIR / "v1_vs_v2_comparison.md").write_text(comparison, encoding="utf-8")
    (RESULTS_DIR / "run_artifacts.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "run_directory": str(run_directory.relative_to(PROJECT_ROOT)),
                "checkpoint_directory": str(
                    checkpoint_manager.directory.relative_to(PROJECT_ROOT)
                ),
                "best_checkpoint": str(best_path.relative_to(PROJECT_ROOT)),
                "test_evaluated_once": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "run_id": run_id,
                "best_epoch": fit.best_epoch + 1,
                "test_accuracy": result.metrics.top1_accuracy,
                "test_macro_f1": result.metrics.macro_f1,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
