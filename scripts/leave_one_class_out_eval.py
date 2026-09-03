"""Leave-one-class-out proxy for the untested "recognizes a brand-new sign" claim.

The project's central feature — registering a sign the base model never saw,
from a handful of reference photos, and then recognizing it — has never been
measured end to end. Getting real new-sign photographs (a Stop sign, a No
Parking sign) is still the honest long-term fix, but this script lets that
claim be measured *today*, using only images already committed to this repo,
by treating one of the 17 known classes as if it were unseen:

1. Remove one class entirely from the training and validation manifests and
   retrain a 16-class base model from scratch (same recipe as
   ``configs/v2_baseline.yaml``: MobileNetV3-Small, AdamW, cosine schedule,
   class-balanced loss, best checkpoint by validation macro F1).
2. Register that held-out class as a new sign, using only a handful of its
   *training-split* images as reference photos.
3. Evaluate three things separately, so the report cannot conflate them:
   - recognition rate: do the held-out class's remaining (non-reference)
     images get correctly verdicted as the newly registered class?
   - false-accept rate: do any of the 16 known classes' test images get
     mis-claimed by the new prototype instead of their own base class?
   - closed-set accuracy on the 16 remaining classes, to confirm removing one
     class did not silently break the rest.

No image used for registration or evaluation is ever also used for training —
that boundary is enforced by construction: registration draws only from rows
whose split is "train" for the held-out class, and those rows are excluded
from the retrained model's own training set.

This still requires the raw Dataset B images under
``data/raw/indian_traffic_vqa/`` (gitignored, manually acquired from Zenodo
record 10.5281/zenodo.17300841) to run. It is proof-of-concept evidence, not a
substitute for evaluating on a sign class the model has truly never seen in
any form — a real Stop/No Parking photo set remains the strongest version of
this experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from collections import Counter  # noqa: E402

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from data.road_sign_dataset import RoadSignDataset, RoadSignDatasetConfig  # noqa: E402
from inference.decision import OpenSetThresholds, Verdict  # noqa: E402
from inference.pipeline import OpenSetRecognizer  # noqa: E402
from models.factory import ModelConfig, build_classifier  # noqa: E402
from training.checkpoint import CheckpointManager, CheckpointMetadata  # noqa: E402
from training.class_weights import normalized_inverse_frequency_weights  # noqa: E402
from training.factories import (  # noqa: E402
    OptimizerConfig,
    SchedulerConfig,
    build_optimizer,
    build_scheduler,
)
from training.trainer import Trainer  # noqa: E402
from training.transforms import (  # noqa: E402
    TransformConfig,
    build_evaluation_transform,
    build_train_transform,
)
from utils.reproducibility import seed_dataloader_worker, seed_everything  # noqa: E402

DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "indian_traffic_vqa"
MANIFEST_DIR = PROJECT_ROOT / "outputs" / "manifests"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "loco_results"
EPOCHS = 30
REGISTRATION_K = 5
SEED = 42
PRETRAINED = True  # module-level so smoke tests can force False (no network needed)


def _relative_or_absolute(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _load_rows(split_name: str) -> list[dict[str, str]]:
    path = MANIFEST_DIR / f"v2_{split_name}.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_manifest(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["image_path", "class_name", "split"]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "image_path": row["image_path"],
                    "class_name": row["class_name"],
                    "split": row["split"],
                }
            )


def _dataset(
    manifest: Path, split: str, mapping: dict[str, int], transform: Any
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
        ),
        transform=transform,
        class_mapping=mapping,
    )


def _loader(dataset: RoadSignDataset, *, shuffle: bool, seed: int) -> DataLoader[Any]:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=16,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
        worker_init_fn=seed_dataloader_worker,
        generator=generator,
    )


def _train_holdout_model(holdout_class: str, work_dir: Path) -> Path:
    """Train a 16-class model with ``holdout_class`` fully excluded. Returns the best checkpoint path."""
    train_rows = [r for r in _load_rows("train") if r["class_name"] != holdout_class]
    val_rows = [r for r in _load_rows("validation") if r["class_name"] != holdout_class]

    remaining_classes = sorted({r["class_name"] for r in train_rows})
    class_mapping = {name: index for index, name in enumerate(remaining_classes)}
    num_classes = len(remaining_classes)

    manifest_path = work_dir / f"loco_{holdout_class}_manifest.csv"
    _write_manifest(train_rows + val_rows, manifest_path)

    seed_everything(SEED, deterministic=True)
    transform_config = TransformConfig(
        image_size=224,
        horizontal_flip_probability=0.0,
        max_rotation_degrees=7.0,
        brightness=0.20,
        contrast=0.20,
    )
    train_dataset = _dataset(
        manifest_path, "train", class_mapping, build_train_transform(transform_config)
    )
    val_dataset = _dataset(
        manifest_path,
        "validation",
        class_mapping,
        build_evaluation_transform(transform_config),
    )

    counts = Counter(r["class_name"] for r in train_rows)
    weight_map = normalized_inverse_frequency_weights(counts)
    weights = torch.tensor(
        [
            weight_map[label]
            for label, _ in sorted(class_mapping.items(), key=lambda item: item[1])
        ],
        dtype=torch.float32,
    )

    model_config = ModelConfig(
        backbone="mobilenetv3_small_100",
        pretrained=PRETRAINED,
        num_classes=num_classes,
        dropout=0.2,
    )
    seed_everything(SEED, deterministic=True)
    model = build_classifier(model_config, class_mapping)
    optimizer = build_optimizer(
        model.parameters(), OptimizerConfig(learning_rate=0.0005, weight_decay=0.0001)
    )
    scheduler = build_scheduler(
        optimizer,
        SchedulerConfig(name="cosine", minimum_learning_rate=0.000001),
        epochs=EPOCHS,
    )
    loss_function = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)

    checkpoint_dir = work_dir / f"loco_{holdout_class}_checkpoint"
    checkpoint_manager = CheckpointManager(checkpoint_dir, dataset_root=DATASET_ROOT)
    metadata = CheckpointMetadata(
        class_mapping=class_mapping,
        model_config=asdict(model_config),
        preprocessing_config=asdict(transform_config),
        random_seed=SEED,
        training_config={"epochs": EPOCHS, "holdout_class": holdout_class},
        project_metadata={"experiment": "leave_one_class_out"},
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
        _loader(train_dataset, shuffle=True, seed=SEED),
        _loader(val_dataset, shuffle=False, seed=SEED + 1),
        epochs=EPOCHS,
        selection_metric="validation_macro_f1",
        checkpoint_manager=checkpoint_manager,
        checkpoint_metadata=metadata,
    )
    best_path = checkpoint_manager.directory / "best.pt"
    if fit.best_epoch is None or not best_path.exists():
        raise RuntimeError(
            f"[{holdout_class}] training completed without a selected best checkpoint"
        )
    print(
        f"[{holdout_class}] best epoch {fit.best_epoch + 1}/{EPOCHS}, "
        f"val_macro_f1={fit.best_validation_macro_f1:.4f}"
    )
    return best_path


def _evaluate(
    holdout_class: str, checkpoint_path: Path, registration_k: int
) -> dict[str, Any]:
    thresholds = OpenSetThresholds(
        calibrated=False
    )  # placeholders remain honest about this
    recognizer = OpenSetRecognizer.from_checkpoint(
        checkpoint_path, thresholds=thresholds, device="cpu"
    )

    holdout_train_rows = [
        r for r in _load_rows("train") if r["class_name"] == holdout_class
    ]
    if len(holdout_train_rows) <= registration_k:
        raise ValueError(
            f"'{holdout_class}' has only {len(holdout_train_rows)} training images; "
            f"need more than registration_k={registration_k} so evaluation images remain."
        )
    registration_rows = holdout_train_rows[:registration_k]
    leftover_train_rows = holdout_train_rows[registration_k:]

    registration_paths = [DATASET_ROOT / r["image_path"] for r in registration_rows]
    recognizer.register_sign_from_paths(
        holdout_class,
        registration_paths,
        persist=False,
        metadata={"experiment": "leave_one_class_out", "reference_split": "train"},
    )

    holdout_eval_rows = (
        leftover_train_rows
        + [r for r in _load_rows("validation") if r["class_name"] == holdout_class]
        + [r for r in _load_rows("test") if r["class_name"] == holdout_class]
    )
    holdout_paths = [DATASET_ROOT / r["image_path"] for r in holdout_eval_rows]
    holdout_decisions = recognizer.predict_paths(holdout_paths) if holdout_paths else []
    recognized = sum(
        1
        for d in holdout_decisions
        if d.verdict is Verdict.REGISTERED_CLASS and d.label == holdout_class
    )
    misrouted_to_base = sum(
        1 for d in holdout_decisions if d.verdict is Verdict.BASE_CLASS
    )
    correctly_unknown = sum(
        1 for d in holdout_decisions if d.verdict is Verdict.UNKNOWN
    )

    known_test_rows = [
        r for r in _load_rows("test") if r["class_name"] != holdout_class
    ]
    known_paths = [DATASET_ROOT / r["image_path"] for r in known_test_rows]
    known_labels = [r["class_name"] for r in known_test_rows]
    known_decisions = recognizer.predict_paths(known_paths) if known_paths else []
    false_accepts = sum(
        1 for d in known_decisions if d.verdict is Verdict.REGISTERED_CLASS
    )
    known_correct = sum(
        1
        for d, true_label in zip(known_decisions, known_labels, strict=True)
        if d.verdict is Verdict.BASE_CLASS and d.label == true_label
    )

    # Raw per-image scores, for pooling into scripts/calibrate_thresholds.py.
    # is_novel=True rows are the held-out class (should be accepted by the
    # prototype); is_novel=False rows are known classes (should NOT be).
    per_image_records = [
        {
            "image_path": str(row["image_path"]),
            "is_novel": True,
            "true_label": holdout_class,
            "verdict": decision.verdict.value,
            "predicted_label": decision.label,
            "base_confidence": decision.base.confidence,
            "prototype_similarity": decision.prototype.similarity,
        }
        for row, decision in zip(holdout_eval_rows, holdout_decisions, strict=True)
    ] + [
        {
            "image_path": str(row["image_path"]),
            "is_novel": False,
            "true_label": row["class_name"],
            "verdict": decision.verdict.value,
            "predicted_label": decision.label,
            "base_confidence": decision.base.confidence,
            "prototype_similarity": decision.prototype.similarity,
        }
        for row, decision in zip(known_test_rows, known_decisions, strict=True)
    ]
    records_path = RESULTS_DIR / f"{holdout_class}_predictions.csv"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with records_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_image_records[0].keys()))
        writer.writeheader()
        writer.writerows(per_image_records)

    return {
        "holdout_class": holdout_class,
        "registration_k": registration_k,
        "registration_source": "train-split images, excluded from this model's own training set",
        "holdout_eval_images": len(holdout_eval_rows),
        "holdout_recognized_as_registered": recognized,
        "holdout_recognition_rate": (
            recognized / len(holdout_eval_rows) if holdout_eval_rows else None
        ),
        "holdout_misrouted_to_a_base_class": misrouted_to_base,
        "holdout_correctly_flagged_unknown": correctly_unknown,
        "known_class_test_images": len(known_test_rows),
        "known_class_false_accepted_by_new_prototype": false_accepts,
        "known_class_false_accept_rate": (
            false_accepts / len(known_test_rows) if known_test_rows else None
        ),
        "known_class_closed_set_accuracy_with_class_removed": (
            known_correct / len(known_test_rows) if known_test_rows else None
        ),
        "per_image_predictions_csv": _relative_or_absolute(records_path),
        "note": "thresholds.calibrated is False: this uses uncalibrated placeholder thresholds from config.yaml. Run scripts/calibrate_thresholds.py (optionally after this) for a measured threshold instead.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--holdout-classes",
        required=True,
        help="Comma-separated class names to test, e.g. school_ahead,no_entry",
    )
    parser.add_argument("--registration-k", type=int, default=REGISTRATION_K)
    parser.add_argument(
        "--work-dir", type=Path, default=PROJECT_ROOT / "outputs" / "loco_work"
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for holdout_class in [
        c.strip() for c in args.holdout_classes.split(",") if c.strip()
    ]:
        checkpoint_path = _train_holdout_model(holdout_class, args.work_dir)
        result = _evaluate(holdout_class, checkpoint_path, args.registration_k)
        all_results.append(result)
        out_path = RESULTS_DIR / f"{holdout_class}.json"
        out_path.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))

    summary_path = RESULTS_DIR / "summary.json"
    summary_path.write_text(json.dumps(all_results, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
