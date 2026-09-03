"""Retrain Baseline V2's recipe on the training pool plus the HF supplement.

Identical architecture, optimizer, schedule, augmentation, loss, seed, epoch
count and selection metric as ``scripts/run_v2_baseline.py``. The only change is
the training pool: the 287 Dataset B training images plus the 771 base-class
mapped rows of ``outputs/manifests/hf_supplement.csv``, giving 1,058.

Validation and test manifests are read verbatim and never written. The locked
test split is decoded only after all epochs and checkpoint selection are done,
matching the V2 protocol.

Results go to ``outputs/v3_results/``. Nothing under ``outputs/v2_results/`` or
``outputs/manifests/v2_*.csv`` is touched.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from data.road_sign_dataset import RoadSignDataset, RoadSignDatasetConfig  # noqa: E402
from evaluation.evaluator import Evaluator, save_evaluation_outputs  # noqa: E402
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

DATA_ROOT = PROJECT_ROOT / "data" / "raw"
VQA_SUBDIR = "indian_traffic_vqa"
HF_SUBDIR = "hf_indian_traffic_sign"
MANIFEST_DIR = PROJECT_ROOT / "outputs" / "manifests"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "v3_results"
WORK_DIR = PROJECT_ROOT / "outputs" / "v3_work"
EPOCHS = 30
SEED = 42
EXPECTED = {"train": 1058, "validation": 62, "test": 63}


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def build_merged_manifest(path: Path) -> dict[str, int]:
    """Write a train+validation manifest rooted at data/raw."""
    rows: list[dict[str, str]] = []
    for record in _read(MANIFEST_DIR / "v2_train.csv"):
        rows.append(
            {
                "image_path": f"{VQA_SUBDIR}/{record['image_path']}",
                "class_name": record["class_name"],
                "split": "train",
                "source": "dataset_b",
            }
        )
    supplement = [
        r
        for r in _read(MANIFEST_DIR / "hf_supplement.csv")
        if r["maps_to_base_class"] == "yes"
    ]
    for record in supplement:
        rows.append(
            {
                "image_path": f"{HF_SUBDIR}/{record['image_path']}",
                "class_name": record["class_name"],
                "split": "train",
                "source": "hf_supplement",
            }
        )
    for record in _read(MANIFEST_DIR / "v2_validation.csv"):
        rows.append(
            {
                "image_path": f"{VQA_SUBDIR}/{record['image_path']}",
                "class_name": record["class_name"],
                "split": "validation",
                "source": "dataset_b",
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["image_path", "class_name", "split", "source"]
        )
        writer.writeheader()
        writer.writerows(rows)
    counts = Counter(r["split"] for r in rows)
    by_source = Counter(r["source"] for r in rows if r["split"] == "train")
    print(f"merged manifest -> {path.relative_to(PROJECT_ROOT).as_posix()}")
    print(
        f"  train {counts['train']} (dataset_b {by_source['dataset_b']}, "
        f"hf_supplement {by_source['hf_supplement']}), validation {counts['validation']}"
    )
    return dict(counts)


def _dataset(
    manifest: Path, split: str, mapping: dict[str, int], transform: Any, root: Path
) -> RoadSignDataset:
    return RoadSignDataset(
        RoadSignDatasetConfig(
            root=root,
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
    return DataLoader(
        dataset,
        batch_size=32,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
        worker_init_fn=seed_dataloader_worker,
        generator=torch.Generator().manual_seed(seed),
    )


def main() -> int:
    """Train on the supplemented pool and evaluate once on the locked test split."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    merged = WORK_DIR / "v3_train_validation.csv"
    counts = build_merged_manifest(merged)
    if counts["train"] != EXPECTED["train"]:
        raise SystemExit(
            f"expected {EXPECTED['train']} training rows, got {counts['train']}"
        )

    mapping = json.loads((MANIFEST_DIR / "v2_class_mapping.json").read_text())
    print(f"classes: {len(mapping)}")

    transform_config = TransformConfig(
        image_size=224,
        horizontal_flip_probability=0.0,
        max_rotation_degrees=7.0,
        brightness=0.20,
        contrast=0.20,
    )

    seed_everything(SEED, True)
    train_dataset = _dataset(
        merged, "train", mapping, build_train_transform(transform_config), DATA_ROOT
    )
    val_dataset = _dataset(
        merged,
        "validation",
        mapping,
        build_evaluation_transform(transform_config),
        DATA_ROOT,
    )

    train_rows = [r for r in _read(merged) if r["split"] == "train"]
    weight_map = normalized_inverse_frequency_weights(
        Counter(r["class_name"] for r in train_rows)
    )
    weights = torch.tensor(
        [weight_map[label] for label, _ in sorted(mapping.items(), key=lambda i: i[1])],
        dtype=torch.float32,
    )

    model_config = ModelConfig(
        backbone="mobilenetv3_small_100", pretrained=True, num_classes=17, dropout=0.2
    )
    seed_everything(SEED, True)
    model = build_classifier(model_config, mapping)
    optimizer = build_optimizer(
        model.parameters(), OptimizerConfig(learning_rate=0.0005, weight_decay=0.0001)
    )
    scheduler = build_scheduler(
        optimizer,
        SchedulerConfig(name="cosine", minimum_learning_rate=0.000001),
        epochs=EPOCHS,
    )
    loss_function = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)

    checkpoint_dir = WORK_DIR / "checkpoints"
    manager = CheckpointManager(checkpoint_dir, dataset_root=DATA_ROOT)
    metadata = CheckpointMetadata(
        class_mapping=mapping,
        model_config=asdict(model_config),
        preprocessing_config=asdict(transform_config),
        random_seed=SEED,
        training_config={"epochs": EPOCHS, "pool": "v2_train + hf_supplement(mapped)"},
        project_metadata={"experiment": "v3_hf_supplement"},
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
        checkpoint_manager=manager,
        checkpoint_metadata=metadata,
    )
    print(
        f"best epoch {fit.best_epoch + 1}/{EPOCHS}, "
        f"val_macro_f1={fit.best_validation_macro_f1:.4f}"
    )

    # Locked test images are decoded only now, after selection is complete.
    test_dataset = _dataset(
        MANIFEST_DIR / "v2_test.csv",
        "test",
        mapping,
        build_evaluation_transform(transform_config),
        DATA_ROOT / VQA_SUBDIR,
    )
    result = Evaluator(
        model=model, device=torch.device("cpu"), class_mapping=mapping
    ).evaluate(_loader(test_dataset, shuffle=False, seed=44))
    save_evaluation_outputs(result, RESULTS_DIR, dataset_root=DATA_ROOT / VQA_SUBDIR)
    print(f"results -> {RESULTS_DIR.relative_to(PROJECT_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
