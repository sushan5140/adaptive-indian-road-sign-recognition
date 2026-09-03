"""Train a V2-pool candidate with stronger augmentation and higher weight decay.

Same 287 Dataset B training images, same MobileNetV3-Small, same 30 epochs, seed
42, batch size, learning rate, cosine schedule, class-weighted cross-entropy with
label smoothing, and selection on validation macro F1. Neither the HF supplement
nor Dataset A is used.

Two deliberate changes, both aimed at overfitting:

* stronger augmentation — random resized crop, rotation to 15 degrees, wider
  colour jitter with mild saturation and hue, and random erasing;
* weight decay raised from 0.0001 to 0.0005.

Horizontal and vertical flips stay disabled: mirroring changes sign meaning.
Evaluation preprocessing is unchanged deterministic resize plus normalization, so
validation numbers stay comparable with V2.

Reports validation macro F1 only. The locked test split is never read here.
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
from torchvision import transforms as T  # noqa: E402

from data.road_sign_dataset import RoadSignDataset, RoadSignDatasetConfig  # noqa: E402
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
    IMAGENET_MEAN,
    IMAGENET_STD,
    TransformConfig,
    build_evaluation_transform,
)
from utils.reproducibility import seed_dataloader_worker, seed_everything  # noqa: E402

DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "indian_traffic_vqa"
MANIFEST_DIR = PROJECT_ROOT / "outputs" / "manifests"
WORK_DIR = PROJECT_ROOT / "outputs" / "v4_work"
EPOCHS = 30
SEED = 42
SIZE = 224
WEIGHT_DECAY = 0.0005


def strong_train_transform() -> Any:
    """Aggressive but semantics-preserving training augmentation."""
    return T.Compose(
        [
            T.ToPILImage(),
            T.RandomResizedCrop(SIZE, scale=(0.70, 1.0), ratio=(0.90, 1.11)),
            T.RandomAffine(degrees=15),
            T.ColorJitter(brightness=0.30, contrast=0.30, saturation=0.20, hue=0.02),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            T.RandomErasing(p=0.25, scale=(0.02, 0.15)),
        ]
    )


def _dataset(split: str, mapping: dict[str, int], transform: Any) -> RoadSignDataset:
    manifest = MANIFEST_DIR / (
        "v2_train.csv" if split == "train" else "v2_validation.csv"
    )
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
    """Train the stronger-augmentation candidate and report validation only."""
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    mapping = json.loads((MANIFEST_DIR / "v2_class_mapping.json").read_text())

    eval_config = TransformConfig(
        image_size=SIZE,
        horizontal_flip_probability=0.0,
        max_rotation_degrees=0.0,
        brightness=0.0,
        contrast=0.0,
    )

    seed_everything(SEED, True)
    train_dataset = _dataset("train", mapping, strong_train_transform())
    val_dataset = _dataset(
        "validation", mapping, build_evaluation_transform(eval_config)
    )
    print(
        f"train {len(train_dataset)}  validation {len(val_dataset)}  classes {len(mapping)}"
    )

    with (MANIFEST_DIR / "v2_train.csv").open(encoding="utf-8-sig", newline="") as fh:
        counts = Counter(r["class_name"] for r in csv.DictReader(fh))
    weight_map = normalized_inverse_frequency_weights(counts)
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
        model.parameters(),
        OptimizerConfig(learning_rate=0.0005, weight_decay=WEIGHT_DECAY),
    )
    scheduler = build_scheduler(
        optimizer,
        SchedulerConfig(name="cosine", minimum_learning_rate=0.000001),
        epochs=EPOCHS,
    )
    loss_function = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)

    manager = CheckpointManager(WORK_DIR / "checkpoints", dataset_root=DATASET_ROOT)
    metadata = CheckpointMetadata(
        class_mapping=mapping,
        model_config=asdict(model_config),
        preprocessing_config=asdict(eval_config),
        random_seed=SEED,
        training_config={
            "epochs": EPOCHS,
            "weight_decay": WEIGHT_DECAY,
            "augmentation": "randomresizedcrop+affine15+jitter.3/.3/.2/.02+erasing.25",
            "pool": "v2_train only (287, dataset_b)",
        },
        project_metadata={"experiment": "v4_strong_augmentation"},
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
        f"validation macro_f1={fit.best_validation_macro_f1:.4f}"
    )
    print("TEST SPLIT NOT READ BY THIS SCRIPT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
