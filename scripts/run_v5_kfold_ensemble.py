"""Five-fold cross-validation ensemble over the merged Dataset B pool.

Merges v2_train.csv (287) and v2_validation.csv (62) into one 349-image pool,
splits it into five stratified folds seeded at 42, trains one model per fold on
the other four fifths, and averages the five softmax vectors at inference.

Deliberate design choices, both reacting to how V4 went wrong:

* V2's original augmentation only — RandomAffine(7 degrees) and
  ColorJitter(0.2, 0.2). V4's stronger set is not used, because V2's recipe is
  the one with evidence of transferring to the test split.
* A fixed 21 epochs per fold with no validation pass and no checkpoint
  selection. Per-fold validation-based selection is the exact mechanism that
  picked the worse model in V4, and merging validation into the pool leaves no
  held-out data to select on anyway.

Caveat recorded here because it affects how the result must be read: 21 epochs
is inherited from V2's best epoch, which was chosen on a validation set that is
now inside this training pool. That is a mild optimism leak. It is far better
than tuning against the test split, but it is not an independent choice.

The locked test split is read exactly once, by the assembled ensemble.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

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
from training.transforms import (  # noqa: E402
    TransformConfig,
    build_evaluation_transform,
    build_train_transform,
)
from utils.reproducibility import seed_dataloader_worker, seed_everything  # noqa: E402

DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "indian_traffic_vqa"
MANIFEST_DIR = PROJECT_ROOT / "outputs" / "manifests"
WORK_DIR = PROJECT_ROOT / "outputs" / "v5_work"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "v5_results"
FOLDS = 5
EPOCHS = 21
SEED = 42

TRAIN_CONFIG = TransformConfig(
    image_size=224,
    horizontal_flip_probability=0.0,
    max_rotation_degrees=7.0,
    brightness=0.20,
    contrast=0.20,
)
EVAL_CONFIG = TransformConfig(
    image_size=224,
    horizontal_flip_probability=0.0,
    max_rotation_degrees=0.0,
    brightness=0.0,
    contrast=0.0,
)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def build_pool() -> list[dict[str, str]]:
    """Merge the train and validation manifests, Dataset B only."""
    pool: list[dict[str, str]] = []
    for name in ("v2_train.csv", "v2_validation.csv"):
        for row in _read(MANIFEST_DIR / name):
            pool.append(
                {
                    "image_path": row["image_path"],
                    "class_name": row["class_name"],
                    "origin": name,
                }
            )
    origins = Counter(r["origin"] for r in pool)
    print(f"pool: {len(pool)} images  {dict(origins)}")

    # Sanity: nothing may come from the HF supplement or Dataset A.
    supplement = {r["image_path"] for r in _read(MANIFEST_DIR / "hf_supplement.csv")}
    leaked = [r for r in pool if r["image_path"] in supplement]
    foreign = [
        r
        for r in pool
        if "hf_indian_traffic_sign" in r["image_path"]
        or "indian_traffic_sign_dataset" in r["image_path"]
    ]
    print(f"  rows from hf_supplement.csv : {len(leaked)}")
    print(f"  rows from Dataset A or HF dirs: {len(foreign)}")
    if leaked or foreign:
        raise SystemExit("pool contamination detected")
    if len(pool) != 349:
        raise SystemExit(f"expected 349 pool images, got {len(pool)}")
    return pool


def write_fold_manifest(rows: list[dict[str, str]], path: Path) -> None:
    """Write a single-split manifest for one fold's training portion."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["image_path", "class_name", "split"]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "image_path": row["image_path"],
                    "class_name": row["class_name"],
                    "split": "train",
                }
            )


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


def train_fold(fold: int, rows: list[dict[str, str]], mapping: dict[str, int]) -> Path:
    """Train one fold for a fixed epoch count with no validation pass."""
    fold_dir = WORK_DIR / f"fold_{fold}"
    manifest = fold_dir / "train.csv"
    write_fold_manifest(rows, manifest)

    seed_everything(SEED + fold, True)
    dataset = _dataset(
        manifest, "train", mapping, build_train_transform(TRAIN_CONFIG), DATASET_ROOT
    )
    loader: DataLoader[Any] = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        worker_init_fn=seed_dataloader_worker,
        generator=torch.Generator().manual_seed(SEED + fold),
    )

    weight_map = normalized_inverse_frequency_weights(
        Counter(r["class_name"] for r in rows)
    )
    weights = torch.tensor(
        [weight_map[label] for label, _ in sorted(mapping.items(), key=lambda i: i[1])],
        dtype=torch.float32,
    )

    model_config = ModelConfig(
        backbone="mobilenetv3_small_100", pretrained=True, num_classes=17, dropout=0.2
    )
    seed_everything(SEED + fold, True)
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

    device = torch.device("cpu")
    model.to(device).train()
    for epoch in range(EPOCHS):
        total, correct, running = 0, 0, 0.0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()
            running += float(loss) * labels.size(0)
            correct += int((logits.argmax(1) == labels).sum())
            total += labels.size(0)
        scheduler.step()
        print(
            f"  fold={fold} epoch={epoch} loss={running / total:.4f} "
            f"train_accuracy={correct / total:.4f}",
            flush=True,
        )

    manager = CheckpointManager(fold_dir, dataset_root=DATASET_ROOT)
    metadata = CheckpointMetadata(
        class_mapping=mapping,
        model_config=asdict(model_config),
        preprocessing_config=asdict(EVAL_CONFIG),
        random_seed=SEED + fold,
        training_config={
            "epochs": EPOCHS,
            "fold": fold,
            "folds": FOLDS,
            "train_images": len(rows),
            "selection": "none; fixed epoch count, no validation pass",
            "augmentation": "v2 original: affine7 + jitter .2/.2",
        },
        project_metadata={"experiment": "v5_kfold_ensemble"},
    )
    path = manager.save(
        "last.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=EPOCHS - 1,
        best_validation_metric=None,
        metadata=metadata,
    )
    print(f"  fold={fold} checkpoint -> {path.relative_to(PROJECT_ROOT).as_posix()}")
    return path


def main() -> int:
    """Train the five folds and evaluate the ensemble once on the locked test."""
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    mapping = json.loads((MANIFEST_DIR / "v2_class_mapping.json").read_text())

    pool = build_pool()
    labels = [r["class_name"] for r in pool]
    splitter = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    folds = list(splitter.split(np.zeros(len(pool)), labels))

    print(f"\nstratified {FOLDS}-fold, seed {SEED} — held-out counts per class")
    header = (
        "  " + "class".ljust(30) + "total" + "".join(f"  f{i}" for i in range(FOLDS))
    )
    print(header)
    held: dict[str, list[int]] = defaultdict(lambda: [0] * FOLDS)
    for index, (_, test_index) in enumerate(folds):
        for i in test_index:
            held[pool[i]["class_name"]][index] += 1
    totals = Counter(labels)
    for name in sorted(totals):
        print(f"  {name:<30}{totals[name]:>5}" + "".join(f"{v:>4}" for v in held[name]))

    checkpoints = []
    for index, (train_index, _) in enumerate(folds):
        rows = [pool[i] for i in train_index]
        print(f"\n=== fold {index}: {len(rows)} training images ===", flush=True)
        checkpoints.append(train_fold(index, rows, mapping))

    # ---- single locked-test evaluation with the assembled ensemble ----
    print("\n=== ensemble evaluation on the locked test split ===", flush=True)
    index_to_label = [k for k, _ in sorted(mapping.items(), key=lambda i: i[1])]
    dataset = _dataset(
        MANIFEST_DIR / "v2_test.csv",
        "test",
        mapping,
        build_evaluation_transform(EVAL_CONFIG),
        DATASET_ROOT,
    )
    loader: DataLoader[Any] = DataLoader(
        dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=False
    )
    truth = [int(label) for _, label in dataset]  # type: ignore[misc]

    per_fold_probabilities = []
    for index, path in enumerate(checkpoints):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        model = build_classifier(
            ModelConfig(
                backbone="mobilenetv3_small_100",
                pretrained=False,
                num_classes=17,
                dropout=0.2,
            ),
            mapping,
        )
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.eval()
        chunks = []
        with torch.no_grad():
            for images, _ in loader:
                chunks.append(torch.softmax(model(images), dim=1))
        probabilities = torch.cat(chunks).numpy()
        per_fold_probabilities.append(probabilities)
        predicted = probabilities.argmax(1)
        print(
            f"  fold {index} standalone: accuracy="
            f"{accuracy_score(truth, predicted):.4f} "
            f"macro_f1={f1_score(truth, predicted, average='macro', zero_division=0):.4f}"
        )

    ensemble = np.mean(per_fold_probabilities, axis=0)
    predicted = ensemble.argmax(1)
    accuracy = accuracy_score(truth, predicted)
    macro = f1_score(truth, predicted, average="macro", zero_division=0)
    per_class = f1_score(
        truth, predicted, average=None, labels=list(range(17)), zero_division=0
    )
    support = Counter(truth)

    print(f"\nENSEMBLE accuracy={accuracy:.4f} macro_f1={macro:.4f}")
    report = {
        "folds": FOLDS,
        "epochs_per_fold": EPOCHS,
        "pool_images": len(pool),
        "top1_accuracy": float(accuracy),
        "macro_f1": float(macro),
        "per_fold_standalone": [
            {
                "fold": i,
                "top1_accuracy": float(accuracy_score(truth, p.argmax(1))),
                "macro_f1": float(
                    f1_score(truth, p.argmax(1), average="macro", zero_division=0)
                ),
            }
            for i, p in enumerate(per_fold_probabilities)
        ],
        "per_class": [
            {
                "label": index_to_label[i],
                "f1": float(per_class[i]),
                "support": int(support.get(i, 0)),
            }
            for i in range(17)
        ],
    }
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")

    with (RESULTS_DIR / "predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as fh:
        writer = csv.writer(fh)
        writer.writerow(["index", "true_label", "predicted_label", "confidence"])
        for i, (t, p) in enumerate(zip(truth, predicted, strict=True)):
            writer.writerow(
                [i, index_to_label[t], index_to_label[p], f"{ensemble[i][p]:.6f}"]
            )
    print(f"results -> {RESULTS_DIR.relative_to(PROJECT_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
