"""One-time locked-test evaluation of the selected V4 strong-augmentation model.

V4 plain was selected on validation macro F1 (0.6493) over V2 plain (0.6223),
V2+TTA (0.6254) and V4+TTA (0.6454). This script reads outputs/manifests/v2_test.csv
once and writes the measured result to outputs/v4_results/.

Deterministic evaluation preprocessing only: resize to 224 and ImageNet
normalization, identical to the V2 evaluation path. No test-time augmentation,
because TTA lost on validation for this checkpoint.
"""

from __future__ import annotations

import json
import sys
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
from training.checkpoint import read_checkpoint_payload  # noqa: E402
from training.transforms import (  # noqa: E402
    TransformConfig,
    build_evaluation_transform,
)
from utils.reproducibility import seed_dataloader_worker  # noqa: E402

DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "indian_traffic_vqa"
MANIFEST_DIR = PROJECT_ROOT / "outputs" / "manifests"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "v4_results"
CHECKPOINT = PROJECT_ROOT / "outputs" / "v4_work" / "checkpoints" / "best.pt"


def main() -> int:
    """Evaluate the selected checkpoint once on the locked test split."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = read_checkpoint_payload(CHECKPOINT)
    mapping = {str(k): int(v) for k, v in payload["class_mapping"].items()}
    print(f"checkpoint epoch {payload['epoch']} | classes {len(mapping)}")
    print(f"training config: {json.dumps(payload['training_config'])}")

    model = build_classifier(
        ModelConfig(
            backbone="mobilenetv3_small_100",
            pretrained=False,
            num_classes=len(mapping),
            dropout=0.2,
        ),
        mapping,
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()

    transform = build_evaluation_transform(
        TransformConfig(
            image_size=224,
            horizontal_flip_probability=0.0,
            max_rotation_degrees=0.0,
            brightness=0.0,
            contrast=0.0,
        )
    )
    dataset = RoadSignDataset(
        RoadSignDatasetConfig(
            root=DATASET_ROOT,
            mode="csv_manifest",
            manifest_path=MANIFEST_DIR / "v2_test.csv",
            image_path_column="image_path",
            label_column="class_name",
            split_column="split",
            split="test",
        ),
        transform=transform,
        class_mapping=mapping,
    )
    loader: DataLoader[Any] = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        worker_init_fn=seed_dataloader_worker,
        generator=torch.Generator().manual_seed(44),
    )
    print(f"locked test images: {len(dataset)}")

    result = Evaluator(
        model=model, device=torch.device("cpu"), class_mapping=mapping
    ).evaluate(loader)
    save_evaluation_outputs(result, RESULTS_DIR, dataset_root=DATASET_ROOT)
    print(f"results -> {RESULTS_DIR.relative_to(PROJECT_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
