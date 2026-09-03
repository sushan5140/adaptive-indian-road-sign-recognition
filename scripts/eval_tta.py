"""Deterministic test-time augmentation evaluation for a trained checkpoint.

Averages softmax probabilities over a fixed set of semantics-preserving views.
No horizontal or vertical flip: mirroring changes the meaning of a traffic sign.
Views are deterministic, so a run is reproducible.

Defaults to the validation manifest. The locked test split must only ever be
passed once, after a configuration has been selected on validation.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402
from torchvision import transforms as T  # noqa: E402

from models.factory import ModelConfig, build_classifier  # noqa: E402
from training.checkpoint import read_checkpoint_payload  # noqa: E402
from training.transforms import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from utils.image_validation import decode_image  # noqa: E402

DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "indian_traffic_vqa"
MANIFEST_DIR = PROJECT_ROOT / "outputs" / "manifests"
SIZE = 224


def _norm() -> Any:
    return T.Compose([T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def build_views(use_tta: bool) -> list[Any]:
    """Return deterministic view transforms; the first is the plain baseline."""
    plain = T.Compose([T.Resize((SIZE, SIZE)), _norm()])
    if not use_tta:
        return [plain]
    return [
        plain,
        # small rotations, matching the scale of training-time affine jitter
        T.Compose([T.Resize((SIZE, SIZE)), T.RandomRotation((-7, -7)), _norm()]),
        T.Compose([T.Resize((SIZE, SIZE)), T.RandomRotation((7, 7)), _norm()]),
        # mild centre zoom: crop 90% then resize back
        T.Compose(
            [
                T.Resize((SIZE, SIZE)),
                T.CenterCrop(int(SIZE * 0.9)),
                T.Resize((SIZE, SIZE)),
                _norm(),
            ]
        ),
        # fixed brightness/contrast lift, no randomness
        T.Compose(
            [
                T.Resize((SIZE, SIZE)),
                T.ColorJitter(brightness=(1.15, 1.15), contrast=(1.15, 1.15)),
                _norm(),
            ]
        ),
    ]


def main() -> int:
    """Evaluate one checkpoint on one manifest, optionally with TTA."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path, default=MANIFEST_DIR / "v2_validation.csv"
    )
    parser.add_argument("--split", default="validation")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    payload = read_checkpoint_payload(args.checkpoint)
    mapping = {str(k): int(v) for k, v in payload["class_mapping"].items()}
    index_to_label = [k for k, _ in sorted(mapping.items(), key=lambda i: i[1])]

    model = build_classifier(
        ModelConfig(
            backbone=str(
                payload["model_config"].get("backbone", "mobilenetv3_small_100")
            ),
            pretrained=False,
            num_classes=len(mapping),
            dropout=float(payload["model_config"].get("dropout", 0.2)),
        ),
        mapping,
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()

    with args.manifest.open(encoding="utf-8-sig", newline="") as handle:
        rows = [
            r
            for r in csv.DictReader(handle)
            if r.get("split", args.split) == args.split
        ]

    views = build_views(args.tta)
    truth, predicted = [], []
    with torch.no_grad():
        for row in rows:
            array = decode_image(DATASET_ROOT / row["image_path"], convert_to_rgb=True)
            image = Image.fromarray(array)
            batch = torch.stack([view(image) for view in views])
            probabilities = torch.softmax(model(batch), dim=1).mean(dim=0)
            predicted.append(index_to_label[int(torch.argmax(probabilities))])
            truth.append(row["class_name"])

    accuracy = accuracy_score(truth, predicted)
    macro = f1_score(truth, predicted, average="macro", zero_division=0)
    name = args.label or args.checkpoint.parent.name
    print(
        f"{name} | {args.split} n={len(rows)} | views={len(views)} "
        f"| accuracy={accuracy:.4f} macro_f1={macro:.4f}"
    )
    print(
        json.dumps(
            {
                "config": name,
                "split": args.split,
                "n": len(rows),
                "views": len(views),
                "accuracy": accuracy,
                "macro_f1": macro,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
