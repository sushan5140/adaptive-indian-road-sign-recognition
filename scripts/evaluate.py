"""Evaluate a measured checkpoint on a configured closed-set split."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import (  # noqa: E402
    ConfigurationError,
    apply_overrides,
    load_yaml_config,
    require_mapping,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the evaluation CLI parser without importing ML dependencies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root")
    parser.add_argument("--manifest")
    parser.add_argument("--split", default=None)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--output-dir")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Validate inputs, evaluate a checkpoint, and write measured reports."""
    arguments = build_parser().parse_args(argv)
    try:
        config = load_yaml_config(arguments.config)
        resolved = apply_overrides(
            config,
            {
                "dataset": {
                    "root": arguments.dataset_root,
                    "manifest_path": arguments.manifest,
                },
                "loader": {"batch_size": arguments.batch_size},
                "device": {"type": arguments.device},
                "evaluation": {
                    "split": arguments.split,
                    "output_directory": arguments.output_dir,
                },
            },
        )
        dataset_root = _dataset_root(resolved)
        if not dataset_root.exists() or not dataset_root.is_dir():
            raise ConfigurationError(
                f"Dataset root does not exist or is not a directory: {dataset_root}"
            )
        checkpoint_path = Path(arguments.checkpoint).expanduser().resolve()
        if not checkpoint_path.exists() or not checkpoint_path.is_file():
            raise ConfigurationError(f"Checkpoint does not exist: {checkpoint_path}")
        return _run_evaluation(resolved, dataset_root, checkpoint_path)
    except Exception as error:  # CLI boundary converts validated failures to status.
        print(f"FATAL: {error}", file=sys.stderr)
        return 2


def _run_evaluation(
    config: dict[str, Any], dataset_root: Path, checkpoint_path: Path
) -> int:
    import torch
    from torch.utils.data import DataLoader

    from data.road_sign_dataset import RoadSignDataset
    from evaluation.evaluator import Evaluator, save_evaluation_outputs
    from models.factory import ModelConfig, build_classifier
    from training.checkpoint import load_checkpoint, read_checkpoint_payload
    from training.configuration import (
        dataset_config_from_yaml,
        loader_config_from_yaml,
        transform_config_from_yaml,
    )
    from training.transforms import TransformConfig, build_evaluation_transform
    from utils.device import select_device

    device = select_device(str(require_mapping(config, "device").get("type", "auto")))
    payload = read_checkpoint_payload(checkpoint_path, map_location=device)
    class_mapping = _class_mapping(payload.get("class_mapping"))
    raw_model_config = payload.get("model_config")
    if not isinstance(raw_model_config, dict):
        raise ConfigurationError("Checkpoint model_config is invalid")
    model_config = ModelConfig(**raw_model_config)
    configured_transform = transform_config_from_yaml(config)
    raw_preprocessing = payload.get("preprocessing_config")
    if not isinstance(raw_preprocessing, dict):
        raise ConfigurationError("Checkpoint preprocessing_config is invalid")
    try:
        transform_config = TransformConfig(**raw_preprocessing)
        transform_config.validate()
    except (TypeError, ValueError) as error:
        raise ConfigurationError(
            "Checkpoint preprocessing configuration is incompatible"
        ) from error
    if configured_transform.image_size != transform_config.image_size:
        print(
            "WARNING: using checkpoint image preprocessing instead of the current "
            "configuration",
            file=sys.stderr,
        )
    evaluation_section = require_mapping(config, "evaluation")
    raw_requested_split = evaluation_section.get("split", "test")
    requested_split = None if raw_requested_split is None else str(raw_requested_split)
    dataset_config = replace(
        dataset_config_from_yaml(
            config,
            root=dataset_root,
            split=requested_split,
        ),
        return_metadata=True,
        class_mapping_path=None,
    )
    dataset = RoadSignDataset(
        dataset_config,
        transform=build_evaluation_transform(transform_config),
        class_mapping=class_mapping,
    )
    if dataset.class_to_index != class_mapping:
        raise ConfigurationError("Evaluation dataset class mapping is incompatible")
    seed = int(require_mapping(config, "reproducibility").get("seed", 42))
    loader_config = loader_config_from_yaml(config, seed=seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    from utils.reproducibility import seed_dataloader_worker

    loader = DataLoader(
        dataset,
        batch_size=loader_config.batch_size,
        shuffle=False,
        num_workers=loader_config.num_workers,
        pin_memory=loader_config.pin_memory,
        worker_init_fn=seed_dataloader_worker,
        generator=generator,
    )
    model = build_classifier(model_config, class_mapping)
    load_checkpoint(
        checkpoint_path,
        model=model,
        expected_class_mapping=class_mapping,
        expected_model_config=raw_model_config,
        map_location=device,
    )
    result = Evaluator(
        model=model,
        device=device,
        class_mapping=class_mapping,
    ).evaluate(loader)
    output_base = Path(
        str(evaluation_section.get("output_directory", "outputs/evaluations"))
    ).expanduser()
    run_id = checkpoint_path.parent.name
    output_directory = output_base.resolve() / run_id
    if output_directory.exists():
        raise ConfigurationError(
            f"Evaluation output already exists; refusing overwrite: {output_directory}"
        )
    save_evaluation_outputs(result, output_directory, dataset_root=dataset_root)
    print(f"Measured top-1 accuracy: {result.metrics.top1_accuracy:.6f}")
    print(f"Measured macro F1: {result.metrics.macro_f1:.6f}")
    print(f"Evaluation directory: {output_directory}")
    print("Softmax confidence is closed-set confidence, not an open-set score.")
    return 0


def _class_mapping(value: Any) -> dict[str, int]:
    if not isinstance(value, dict) or any(
        not isinstance(label, str)
        or isinstance(index, bool)
        or not isinstance(index, int)
        for label, index in value.items()
    ):
        raise ConfigurationError("Checkpoint class_mapping is invalid")
    return dict(value)


def _dataset_root(config: dict[str, Any]) -> Path:
    value = require_mapping(config, "dataset").get("root")
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError("dataset.root must be a non-empty path")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
