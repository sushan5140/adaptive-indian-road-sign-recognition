"""Train the supervised MobileNetV3-Small base classifier."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
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
    """Build the training CLI parser without importing ML dependencies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--dataset-root")
    parser.add_argument("--manifest")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--output-dir")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Validate configuration, train, checkpoint, and save measured history."""
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
                "training": {"epochs": arguments.epochs},
                "loader": {"batch_size": arguments.batch_size},
                "device": {"type": arguments.device},
                "reproducibility": {"seed": arguments.seed},
            },
        )
        if arguments.output_dir is not None:
            output_root = Path(arguments.output_dir).expanduser().resolve()
            resolved = apply_overrides(
                resolved,
                {
                    "checkpoint": {"directory": str(output_root / "checkpoints")},
                    "logging": {"runs_directory": str(output_root / "runs")},
                },
            )
        dataset_root = _dataset_root(resolved)
        if not dataset_root.exists() or not dataset_root.is_dir():
            raise ConfigurationError(
                f"Dataset root does not exist or is not a directory: {dataset_root}"
            )
        return _run_training(resolved, dataset_root, arguments.resume)
    except Exception as error:  # CLI boundary converts validated failures to status.
        print(f"FATAL: {error}", file=sys.stderr)
        return 2


def _run_training(
    config: dict[str, Any], dataset_root: Path, resume_path: str | None
) -> int:
    from models.factory import build_classifier
    from training.checkpoint import (
        CheckpointManager,
        CheckpointMetadata,
        load_checkpoint,
    )
    from training.configuration import (
        dataset_config_from_yaml,
        loader_config_from_yaml,
        loss_config_from_yaml,
        model_config_from_yaml,
        optimizer_config_from_yaml,
        scheduler_config_from_yaml,
        transform_config_from_yaml,
    )
    from training.dataloaders import build_dataloaders
    from training.factories import build_loss, build_optimizer, build_scheduler
    from training.run import (
        collect_environment_details,
        create_run_directory,
        create_run_id,
        save_run_metadata,
    )
    from training.trainer import Trainer
    from training.transforms import build_evaluation_transform, build_train_transform
    from utils.device import describe_device, select_device
    from utils.reproducibility import seed_everything

    reproducibility = require_mapping(config, "reproducibility")
    seed = int(reproducibility.get("seed", 42))
    deterministic = bool(reproducibility.get("deterministic", True))
    seed_everything(seed, deterministic)
    device = select_device(str(require_mapping(config, "device").get("type", "auto")))
    transform_config = transform_config_from_yaml(config)
    train_transform = build_train_transform(transform_config)
    evaluation_transform = build_evaluation_transform(transform_config)
    training_section = require_mapping(config, "training")
    selection_protocol = str(
        training_section.get("model_selection", "validation_accuracy")
    )
    if selection_protocol not in {
        "validation_accuracy",
        "fixed_epochs_no_validation",
    }:
        raise ConfigurationError(
            "training.model_selection must be validation_accuracy or "
            "fixed_epochs_no_validation"
        )
    require_validation = selection_protocol == "validation_accuracy"
    dataset_config = dataset_config_from_yaml(config, root=dataset_root, split="train")
    loaders = build_dataloaders(
        dataset_config,
        loader_config_from_yaml(config, seed=seed),
        train_transform=train_transform,
        evaluation_transform=evaluation_transform,
        include_test=require_validation,
        require_validation=require_validation,
    )
    model_config = model_config_from_yaml(config)
    model = build_classifier(model_config, loaders.class_mapping)
    epochs = int(training_section.get("epochs", 30))
    loss_function = build_loss(loss_config_from_yaml(config))
    optimizer = build_optimizer(model.parameters(), optimizer_config_from_yaml(config))
    scheduler = build_scheduler(
        optimizer,
        scheduler_config_from_yaml(config),
        epochs=epochs,
    )

    run_id = create_run_id(model_config.backbone)
    run_directory = create_run_directory(
        str(require_mapping(config, "logging").get("runs_directory", "outputs/runs")),
        run_id,
    )
    checkpoint_base = Path(
        str(
            require_mapping(config, "checkpoint").get(
                "directory", "outputs/checkpoints"
            )
        )
    )
    checkpoint_section = require_mapping(config, "checkpoint")
    monitor = str(checkpoint_section.get("monitor", "val_accuracy"))
    checkpoint_mode = str(checkpoint_section.get("mode", "max"))
    if require_validation:
        if monitor != "val_accuracy" or checkpoint_mode != "max":
            raise ConfigurationError(
                "Validation selection requires checkpoint monitor=val_accuracy and mode=max"
            )
    elif monitor != "none" or checkpoint_mode != "fixed_epoch":
        raise ConfigurationError(
            "Fixed-epoch training requires checkpoint monitor=none and mode=fixed_epoch"
        )
    checkpoint_manager = CheckpointManager(
        checkpoint_base / run_id,
        dataset_root=dataset_root,
    )
    device_report = describe_device(device)
    environment = collect_environment_details(
        selected_device=device_report.selected_device,
        dataset_sizes=loaders.dataset_sizes,
        class_count=len(loaders.class_mapping),
    )
    environment["cuda_available"] = device_report.cuda_available
    environment["cuda_device_count"] = device_report.cuda_device_count
    environment["cuda_device_name"] = device_report.cuda_device_name
    save_run_metadata(
        run_directory,
        resolved_config=config,
        environment=environment,
    )
    metadata = CheckpointMetadata(
        class_mapping=loaders.class_mapping,
        model_config=asdict(model_config),
        preprocessing_config=asdict(transform_config),
        random_seed=seed,
        training_config=dict(training_section),
        project_metadata=dict(config.get("project", {})),
    )
    start_epoch = 0
    best_metric: float | None = float("-inf") if require_validation else None
    if resume_path is not None:
        resume_state = load_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_class_mapping=loaders.class_mapping,
            expected_model_config=asdict(model_config),
            map_location=device,
        )
        start_epoch = resume_state.next_epoch
        best_metric = resume_state.best_validation_metric
    gradient_clip = training_section.get("gradient_clip_norm")
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_function=loss_function,
        device=device,
        scheduler=scheduler,
        gradient_clip_norm=(
            float(gradient_clip) if gradient_clip is not None else None
        ),
        progress_callback=print,
    )
    result = trainer.fit(
        loaders.train,
        loaders.validation,
        epochs=epochs,
        start_epoch=start_epoch,
        best_validation_accuracy=best_metric,
        checkpoint_manager=checkpoint_manager,
        checkpoint_metadata=metadata,
    )
    result.history.save(run_directory)
    print(f"Run ID: {run_id}")
    if result.best_validation_accuracy is None:
        print("Model selection: fixed epochs without validation; saved last.pt only")
    else:
        print(
            "Best measured validation accuracy: "
            f"{result.best_validation_accuracy:.6f}"
        )
    print(f"Run directory: {run_directory}")
    return 0


def _dataset_root(config: dict[str, Any]) -> Path:
    value = require_mapping(config, "dataset").get("root")
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError("dataset.root must be a non-empty path")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
