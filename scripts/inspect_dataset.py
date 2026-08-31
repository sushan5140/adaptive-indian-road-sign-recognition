"""Command-line entry point for read-only dataset inspection."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset_inspector import (  # noqa: E402
    DatasetInspectionError,
    DatasetInspectionReport,
    DatasetInspector,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the inspection command's argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--dataset-root")
    parser.add_argument("--mode")
    parser.add_argument("--manifest")
    parser.add_argument("--output-json")
    parser.add_argument("--output-csv")
    parser.add_argument(
        "--verify-images",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-images", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Inspect a configured dataset and return a process exit status."""
    arguments = build_parser().parse_args(argv)
    try:
        config = _load_config(Path(arguments.config))
        dataset_config = config.get("dataset")
        if not isinstance(dataset_config, dict):
            raise DatasetInspectionError("Configuration requires a dataset object")
        root_value = arguments.dataset_root or dataset_config.get("root")
        if not isinstance(root_value, str) or not root_value.strip():
            raise DatasetInspectionError("dataset.root must be a non-empty path")
        dataset_root = Path(root_value).expanduser()
        if not dataset_root.is_absolute():
            dataset_root = PROJECT_ROOT / dataset_root
        manifest = arguments.manifest or dataset_config.get("manifest_path")
        inspector = DatasetInspector(
            dataset_root,
            mode=arguments.mode or str(dataset_config.get("mode", "auto")),
            manifest_path=manifest,
            image_path_column=str(
                dataset_config.get("image_path_column", "image_path")
            ),
            label_column=str(dataset_config.get("label_column", "label")),
            split_column=_optional_string(dataset_config.get("split_column", "split")),
            allowed_extensions=_string_list(
                dataset_config.get(
                    "allowed_extensions", [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
                )
            ),
            verify_images=arguments.verify_images,
            max_images=arguments.max_images,
        )
        report = inspector.inspect()
        _print_report(report)
        if arguments.output_json:
            _write_json(report, Path(arguments.output_json), inspector.root)
        if arguments.output_csv:
            _write_csv(report, Path(arguments.output_csv), inspector.root)
        return 0
    except (
        DatasetInspectionError,
        OSError,
        TypeError,
        ValueError,
        yaml.YAMLError,
    ) as error:
        print(f"FATAL: {error}", file=sys.stderr)
        return 2


def _load_config(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise DatasetInspectionError(f"Could not read configuration {path}") from error
    if not isinstance(payload, dict):
        raise DatasetInspectionError("Configuration root must be an object")
    return payload


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise DatasetInspectionError("split_column must be a string or null")
    return value


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise DatasetInspectionError("allowed_extensions must be a list of strings")
    return value


def _print_report(report: DatasetInspectionReport) -> None:
    print(f"Dataset root: {report.dataset_root}")
    print(f"Detected mode: {report.detected_mode}")
    print(f"Images: {report.total_image_count}")
    print(f"Classes: {report.number_of_classes}")
    print(f"Class names: {', '.join(report.class_names) or '(unlabelled)'}")
    print(f"Splits: {', '.join(report.possible_splits) or '(none detected)'}")
    print(f"Verified images: {report.verified_image_count}")
    print(f"Unreadable files: {len(report.unreadable_files)}")
    print(f"Unsupported files: {len(report.unsupported_files)}")
    print(f"Duplicate paths: {len(report.duplicate_file_paths)}")
    print(f"Empty class directories: {len(report.empty_class_directories)}")
    print(f"Class imbalance detected: {report.class_imbalance_exists}")
    for warning in report.warnings:
        print(f"WARNING: {warning}", file=sys.stderr)


def _validate_output_path(path: Path, dataset_root: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved == dataset_root or resolved.is_relative_to(dataset_root):
        raise DatasetInspectionError(
            f"Report output must be outside the source dataset: {resolved}"
        )
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _write_json(
    report: DatasetInspectionReport, output_path: Path, dataset_root: Path
) -> None:
    destination = _validate_output_path(output_path, dataset_root)
    destination.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"JSON report: {destination}")


def _write_csv(
    report: DatasetInspectionReport, output_path: Path, dataset_root: Path
) -> None:
    destination = _validate_output_path(output_path, dataset_root)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("metric", "value"))
        for key, value in report.to_dict().items():
            writer.writerow((key, json.dumps(value, ensure_ascii=False)))
    print(f"CSV report: {destination}")


if __name__ == "__main__":
    raise SystemExit(main())
