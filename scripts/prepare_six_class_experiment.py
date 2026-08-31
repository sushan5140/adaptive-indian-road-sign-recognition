"""Prepare manual review and Dataset A pool artifacts without training a model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.experiment_preparation import (  # noqa: E402
    SPECS_BY_B_CLASS,
    ExperimentPreparationError,
    build_dataset_a_training_pool,
    build_review_rows,
    create_contact_sheets,
    write_dataset_a_pool,
    write_review_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    """Build command-line arguments for six-class experiment preparation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--classification-manifest",
        default="outputs/manifests/dataset_b_classification.csv",
    )
    parser.add_argument("--dataset-b-root", default="data/raw/indian_traffic_vqa")
    parser.add_argument(
        "--review-manifest",
        default="outputs/manual_review/dataset_b_six_class_review.csv",
    )
    parser.add_argument(
        "--contact-sheet-dir", default="outputs/manual_review/contact_sheets"
    )
    parser.add_argument(
        "--training-pool",
        default="outputs/manifests/dataset_a_six_class_training_pool.csv",
    )
    parser.add_argument(
        "--grouping-report",
        default="outputs/experiment_protocol/dataset_a_six_class_grouping_report.json",
    )
    parser.add_argument(
        "--class-mapping",
        default="outputs/manifests/dataset_a_six_class_mapping.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Generate review and training-pool references without modifying datasets."""
    arguments = build_parser().parse_args(argv)
    try:
        config = _load_yaml(_project_path(arguments.config))
        dataset_a_root = _dataset_a_root(config)
        review_rows = build_review_rows(
            _project_path(arguments.classification_manifest)
        )
        if len(review_rows) != 147:
            raise ExperimentPreparationError(
                f"Expected 147 exact-overlap review rows, found {len(review_rows)}"
            )
        review_path = write_review_manifest(
            review_rows, _project_path(arguments.review_manifest)
        )
        sheets = create_contact_sheets(
            review_rows,
            dataset_root=_project_path(arguments.dataset_b_root),
            output_directory=_project_path(arguments.contact_sheet_dir),
        )
        pool, report = build_dataset_a_training_pool(dataset_a_root)
        pool_path, report_path = write_dataset_a_pool(
            pool,
            report,
            manifest_path=_project_path(arguments.training_pool),
            report_path=_project_path(arguments.grouping_report),
        )
        mapping_path = _write_class_mapping(_project_path(arguments.class_mapping))
    except (ExperimentPreparationError, OSError, ValueError, yaml.YAMLError) as error:
        print(f"FATAL: {error}", file=sys.stderr)
        return 2
    print(f"Review manifest: {review_path}")
    print(f"Review rows: {len(review_rows)}")
    print(f"Contact sheets: {len(sheets)}")
    print(f"Dataset A training pool: {pool_path}")
    print(f"Dataset A training samples: {len(pool)}")
    print(f"Grouping report: {report_path}")
    print(f"Class mapping: {mapping_path}")
    print("Training started: no")
    return 0


def _write_class_mapping(path: Path) -> Path:
    if path.exists():
        raise ExperimentPreparationError(f"Refusing to overwrite class mapping: {path}")
    mapping = {label: index for index, label in enumerate(sorted(SPECS_BY_B_CLASS))}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ExperimentPreparationError("Configuration root must be an object")
    return payload


def _dataset_a_root(config: dict[str, Any]) -> Path:
    dataset = config.get("dataset")
    if not isinstance(dataset, dict):
        raise ExperimentPreparationError("Configuration requires dataset settings")
    value = dataset.get("root")
    if not isinstance(value, str) or not value.strip():
        raise ExperimentPreparationError("dataset.root must be a path")
    return Path(value).expanduser()


def _project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
