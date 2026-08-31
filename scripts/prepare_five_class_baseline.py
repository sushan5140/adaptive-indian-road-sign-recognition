"""Create the relocked five-class training and external-test manifests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.five_class_experiment import (  # noqa: E402
    FiveClassExperimentError,
    build_five_class_external_test,
    build_five_class_training_pool,
    write_five_class_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    """Build command-line options for reproducible artifact preparation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--six-class-pool",
        default="outputs/manifests/dataset_a_six_class_training_pool.csv",
    )
    parser.add_argument(
        "--reviewed-external",
        default="outputs/manifests/dataset_b_external_test.csv",
    )
    parser.add_argument(
        "--training-output",
        default="outputs/manifests/dataset_a_five_class_training_pool.csv",
    )
    parser.add_argument(
        "--mapping-output",
        default="outputs/manifests/dataset_a_five_class_mapping.json",
    )
    parser.add_argument(
        "--external-output",
        default="outputs/manifests/dataset_b_external_test_five_class.csv",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Validate source artifacts and write the relocked five-class artifacts."""
    arguments = build_parser().parse_args(argv)
    try:
        training_rows = build_five_class_training_pool(
            _project_path(arguments.six_class_pool)
        )
        external_rows = build_five_class_external_test(
            _project_path(arguments.reviewed_external)
        )
        training, mapping, external = write_five_class_artifacts(
            training_rows,
            external_rows,
            training_manifest=_project_path(arguments.training_output),
            class_mapping_path=_project_path(arguments.mapping_output),
            external_manifest=_project_path(arguments.external_output),
        )
    except (FiveClassExperimentError, OSError, ValueError) as error:
        print(f"FATAL: {error}", file=sys.stderr)
        return 2
    print(f"Dataset A training pool: {training} ({len(training_rows)} rows)")
    print(f"Model class mapping: {mapping}")
    print(f"Dataset B external test: {external} ({len(external_rows)} rows)")
    print("Dataset B role: external test only")
    return 0


def _project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
