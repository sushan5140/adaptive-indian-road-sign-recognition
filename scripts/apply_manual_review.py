"""Validate a completed six-class review and emit the external-test manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.experiment_preparation import (  # noqa: E402
    ExperimentPreparationError,
    build_review_rows,
)
from data.manual_review import (  # noqa: E402
    ManualReviewError,
    apply_manual_review,
    write_applied_review,
)


def build_parser() -> argparse.ArgumentParser:
    """Build arguments for strict manual-review application."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--review-csv",
        default="outputs/manual_review/dataset_b_six_class_review.csv",
    )
    parser.add_argument(
        "--classification-manifest",
        default="outputs/manifests/dataset_b_classification.csv",
    )
    parser.add_argument(
        "--output-manifest",
        default="outputs/manifests/dataset_b_external_test.csv",
    )
    parser.add_argument(
        "--output-summary",
        default="outputs/manifests/dataset_b_external_test_summary.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Validate all review rows and return a process status code."""
    arguments = build_parser().parse_args(argv)
    try:
        expected = build_review_rows(_project_path(arguments.classification_manifest))
        result = apply_manual_review(_project_path(arguments.review_csv), expected)
        manifest, summary = write_applied_review(
            result,
            manifest_path=_project_path(arguments.output_manifest),
            summary_path=_project_path(arguments.output_summary),
        )
    except (
        ExperimentPreparationError,
        ManualReviewError,
        OSError,
        ValueError,
    ) as error:
        print(f"FATAL: {error}", file=sys.stderr)
        return 2
    print(f"External-test manifest: {manifest}")
    print(f"Review summary: {summary}")
    print(f"Approved/relabelled samples: {len(result.external_rows)}")
    print("This manifest is external test only; do not use it for model selection.")
    return 0


def _project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
