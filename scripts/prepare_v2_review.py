"""Prepare Baseline V2 manual-review artifacts without splitting or training."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.experiment_preparation import (  # noqa: E402
    ExperimentPreparationError,
    create_contact_sheets,
)
from data.v2_review import (  # noqa: E402
    V2ReviewError,
    pending_review_rows,
    prepare_v2_review,
    write_v2_review,
)

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build command-line arguments for V2 review preparation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-manifest",
        default="outputs/manifests/dataset_b_classification.csv",
    )
    parser.add_argument(
        "--previous-review",
        default="outputs/manual_review/dataset_b_six_class_review.csv",
    )
    parser.add_argument(
        "--excluded-candidates",
        default="outputs/dataset_b_audit/excluded_candidates.csv",
    )
    parser.add_argument("--dataset-root", default="data/raw/indian_traffic_vqa")
    parser.add_argument(
        "--review-output", default="outputs/v2_review/dataset_b_v2_review.csv"
    )
    parser.add_argument(
        "--summary-output", default="outputs/v2_review/review_summary.json"
    )
    parser.add_argument(
        "--contact-sheet-dir", default="outputs/v2_review/contact_sheets"
    )
    parser.add_argument("--items-per-page", type=int, default=12)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Write a protected V2 review queue and pending-only contact sheets."""
    logging.basicConfig(
        level=logging.INFO,
        format=(
            '{"time":"%(asctime)s","level":"%(levelname)s",'
            '"logger":"%(name)s","message":"%(message)s"}'
        ),
    )
    arguments = build_parser().parse_args(argv)
    try:
        preparation = prepare_v2_review(
            _project_path(arguments.candidate_manifest),
            _project_path(arguments.previous_review),
            _project_path(arguments.excluded_candidates),
        )
        review_path, summary_path = write_v2_review(
            preparation,
            review_path=_project_path(arguments.review_output),
            summary_path=_project_path(arguments.summary_output),
        )
        pending = pending_review_rows(preparation.rows)
        sheets = create_contact_sheets(
            pending,
            dataset_root=_project_path(arguments.dataset_root),
            output_directory=_project_path(arguments.contact_sheet_dir),
            items_per_page=arguments.items_per_page,
        )
    except (OSError, ValueError, V2ReviewError, ExperimentPreparationError) as error:
        LOGGER.error("V2 review preparation failed: %s", error)
        return 2
    LOGGER.info("Review manifest: %s", review_path)
    LOGGER.info("Review summary: %s", summary_path)
    LOGGER.info("Candidates: %d", len(preparation.rows))
    LOGGER.info("Pending: %d", len(pending))
    LOGGER.info("Contact sheets: %d", len(sheets))
    LOGGER.info("Training started: no")
    return 0


def _project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
