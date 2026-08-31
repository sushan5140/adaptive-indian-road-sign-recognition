"""Audit the local Indian Traffic VQA dataset and write measured artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.vqa_audit import (  # noqa: E402
    VqaAuditConfig,
    VqaAuditError,
    VqaAuditResult,
    VqaDatasetAuditor,
)


def build_parser() -> argparse.ArgumentParser:
    """Build command-line arguments for the Dataset B audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="data/raw/indian_traffic_vqa")
    parser.add_argument("--output-dir", default="outputs/dataset_b_audit")
    parser.add_argument("--manifest-dir", default="outputs/manifests")
    parser.add_argument("--download-dir", default="data/downloads")
    parser.add_argument("--minimum-class-images", type=int, default=10)
    parser.add_argument("--minimum-independent-groups", type=int, default=5)
    parser.add_argument("--near-duplicate-distance", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the read-only audit and return a process status code."""
    arguments = build_parser().parse_args(argv)
    try:
        root = _project_path(arguments.dataset_root)
        output = _project_path(arguments.output_dir)
        manifests = _project_path(arguments.manifest_dir)
        downloads = _project_path(arguments.download_dir)
        if output == root or output.is_relative_to(root):
            raise VqaAuditError("Audit output must be outside the source dataset")
        result = VqaDatasetAuditor(
            VqaAuditConfig(
                dataset_root=root,
                images_directory=root / "traffic512final",
                compact_csv=root / "traffic_vqa_1085.csv",
                full_csv=root / "traffic_vqa_4341.csv",
                minimum_class_images=arguments.minimum_class_images,
                minimum_independent_groups=arguments.minimum_independent_groups,
                near_duplicate_hamming_distance=arguments.near_duplicate_distance,
            )
        ).audit()
        _write_outputs(result, output, downloads, manifests)
    except (OSError, VqaAuditError, ValueError) as error:
        print(f"FATAL: {error}", file=sys.stderr)
        return 2
    print(f"Dataset B audit: {output}")
    print(f"Selected images: {len(result.candidate_manifest)}")
    print(f"Selected classes: {len(result.class_mapping)}")
    print("Training started: no")
    return 0


def _write_outputs(
    result: VqaAuditResult, output: Path, downloads: Path, manifests: Path
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    manifests.mkdir(parents=True, exist_ok=True)
    _write_json(output / "dataset_report.json", result.report)
    _write_json(output / "dataset_quality.json", result.quality)
    _write_json(output / "class_mapping.json", result.class_mapping)
    _write_json(output / "source_files.json", _source_files(downloads))
    _write_csv(output / "image_inventory.csv", result.image_inventory)
    _write_csv(output / "question_frequency.csv", result.question_frequency)
    _write_csv(output / "answer_frequency.csv", result.answer_frequency)
    _write_csv(
        output / "question_answer_frequency.csv", result.question_answer_frequency
    )
    _write_csv(output / "exact_duplicate_groups.csv", result.exact_duplicate_groups)
    _write_csv(output / "near_duplicate_pairs.csv", result.near_duplicate_pairs)
    _write_csv(
        output / "candidate_classification_manifest.csv", result.candidate_manifest
    )
    _write_csv(output / "excluded_candidates.csv", result.excluded_candidates)
    _write_csv(output / "candidate_class_distribution.csv", result.class_distribution)
    _write_csv(
        output / "cross_dataset_semantic_alignment.csv", result.semantic_alignment
    )
    classification_rows = tuple(
        {
            "image_path": row["relative_image_path"],
            "class_name": row["label"],
            "source_answer": row["source_answer"],
            "source_question": row["source_question"],
            "source_dataset": "Indian Traffic VQA Dataset (Zenodo 17300841)",
            "source_image_id": row["image_id"],
            "leakage_group_id": row["leakage_group_id"],
        }
        for row in result.candidate_manifest
    )
    _write_csv(manifests / "dataset_b_classification.csv", classification_rows)
    _write_json(manifests / "dataset_b_class_mapping.json", result.class_mapping)
    _write_csv(
        manifests / "cross_dataset_class_alignment.csv", result.semantic_alignment
    )


def _source_files(downloads: Path) -> list[dict[str, Any]]:
    published_md5 = {
        "traffic512final.zip": "88b53f415736d90c2b9d3c935924e397",
        "traffic_vqa_1085.csv": "c03eac0d01758822fe65baf69773ff46",
        "traffic_vqa_4341.csv": "028296415bc8a19de148ef05fb4bef05",
    }
    rows = []
    for name, expected_md5 in published_md5.items():
        path = downloads / name
        if not path.is_file():
            raise VqaAuditError(f"Missing downloaded source file: {path}")
        actual_md5 = _digest(path, "md5")
        rows.append(
            {
                "name": name,
                "bytes": path.stat().st_size,
                "source": f"https://zenodo.org/records/17300841/files/{name}",
                "published_md5": expected_md5,
                "actual_md5": actual_md5,
                "md5_matches": actual_md5 == expected_md5,
                "sha256": _digest(path, "sha256"),
            }
        )
    return rows


def _digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _write_csv(path: Path, rows: tuple[dict[str, Any], ...]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, ensure_ascii=False)
                        if isinstance(value, (list, dict))
                        else value
                    )
                    for key, value in row.items()
                }
            )


def _project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
