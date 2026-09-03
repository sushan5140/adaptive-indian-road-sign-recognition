"""Measure open_set thresholds on held-out data instead of leaving them as
placeholders.

``configs/config.yaml`` ships ``base_confidence_threshold: 0.70`` and
``prototype_similarity_threshold: 0.75`` with ``calibrated: false`` — those are
guesses, not measurements. This script replaces the base-classifier threshold
with a value measured on the VALIDATION split (never the locked test split),
using a standard selective-classification calibration: scan candidate
thresholds and pick the lowest one whose accepted predictions meet a target
accuracy (default 90%). That trades acceptance rate ("coverage") for
reliability, and the full curve is written out so a different point on the
trade-off can be chosen later without re-running anything.

Prototype-similarity calibration needs real negative examples — images that
are NOT the registered sign — which the base model's own validation split
cannot provide. If ``outputs/loco_results/*_predictions.csv`` files exist
(produced by scripts/leave_one_class_out_eval.py), this script pools their
per-image similarity scores and calibrates the prototype threshold from that
proxy evidence via the ROC-optimal (Youden's J) operating point. Without any
LOCO results, the prototype threshold is left uncalibrated and the script says
so explicitly rather than guessing.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.pipeline import OpenSetRecognizer  # noqa: E402

MANIFEST_DIR = PROJECT_ROOT / "outputs" / "manifests"
DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "indian_traffic_vqa"
LOCO_RESULTS_DIR = PROJECT_ROOT / "outputs" / "loco_results"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "open_set"
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def _load_validation_rows() -> list[dict[str, str]]:
    path = MANIFEST_DIR / "v2_validation.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _base_confidence_curve(
    recognizer: OpenSetRecognizer, rows: list[dict[str, str]]
) -> list[dict[str, float]]:
    paths = [DATASET_ROOT / row["image_path"] for row in rows]
    true_labels = [row["class_name"] for row in rows]
    decisions = recognizer.predict_paths(paths)

    confidences_and_correctness = [
        (decision.base.confidence, decision.base.label == true_label)
        for decision, true_label in zip(decisions, true_labels, strict=True)
    ]
    thresholds = sorted({round(c, 4) for c, _ in confidences_and_correctness})
    curve = []
    for threshold in thresholds:
        accepted = [
            correct
            for conf, correct in confidences_and_correctness
            if conf >= threshold
        ]
        if not accepted:
            continue
        coverage = len(accepted) / len(confidences_and_correctness)
        accuracy_among_accepted = sum(accepted) / len(accepted)
        curve.append(
            {
                "threshold": threshold,
                "coverage": coverage,
                "accuracy_among_accepted": accuracy_among_accepted,
                "n_accepted": len(accepted),
            }
        )
    return curve


def _pick_threshold(
    curve: list[dict[str, float]], target_accuracy: float
) -> dict[str, float] | None:
    qualifying = [
        point for point in curve if point["accuracy_among_accepted"] >= target_accuracy
    ]
    if not qualifying:
        return None
    # Lowest threshold meeting the bar maximizes coverage among qualifying points.
    return min(qualifying, key=lambda point: point["threshold"])


def _pool_loco_predictions() -> list[dict[str, Any]]:
    if not LOCO_RESULTS_DIR.exists():
        return []
    pooled: list[dict[str, Any]] = []
    for path in sorted(LOCO_RESULTS_DIR.glob("*_predictions.csv")):
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                pooled.append(
                    {
                        "is_novel": row["is_novel"] == "True",
                        "prototype_similarity": float(row["prototype_similarity"]),
                        "source": path.name,
                    }
                )
    return pooled


def _calibrate_prototype_threshold(
    pooled: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not pooled:
        return None
    similarities = sorted({round(r["prototype_similarity"], 4) for r in pooled})
    best = None
    for threshold in similarities:
        true_positive = sum(
            1
            for r in pooled
            if r["is_novel"] and r["prototype_similarity"] >= threshold
        )
        false_negative = sum(
            1 for r in pooled if r["is_novel"] and r["prototype_similarity"] < threshold
        )
        false_positive = sum(
            1
            for r in pooled
            if not r["is_novel"] and r["prototype_similarity"] >= threshold
        )
        true_negative = sum(
            1
            for r in pooled
            if not r["is_novel"] and r["prototype_similarity"] < threshold
        )
        n_pos = true_positive + false_negative
        n_neg = false_positive + true_negative
        if n_pos == 0 or n_neg == 0:
            continue
        tpr = true_positive / n_pos
        fpr = false_positive / n_neg
        youden_j = tpr - fpr
        candidate = {
            "threshold": threshold,
            "true_positive_rate": tpr,
            "false_positive_rate": fpr,
            "youden_j": youden_j,
            "n_novel": n_pos,
            "n_known": n_neg,
        }
        if best is None or youden_j > best["youden_j"]:
            best = candidate
    return best


def _patch_open_set_config(config_path: Path, updates: dict[str, str]) -> None:
    """Rewrite only the given ``key: value`` lines inside the ``open_set:``
    block, leaving every comment, blank line, and other section untouched.

    A full ``yaml.safe_load``/``yaml.safe_dump`` round trip is not used here
    because it silently drops every comment in the file, including the notes
    explaining why ``strategy: conservative`` is the default — exactly the
    kind of documented rationale this project relies on. This patches text
    in place instead, then re-parses the result to confirm it is still valid
    YAML with the expected values before writing it back.
    """
    lines = config_path.read_text().splitlines(keepends=True)
    top_level_pattern = re.compile(r"^\S")
    open_set_pattern = re.compile(r"^open_set:\s*$")
    key_pattern = re.compile(r"^(\s*)([A-Za-z0-9_]+):(\s+).*$")

    start = next(
        (i for i, line in enumerate(lines) if open_set_pattern.match(line)), None
    )
    if start is None:
        raise ValueError(f"No top-level 'open_set:' section found in {config_path}")
    end = next(
        (i for i in range(start + 1, len(lines)) if top_level_pattern.match(lines[i])),
        len(lines),
    )

    remaining = dict(updates)
    for i in range(start + 1, end):
        match = key_pattern.match(lines[i])
        if match and match.group(2) in remaining:
            indent, key, gap = match.groups()
            lines[i] = f"{indent}{key}:{gap}{remaining.pop(key)}\n"
    if remaining:
        raise ValueError(
            f"Could not find existing open_set key(s) to update: {sorted(remaining)}"
        )

    patched_text = "".join(lines)
    reparsed = yaml.safe_load(patched_text)
    for key, raw_value in updates.items():
        expected = yaml.safe_load(raw_value)
        if reparsed["open_set"][key] != expected:
            raise ValueError(
                f"Patched config failed verification for '{key}': "
                f"expected {expected!r}, got {reparsed['open_set'][key]!r}"
            )
    config_path.write_text(patched_text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a trained checkpoint (e.g. outputs/checkpoints/<run_id>/best.pt)",
    )
    parser.add_argument(
        "--target-accuracy",
        type=float,
        default=0.90,
        help="Minimum accuracy required among accepted validation predictions",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the measured thresholds back into configs/config.yaml",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    recognizer = OpenSetRecognizer.from_checkpoint(args.checkpoint, device="cpu")

    validation_rows = _load_validation_rows()
    curve = _base_confidence_curve(recognizer, validation_rows)
    curve_path = OUTPUT_DIR / "base_confidence_curve.csv"
    with curve_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "threshold",
                "coverage",
                "accuracy_among_accepted",
                "n_accepted",
            ],
        )
        writer.writeheader()
        writer.writerows(curve)

    chosen = _pick_threshold(curve, args.target_accuracy)
    if chosen is None:
        print(
            f"No threshold on the validation split reaches {args.target_accuracy:.0%} "
            "accuracy among accepted predictions. The base classifier itself needs "
            "improvement before a threshold can deliver that bar; see "
            f"{curve_path} for the full achievable trade-off."
        )
        base_result = None
    else:
        print(
            f"base_confidence_threshold = {chosen['threshold']:.4f} "
            f"(coverage={chosen['coverage']:.2%}, "
            f"accuracy_among_accepted={chosen['accuracy_among_accepted']:.2%}, "
            f"measured on {len(validation_rows)} validation images)"
        )
        base_result = chosen

    pooled = _pool_loco_predictions()
    prototype_result = _calibrate_prototype_threshold(pooled)
    if prototype_result is None:
        print(
            "No leave-one-class-out results found under "
            f"{LOCO_RESULTS_DIR} (or too few for a two-class ROC). "
            "prototype_similarity_threshold stays uncalibrated. Run "
            "scripts/leave_one_class_out_eval.py on a few classes first, or "
            "collect real new-sign photos and register+evaluate them directly, "
            "to get a trustworthy prototype threshold."
        )
    else:
        print(
            f"prototype_similarity_threshold = {prototype_result['threshold']:.4f} "
            f"(TPR={prototype_result['true_positive_rate']:.2%}, "
            f"FPR={prototype_result['false_positive_rate']:.2%}, "
            f"pooled from {prototype_result['n_novel']} novel-class and "
            f"{prototype_result['n_known']} known-class LOCO predictions — a proxy "
            "for real unseen-sign photos, not a substitute for them)"
        )

    report = {
        "checkpoint": str(args.checkpoint),
        "calibrated_at_utc": datetime.now(UTC).isoformat(),
        "target_accuracy": args.target_accuracy,
        "base_confidence_threshold": base_result,
        "base_confidence_curve_csv": curve_path.relative_to(PROJECT_ROOT).as_posix(),
        "prototype_similarity_threshold": prototype_result,
        "prototype_calibration_is_proxy": True,
    }
    report_path = OUTPUT_DIR / "calibration_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Full report written to {report_path}")

    if args.apply:
        if base_result is None and prototype_result is None:
            print("Nothing measured; configs/config.yaml left unchanged.")
            return 0
        updates: dict[str, str] = {"calibrated": "true"}
        if base_result is not None:
            updates["base_confidence_threshold"] = f"{base_result['threshold']:.4f}"
        if prototype_result is not None:
            updates["prototype_similarity_threshold"] = (
                f"{prototype_result['threshold']:.4f}"
            )
        updates["calibration_reference"] = json.dumps(
            report_path.relative_to(PROJECT_ROOT).as_posix()
        )
        _patch_open_set_config(CONFIG_PATH, updates)
        print(f"Wrote calibrated thresholds into {CONFIG_PATH} (comments preserved)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
