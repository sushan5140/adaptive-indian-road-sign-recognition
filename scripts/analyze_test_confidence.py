"""Compute a confidence interval for the v2 test-set top-1 accuracy.

The 63-image external test set is small enough that a single point accuracy
(60.3%) is not trustworthy on its own. This script reports a Wilson score
interval and a bootstrap interval computed directly from the per-image
predictions already committed at outputs/v2_results/predictions.csv, and
flags any class whose support is too small for its per-class precision/
recall to be reported on its own (support < 3).

No model, checkpoint, or raw image access is required; this only reads the
already-measured prediction log, so it is safe to run in any environment.
"""

from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter
from pathlib import Path

PREDICTIONS_PATH = Path("outputs/v2_results/predictions.csv")
OUTPUT_PATH = Path("outputs/v2_results/test_accuracy_confidence.json")
SEED = 42
BOOTSTRAP_ITERATIONS = 20000
LOW_SUPPORT_THRESHOLD = 3


def wilson_interval(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return center - margin, center + margin


def bootstrap_interval(
    correct: list[bool], iterations: int, seed: int
) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(correct)
    accuracies = []
    for _ in range(iterations):
        sample_correct = sum(rng.choice(correct) for _ in range(n))
        accuracies.append(sample_correct / n)
    accuracies.sort()
    lo_index = int(0.025 * iterations)
    hi_index = int(0.975 * iterations)
    return accuracies[lo_index], accuracies[hi_index]


def main() -> None:
    rows = list(csv.DictReader(PREDICTIONS_PATH.open()))
    correct = [row["correct"] == "True" for row in rows]
    n = len(correct)
    k = sum(correct)
    point_accuracy = k / n

    wilson_lo, wilson_hi = wilson_interval(k, n)
    boot_lo, boot_hi = bootstrap_interval(correct, BOOTSTRAP_ITERATIONS, SEED)

    support_counts = Counter(row["true_label"] for row in rows)
    low_support_classes = sorted(
        label
        for label, count in support_counts.items()
        if count < LOW_SUPPORT_THRESHOLD
    )

    result = {
        "n_test_images": n,
        "n_correct": k,
        "point_accuracy": point_accuracy,
        "wilson_95_ci": {"low": wilson_lo, "high": wilson_hi},
        "bootstrap_95_ci": {
            "low": boot_lo,
            "high": boot_hi,
            "iterations": BOOTSTRAP_ITERATIONS,
            "seed": SEED,
        },
        "low_support_classes_below_threshold": low_support_classes,
        "low_support_threshold": LOW_SUPPORT_THRESHOLD,
        "note": (
            "Per-class precision/recall for the classes listed under "
            "low_support_classes_below_threshold is measured on fewer than "
            f"{LOW_SUPPORT_THRESHOLD} test images and must not be reported as "
            "a reliable percentage (e.g. a 1-image class showing 100% recall "
            "means one image, not perfect performance)."
        ),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2) + "\n")

    print(f"n={n}, correct={k}, point accuracy={point_accuracy:.4f}")
    print(f"Wilson 95% CI:    [{wilson_lo:.4f}, {wilson_hi:.4f}]")
    print(f"Bootstrap 95% CI: [{boot_lo:.4f}, {boot_hi:.4f}]")
    print(
        f"Low-support classes (<{LOW_SUPPORT_THRESHOLD} test images): {low_support_classes}"
    )
    print(f"Written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
