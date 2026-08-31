"""Closed-set model evaluation and measured report persistence."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from evaluation.metrics import ClassificationMetrics, calculate_classification_metrics
from utils.dependencies import DependencyUnavailableError


class EvaluationError(RuntimeError):
    """Raised when evaluation data or output configuration is invalid."""


@dataclass(frozen=True, slots=True)
class PredictionRecord:
    """One measured closed-set classifier prediction."""

    image_path: str
    review_id: str
    true_label: str
    predicted_label: str
    confidence: float
    correct: bool
    true_index: int
    predicted_index: int


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Measured metrics and individual predictions."""

    metrics: ClassificationMetrics
    predictions: tuple[PredictionRecord, ...]


class Evaluator:
    """Evaluate a classifier under no-grad mode on one device."""

    def __init__(
        self,
        *,
        model: Any,
        device: Any,
        class_mapping: Mapping[str, int],
    ) -> None:
        self.torch = _require_torch()
        self.model = model.to(device)
        self.device = device
        self.class_mapping = dict(class_mapping)
        self.index_to_class = {
            index: label for label, index in self.class_mapping.items()
        }
        if sorted(self.index_to_class) != list(range(len(self.index_to_class))):
            raise EvaluationError("class mapping must be contiguous from zero")

    def evaluate(self, loader: Any) -> EvaluationResult:
        """Measure predictions and standard multiclass metrics."""
        try:
            if len(loader) == 0:
                raise EvaluationError("Cannot evaluate an empty DataLoader")
        except TypeError as error:
            raise EvaluationError("DataLoader must provide a finite length") from error
        self.model.eval()
        predictions: list[PredictionRecord] = []
        true_indices: list[int] = []
        predicted_indices: list[int] = []
        with self.torch.no_grad():
            for batch in loader:
                if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                    raise EvaluationError("Each batch must contain images and targets")
                images, targets = batch[0], batch[1]
                metadata = batch[2] if len(batch) >= 3 else None
                images = images.to(self.device)
                targets = targets.to(self.device)
                logits = self.model(images)
                probabilities = self.torch.softmax(logits, dim=1)
                confidence, predicted = probabilities.max(dim=1)
                target_values = [int(value) for value in targets.cpu().tolist()]
                predicted_values = [int(value) for value in predicted.cpu().tolist()]
                confidence_values = [
                    float(value) for value in confidence.cpu().tolist()
                ]
                image_paths, review_ids = _extract_prediction_metadata(
                    metadata, len(target_values)
                )
                for index, true_index in enumerate(target_values):
                    predicted_index = predicted_values[index]
                    try:
                        true_label = self.index_to_class[true_index]
                        predicted_label = self.index_to_class[predicted_index]
                    except KeyError as error:
                        raise EvaluationError(
                            f"Prediction contains unknown class index {error.args[0]}"
                        ) from error
                    predictions.append(
                        PredictionRecord(
                            image_path=image_paths[index],
                            review_id=review_ids[index],
                            true_label=true_label,
                            predicted_label=predicted_label,
                            confidence=confidence_values[index],
                            correct=true_index == predicted_index,
                            true_index=true_index,
                            predicted_index=predicted_index,
                        )
                    )
                true_indices.extend(target_values)
                predicted_indices.extend(predicted_values)
        if not predictions:
            raise EvaluationError("DataLoader yielded no samples")
        measured_metrics = calculate_classification_metrics(
            true_indices,
            predicted_indices,
            self.class_mapping,
        )
        return EvaluationResult(
            metrics=measured_metrics,
            predictions=tuple(predictions),
        )


def save_evaluation_outputs(
    result: EvaluationResult,
    output_directory: str | Path,
    *,
    dataset_root: str | Path | None = None,
) -> dict[str, Path]:
    """Write metrics, per-class data, confusion matrix, and predictions."""
    directory = Path(output_directory).expanduser().resolve()
    if dataset_root is not None:
        root = Path(dataset_root).expanduser().resolve()
        if directory == root or directory.is_relative_to(root):
            raise EvaluationError(
                f"Evaluation outputs must be outside dataset root {root}"
            )
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "metrics": directory / "metrics.json",
        "per_class": directory / "per_class_metrics.csv",
        "confusion_matrix": directory / "confusion_matrix.csv",
        "predictions": directory / "predictions.csv",
    }
    metrics_payload = result.metrics.to_dict()
    metrics_payload["confidence_note"] = (
        "Maximum softmax probability is closed-set classifier confidence, not an "
        "open-set or unknown-sign score."
    )
    paths["metrics"].write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    _write_dataclass_csv(
        paths["per_class"],
        [asdict(item) for item in result.metrics.per_class],
    )
    labels = [item.label for item in result.metrics.per_class]
    with paths["confusion_matrix"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("true_label", *labels))
        for label, row in zip(labels, result.metrics.confusion_matrix, strict=True):
            writer.writerow((label, *row))
    _write_dataclass_csv(
        paths["predictions"],
        [asdict(item) for item in result.predictions],
    )
    return paths


def _extract_prediction_metadata(
    metadata: Any, count: int
) -> tuple[list[str], list[str]]:
    if not isinstance(metadata, dict):
        return [""] * count, [""] * count
    path_value = metadata.get("relative_image_path", metadata.get("image_path"))
    review_value = metadata.get("review_id")
    image_paths = (
        [str(item) for item in path_value]
        if isinstance(path_value, (list, tuple)) and len(path_value) == count
        else [""] * count
    )
    review_ids = (
        [str(item) for item in review_value]
        if isinstance(review_value, (list, tuple)) and len(review_value) == count
        else [""] * count
    )
    return image_paths, review_ids


def _write_dataclass_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise EvaluationError(f"Cannot write empty report {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise DependencyUnavailableError(
            "PyTorch is required for evaluation"
        ) from error
    return torch
