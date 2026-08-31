"""Tests for metadata-rich closed-set evaluation outputs."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="PyTorch is required for evaluation")

from evaluation.evaluator import Evaluator, save_evaluation_outputs


class _FixedModel(torch.nn.Module):
    def forward(self, images):
        return torch.tensor([[2.0, 0.0]], dtype=torch.float32).repeat(
            images.shape[0], 1
        )


def test_predictions_preserve_relative_path_and_review_id(tmp_path: Path) -> None:
    samples = [
        (
            torch.zeros(3, 8, 8),
            0,
            {
                "relative_image_path": "images/one.jpg",
                "review_id": "B6-0001",
            },
        ),
        (
            torch.zeros(3, 8, 8),
            1,
            {
                "relative_image_path": "images/two.jpg",
                "review_id": "B6-0002",
            },
        ),
    ]
    loader = torch.utils.data.DataLoader(samples, batch_size=2, shuffle=False)

    result = Evaluator(
        model=_FixedModel(),
        device=torch.device("cpu"),
        class_mapping={"give_way": 0, "no_entry": 1},
    ).evaluate(loader)
    paths = save_evaluation_outputs(result, tmp_path / "evaluation")

    with paths["predictions"].open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["image_path"] == "images/one.jpg"
    assert rows[0]["review_id"] == "B6-0001"
    assert rows[1]["correct"] == "False"
