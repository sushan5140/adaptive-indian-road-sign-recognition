"""Tests for measured history and run metadata persistence."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from training.history import EpochRecord, TrainingHistory
from training.run import create_run_directory, create_run_id, save_run_metadata


def test_history_saves_csv_and_json(tmp_path: Path) -> None:
    history = TrainingHistory()
    history.append(
        EpochRecord(
            epoch=0,
            train_loss=1.0,
            train_accuracy=0.5,
            val_loss=0.9,
            val_accuracy=0.6,
            learning_rate=0.001,
            elapsed_seconds=1.5,
        )
    )

    csv_path, json_path = history.save(tmp_path)

    assert csv_path.exists()
    assert json.loads(json_path.read_text(encoding="utf-8"))[0]["epoch"] == 0


def test_history_rejects_non_finite_metrics() -> None:
    history = TrainingHistory()
    with pytest.raises(ValueError, match="finite"):
        history.append(
            EpochRecord(
                epoch=0,
                train_loss=float("nan"),
                train_accuracy=0.5,
                val_loss=1.0,
                val_accuracy=0.5,
                learning_rate=0.001,
                elapsed_seconds=1.0,
            )
        )


def test_history_accepts_explicitly_absent_validation(tmp_path: Path) -> None:
    history = TrainingHistory()
    history.append(
        EpochRecord(
            epoch=0,
            train_loss=1.0,
            train_accuracy=0.5,
            val_loss=None,
            val_accuracy=None,
            learning_rate=0.001,
            elapsed_seconds=1.0,
        )
    )

    _, json_path = history.save(tmp_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload[0]["val_loss"] is None
    assert payload[0]["val_accuracy"] is None


def test_run_identity_and_metadata_do_not_overwrite(tmp_path: Path) -> None:
    run_id = create_run_id(
        "mobilenetv3_small_100",
        timestamp=datetime(2026, 8, 30, 15, 35, tzinfo=UTC),
    )
    directory = create_run_directory(tmp_path, run_id)
    save_run_metadata(
        directory,
        resolved_config={"seed": 42},
        environment={"python_version": "test"},
    )

    assert (directory / "resolved_config.yaml").exists()
    assert (directory / "environment.json").exists()
    with pytest.raises(FileExistsError):
        create_run_directory(tmp_path, run_id)
