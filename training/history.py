"""Measured per-epoch history with atomic CSV and JSON persistence."""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class EpochRecord:
    """Measured loss, accuracy, learning rate, and duration for one epoch."""

    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float | None
    val_accuracy: float | None
    learning_rate: float
    elapsed_seconds: float

    def validate(self) -> None:
        """Reject missing, NaN, infinite, or nonsensical measurements."""
        if self.epoch < 0:
            raise ValueError("epoch must be non-negative")
        values = tuple(
            value
            for value in (
                self.train_loss,
                self.train_accuracy,
                self.val_loss,
                self.val_accuracy,
                self.learning_rate,
                self.elapsed_seconds,
            )
            if value is not None
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("history measurements must be finite")
        if not 0.0 <= self.train_accuracy <= 1.0:
            raise ValueError("train_accuracy must be in [0, 1]")
        if self.val_accuracy is not None and not 0.0 <= self.val_accuracy <= 1.0:
            raise ValueError("val_accuracy must be in [0, 1]")
        if self.train_loss < 0.0 or (self.val_loss is not None and self.val_loss < 0.0):
            raise ValueError("loss values must be non-negative")
        if self.learning_rate < 0.0 or self.elapsed_seconds < 0.0:
            raise ValueError("learning rate and elapsed time must be non-negative")


class TrainingHistory:
    """Ordered measured epoch records."""

    def __init__(self) -> None:
        self._records: list[EpochRecord] = []

    @property
    def records(self) -> tuple[EpochRecord, ...]:
        """Return immutable access to recorded epochs."""
        return tuple(self._records)

    def append(self, record: EpochRecord) -> None:
        """Append a validated record with a strictly increasing epoch."""
        record.validate()
        if self._records and record.epoch <= self._records[-1].epoch:
            raise ValueError("history epochs must be strictly increasing")
        self._records.append(record)

    def save(self, directory: str | Path) -> tuple[Path, Path]:
        """Atomically save ``history.csv`` and ``history.json``."""
        output_directory = Path(directory).expanduser().resolve()
        output_directory.mkdir(parents=True, exist_ok=True)
        csv_path = output_directory / "history.csv"
        json_path = output_directory / "history.json"
        rows = [asdict(record) for record in self._records]
        _write_csv_atomic(csv_path, rows)
        _write_text_atomic(
            json_path,
            json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=False),
        )
        return csv_path, json_path


def _write_csv_atomic(path: Path, rows: list[dict[str, float | int | None]]) -> None:
    fieldnames = [field.name for field in EpochRecord.__dataclass_fields__.values()]
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_path, path)
    except OSError:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _write_text_atomic(path: Path, content: str) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(content)
        os.replace(temporary_path, path)
    except OSError:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
