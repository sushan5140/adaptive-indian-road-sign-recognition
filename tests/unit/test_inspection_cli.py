"""Smoke tests for the dataset inspection command."""

from pathlib import Path
from typing import Callable

from scripts.inspect_dataset import main

ImageFactory = Callable[..., Path]


def test_cli_writes_reports_outside_dataset(
    tmp_path: Path, make_image: ImageFactory
) -> None:
    root = tmp_path / "dataset"
    make_image(root / "stop" / "one.jpg")
    config = tmp_path / "config.yaml"
    config.write_text(
        "dataset:\n"
        f"  root: '{root.as_posix()}'\n"
        "  mode: directory\n"
        "  allowed_extensions: ['.jpg']\n",
        encoding="utf-8",
    )
    json_report = tmp_path / "reports" / "report.json"
    csv_report = tmp_path / "reports" / "report.csv"

    exit_code = main(
        [
            "--config",
            str(config),
            "--output-json",
            str(json_report),
            "--output-csv",
            str(csv_report),
        ]
    )

    assert exit_code == 0
    assert json_report.exists()
    assert csv_report.exists()


def test_cli_returns_nonzero_for_missing_dataset(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "dataset:\n"
        f"  root: '{(tmp_path / 'missing').as_posix()}'\n"
        "  mode: auto\n",
        encoding="utf-8",
    )

    assert main(["--config", str(config)]) == 2
