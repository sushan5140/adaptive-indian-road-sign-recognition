"""Dependency-independent CLI validation tests."""

from pathlib import Path

from scripts.evaluate import main as evaluate_main
from scripts.train import main as train_main


def test_training_fails_before_model_creation_when_dataset_missing(
    tmp_path: Path,
) -> None:
    config = _write_config(tmp_path)
    assert train_main(["--config", str(config)]) == 2


def test_evaluation_fails_before_checkpoint_loading_when_dataset_missing(
    tmp_path: Path,
) -> None:
    config = _write_config(tmp_path)
    assert (
        evaluate_main(
            ["--config", str(config), "--checkpoint", str(tmp_path / "missing.pt")]
        )
        == 2
    )


def _write_config(tmp_path: Path) -> Path:
    config = tmp_path / "config.yaml"
    config.write_text(
        "dataset:\n"
        f"  root: '{(tmp_path / 'missing').as_posix()}'\n"
        "  mode: auto\n",
        encoding="utf-8",
    )
    return config
