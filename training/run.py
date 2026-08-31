"""Unique run identity, resolved configuration, and safe environment metadata."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


def create_run_id(backbone: str, *, timestamp: datetime | None = None) -> str:
    """Create a timestamped, filesystem-safe run identifier."""
    current = timestamp or datetime.now(UTC)
    safe_backbone = re.sub(r"[^A-Za-z0-9_.-]+", "_", backbone).strip("_")
    if not safe_backbone:
        raise ValueError("backbone must contain at least one safe identifier character")
    return f"{current.strftime('%Y%m%d_%H%M%S_%f')}_{safe_backbone}"


def create_run_directory(base_directory: str | Path, run_id: str) -> Path:
    """Create a new run directory and refuse to overwrite an existing run."""
    if Path(run_id).name != run_id or not run_id:
        raise ValueError("run_id must be a plain non-empty directory name")
    destination = Path(base_directory).expanduser().resolve() / run_id
    destination.mkdir(parents=True, exist_ok=False)
    return destination


def collect_environment_details(
    *,
    selected_device: str,
    dataset_sizes: dict[str, int],
    class_count: int,
) -> dict[str, Any]:
    """Collect version and training context without hostname or user information."""
    return {
        "python_version": platform.python_version(),
        "torch_version": _package_version("torch"),
        "torchvision_version": _package_version("torchvision"),
        "timm_version": _package_version("timm"),
        "selected_device": selected_device,
        "dataset_sizes": dict(dataset_sizes),
        "class_count": class_count,
    }


def save_run_metadata(
    run_directory: str | Path,
    *,
    resolved_config: dict[str, Any],
    environment: dict[str, Any],
) -> tuple[Path, Path]:
    """Atomically save resolved YAML and environment JSON for a run."""
    directory = Path(run_directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    config_path = directory / "resolved_config.yaml"
    environment_path = directory / "environment.json"
    _write_text_atomic(
        config_path,
        yaml.safe_dump(resolved_config, sort_keys=False, allow_unicode=True),
    )
    _write_text_atomic(
        environment_path,
        json.dumps(environment, ensure_ascii=False, indent=2, allow_nan=False),
    )
    return config_path, environment_path


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


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
