"""YAML configuration loading and explicit nested overrides."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


class ConfigurationError(ValueError):
    """Raised when project configuration is missing or malformed."""


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping without executing arbitrary constructors."""
    config_path = Path(path).expanduser().resolve()
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ConfigurationError(
            f"Could not load configuration {config_path}"
        ) from error
    if not isinstance(payload, dict):
        raise ConfigurationError("Configuration root must be a mapping")
    return payload


def apply_overrides(
    config: Mapping[str, Any], overrides: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Return a deep copy with non-``None`` CLI values applied by section."""
    resolved = copy.deepcopy(dict(config))
    for section_name, values in overrides.items():
        section = resolved.setdefault(section_name, {})
        if not isinstance(section, dict):
            raise ConfigurationError(
                f"Configuration section {section_name!r} must be a mapping"
            )
        for key, value in values.items():
            if value is not None:
                section[key] = value
    return resolved


def require_mapping(config: Mapping[str, Any], section: str) -> dict[str, Any]:
    """Return one required configuration section as a mutable mapping copy."""
    value = config.get(section)
    if not isinstance(value, dict):
        raise ConfigurationError(f"Configuration requires a {section!r} mapping")
    return dict(value)
