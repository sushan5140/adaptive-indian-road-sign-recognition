"""Explicit CPU/CUDA device selection and reporting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from utils.dependencies import DependencyUnavailableError

if TYPE_CHECKING:
    import torch


class DeviceSelectionError(RuntimeError):
    """Raised when a requested compute device cannot be selected."""


@dataclass(frozen=True, slots=True)
class DeviceReport:
    """Non-sensitive details about available and selected compute devices."""

    cuda_available: bool
    cuda_device_count: int
    cuda_device_name: str | None
    selected_device: str


def select_device(requested: str = "auto") -> torch.device:
    """Select ``auto``, ``cpu``, or ``cuda`` without silent CUDA fallback."""
    torch_module = _require_torch()
    normalized = requested.strip().lower()
    if normalized not in {"auto", "cpu", "cuda"}:
        raise DeviceSelectionError(
            f"Unsupported device {requested!r}; expected auto, cpu, or cuda"
        )
    if normalized == "cuda" and not torch_module.cuda.is_available():
        raise DeviceSelectionError(
            "CUDA was explicitly requested, but no CUDA device is available"
        )
    if normalized == "auto":
        normalized = "cuda" if torch_module.cuda.is_available() else "cpu"
    return cast("torch.device", torch_module.device(normalized))


def describe_device(device: torch.device) -> DeviceReport:
    """Report CUDA availability and the selected device."""
    torch_module = _require_torch()
    cuda_available = bool(torch_module.cuda.is_available())
    device_count = int(torch_module.cuda.device_count()) if cuda_available else 0
    device_name = str(torch_module.cuda.get_device_name(0)) if cuda_available else None
    return DeviceReport(
        cuda_available=cuda_available,
        cuda_device_count=device_count,
        cuda_device_name=device_name,
        selected_device=str(device),
    )


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise DependencyUnavailableError(
            "PyTorch is required for device selection"
        ) from error
    return torch
