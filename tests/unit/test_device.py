"""Tests for explicit device selection."""

import pytest

torch = pytest.importorskip(
    "torch", reason="PyTorch is required for device-selection tests"
)

from utils.device import DeviceSelectionError, describe_device, select_device


def test_cpu_selection_and_report() -> None:
    device = select_device("cpu")
    report = describe_device(device)

    assert device.type == "cpu"
    assert report.selected_device == "cpu"


def test_explicit_unavailable_cuda_is_rejected() -> None:
    if torch.cuda.is_available():
        pytest.skip("CUDA is available on this host")
    with pytest.raises(DeviceSelectionError, match="explicitly requested"):
        select_device("cuda")
