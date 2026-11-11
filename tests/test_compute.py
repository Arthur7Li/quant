"""Unit and property tests for quantlab.compute."""

from __future__ import annotations

import builtins
import importlib
import random
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from quantlab.compute import (
    DEFAULT_M2_ARRAY_BUDGET_BYTES,
    check_array_memory,
    get_device,
    get_memory_budget_bytes,
    set_global_seed,
)


def _make_fake_torch(*, cuda: bool, mps: bool, mps_built: bool = True) -> SimpleNamespace:
    """Create a lightweight fake torch module for deterministic backend testing."""
    return SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: cuda,
            manual_seed_all=lambda _seed: None,
        ),
        backends=SimpleNamespace(
            mps=SimpleNamespace(
                is_available=lambda: mps,
                is_built=lambda: mps_built,
            )
        ),
        manual_seed=lambda _seed: None,
    )


def test_get_device_priority_cuda_over_mps_and_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `cuda` wins when both CUDA and MPS report available."""
    monkeypatch.delenv("QUANTLAB_DEVICE", raising=False)
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda _name: _make_fake_torch(cuda=True, mps=True),
    )
    assert get_device() == "cuda"
    assert get_device("auto") == "cuda"
    assert get_device("cuda") == "cuda"
    assert get_device("mps") == "mps"
    assert get_device("cpu") == "cpu"


def test_get_device_priority_mps_when_no_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `mps` is selected on Apple Silicon when CUDA is unavailable."""
    monkeypatch.delenv("QUANTLAB_DEVICE", raising=False)
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda _name: _make_fake_torch(cuda=False, mps=True),
    )
    assert get_device() == "mps"
    assert get_device("cuda") == "cpu"
    assert get_device("mps") == "mps"


def test_get_device_falls_back_to_cpu_when_mps_not_built(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify fallback to `cpu` when MPS is not built."""
    monkeypatch.delenv("QUANTLAB_DEVICE", raising=False)
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda _name: _make_fake_torch(cuda=False, mps=True, mps_built=False),
    )
    assert get_device() == "cpu"
    assert get_device("mps") == "cpu"


def test_get_device_falls_back_to_cpu_without_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify graceful fallback to `cpu` when PyTorch is not installed."""
    monkeypatch.delenv("QUANTLAB_DEVICE", raising=False)

    def _raise_import_error(_name: str) -> Any:
        raise ImportError("No module named 'torch'")

    monkeypatch.setattr(importlib, "import_module", _raise_import_error)
    assert get_device() == "cpu"
    assert get_device("cuda") == "cpu"
    assert get_device("mps") == "cpu"


def test_get_device_env_override_and_invalid_preference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify QUANTLAB_DEVICE override and ValueError on invalid device names."""
    monkeypatch.setenv("QUANTLAB_DEVICE", "CPU")
    assert get_device() == "cpu"

    monkeypatch.setenv("QUANTLAB_DEVICE", "tpu")
    with pytest.raises(ValueError, match="Invalid device preference"):
        get_device()

    with pytest.raises(ValueError, match="Invalid device preference"):
        get_device("quantum")


def test_memory_budget_and_check_array_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify known-answer byte calculations and MemoryError enforcement."""
    monkeypatch.delenv("QUANTLAB_MAX_ARRAY_BYTES", raising=False)
    assert get_memory_budget_bytes() == DEFAULT_M2_ARRAY_BUDGET_BYTES

    # Known-answer: (2520, 100) float64 panel = 252,000 * 8 = 2,016,000 bytes
    n_bytes = check_array_memory((2520, 100), dtype=np.float64)
    assert n_bytes == 2_016_000

    # Scalar 0D shape
    assert check_array_memory((), dtype=np.float32) == 4

    # Exceeding explicit 1 MiB budget raises MemoryError
    with pytest.raises(MemoryError, match="exceeding budget"):
        check_array_memory((2520, 100), dtype=np.float64, max_bytes=1024 * 1024)

    # Negative dimension raises ValueError
    with pytest.raises(ValueError, match="non-negative"):
        check_array_memory((100, -1), dtype=np.float64)

    # Non-positive budget raises ValueError
    with pytest.raises(ValueError, match="must be positive"):
        check_array_memory((10, 10), max_bytes=0)

    monkeypatch.setenv("QUANTLAB_MAX_ARRAY_BYTES", "4096")
    assert get_memory_budget_bytes() == 4096

    monkeypatch.setenv("QUANTLAB_MAX_ARRAY_BYTES", "-10")
    with pytest.raises(ValueError, match="QUANTLAB_MAX_ARRAY_BYTES must be positive"):
        get_memory_budget_bytes()

    monkeypatch.delenv("QUANTLAB_MAX_ARRAY_BYTES", raising=False)
    with pytest.raises(ValueError, match="default_bytes must be positive"):
        get_memory_budget_bytes(default_bytes=0)


@given(
    rows=st.integers(min_value=0, max_value=5000),
    cols=st.integers(min_value=0, max_value=500),
)
def test_check_array_memory_property_exact_float64_bytes(rows: int, cols: int) -> None:
    """Property test: float64 2D array byte requirement equals 8 * rows * cols."""
    expected = 8 * rows * cols
    actual = check_array_memory((rows, cols), dtype=np.float64, max_bytes=max(1, expected + 8))
    assert actual == expected


def test_set_global_seed_reproducibility(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify deterministic seeding across Python random, NumPy, and fake torch."""
    seeded_calls: list[int] = []

    fake_torch = SimpleNamespace(
        manual_seed=lambda s: seeded_calls.append(s),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            manual_seed_all=lambda s: seeded_calls.append(s),
        ),
    )
    monkeypatch.setattr(importlib, "import_module", lambda _name: fake_torch)

    rng1 = set_global_seed(12345)
    py_val1 = random.random()
    np_legacy1 = float(np.random.rand())
    arr1 = rng1.standard_normal(5)

    rng2 = set_global_seed(12345)
    py_val2 = random.random()
    np_legacy2 = float(np.random.rand())
    arr2 = rng2.standard_normal(5)

    assert py_val1 == py_val2
    assert np_legacy1 == np_legacy2
    np.testing.assert_array_equal(arr1, arr2)
    assert seeded_calls == [12345, 12345, 12345, 12345]

    with pytest.raises(ValueError, match="non-negative"):
        set_global_seed(-1)


def test_set_global_seed_without_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify set_global_seed succeeds when torch is not installed."""

    def _raise_import_error(_name: str) -> Any:
        raise builtins.ImportError("no torch")

    monkeypatch.setattr(importlib, "import_module", _raise_import_error)
    rng = set_global_seed(99)
    assert isinstance(rng, np.random.Generator)
