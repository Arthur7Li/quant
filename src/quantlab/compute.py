"""Hardware device selection, deterministic seeding, and memory guardrails.

Enforces the device-agnostic and 8 GB Apple M2 memory rules defined in
QUANT_PROJECT_DESIGN.md §3.
"""

from __future__ import annotations

import importlib
import math
import os
import random
from typing import Any, Final, Literal

import numpy as np

DeviceType = Literal["cuda", "mps", "cpu"]

VALID_PREFERENCES: Final[frozenset[str]] = frozenset({"auto", "cuda", "mps", "cpu"})

# Default single-allocation cap (2 GiB) to protect the 8 GB unified memory on Apple M2.
DEFAULT_M2_ARRAY_BUDGET_BYTES: Final[int] = 2 * 1024**3


def _probe_torch_backends() -> tuple[bool, bool]:
    """Probe whether PyTorch is installed and whether CUDA or MPS is available.

    Returns
    -------
    tuple[bool, bool]
        `(cuda_available, mps_available)`. Returns `(False, False)` when `torch`
        is not installed in the current environment.
    """
    try:
        torch_mod: Any = importlib.import_module("torch")
    except ImportError:
        return False, False

    cuda_submod: Any = getattr(torch_mod, "cuda", None)
    cuda_available = bool(
        cuda_submod is not None
        and callable(getattr(cuda_submod, "is_available", None))
        and cuda_submod.is_available()
    )

    backends_mod: Any = getattr(torch_mod, "backends", None)
    mps_submod: Any = getattr(backends_mod, "mps", None) if backends_mod is not None else None
    mps_available = bool(
        mps_submod is not None
        and callable(getattr(mps_submod, "is_available", None))
        and mps_submod.is_available()
        and (not callable(getattr(mps_submod, "is_built", None)) or bool(mps_submod.is_built()))
    )
    return cuda_available, mps_available


def get_device(prefer: str | None = None) -> DeviceType:
    """Select the compute device following the `cuda -> mps -> cpu` hierarchy.

    Parameters
    ----------
    prefer : str | None, optional
        Explicit device preference (`"auto"`, `"cuda"`, `"mps"`, or `"cpu"`).
        If `None`, checks the `QUANTLAB_DEVICE` environment variable before
        defaulting to `"auto"`.

    Returns
    -------
    DeviceType
        One of `"cuda"`, `"mps"`, or `"cpu"`.

    Raises
    ------
    ValueError
        If `prefer` (or `QUANTLAB_DEVICE`) is not in `{"auto", "cuda", "mps", "cpu"}`.
    """
    raw_pref = prefer if prefer is not None else os.environ.get("QUANTLAB_DEVICE", "auto")
    normalized = raw_pref.strip().lower()

    if normalized not in VALID_PREFERENCES:
        msg = (
            f"Invalid device preference {raw_pref!r}. Expected one of {sorted(VALID_PREFERENCES)}."
        )
        raise ValueError(msg)

    if normalized == "cpu":
        return "cpu"

    cuda_ok, mps_ok = _probe_torch_backends()

    if normalized == "cuda":
        return "cuda" if cuda_ok else "cpu"
    if normalized == "mps":
        return "mps" if mps_ok else "cpu"

    # normalized == "auto": strict priority cuda -> mps -> cpu
    if cuda_ok:
        return "cuda"
    if mps_ok:
        return "mps"
    return "cpu"


def get_memory_budget_bytes(default_bytes: int = DEFAULT_M2_ARRAY_BUDGET_BYTES) -> int:
    """Return the maximum byte budget for a single in-memory array allocation.

    Reads `QUANTLAB_MAX_ARRAY_BYTES` from the environment if set; otherwise
    returns `default_bytes` (2 GiB by default for the 8 GB M2 guardrail).
    """
    env_val = os.environ.get("QUANTLAB_MAX_ARRAY_BYTES")
    if env_val is not None:
        budget = int(env_val)
        if budget <= 0:
            msg = f"QUANTLAB_MAX_ARRAY_BYTES must be positive, got {budget}."
            raise ValueError(msg)
        return budget
    if default_bytes <= 0:
        msg = f"default_bytes must be positive, got {default_bytes}."
        raise ValueError(msg)
    return default_bytes


def check_array_memory(
    shape: tuple[int, ...],
    dtype: np.dtype[Any] | type[np.generic] = np.float64,
    max_bytes: int | None = None,
) -> int:
    """Verify that an array of `shape` and `dtype` fits within the memory budget.

    Parameters
    ----------
    shape : tuple[int, ...]
        Dimensions of the proposed array.
    dtype : np.dtype[Any] | type[np.generic], default np.float64
        NumPy data type of the array elements.
    max_bytes : int | None, optional
        Explicit byte limit. Defaults to `get_memory_budget_bytes()`.

    Returns
    -------
    int
        Total bytes required by the array.

    Raises
    ------
    ValueError
        If any dimension in `shape` is negative.
    MemoryError
        If the required bytes exceed `max_bytes`.
    """
    if any(dim < 0 for dim in shape):
        msg = f"Array dimensions must be non-negative, got shape={shape}."
        raise ValueError(msg)

    resolved_dtype = np.dtype(dtype)
    n_elements = math.prod(shape) if shape else 1
    required_bytes = int(n_elements * resolved_dtype.itemsize)
    budget = max_bytes if max_bytes is not None else get_memory_budget_bytes()

    if budget <= 0:
        msg = f"Memory budget max_bytes must be positive, got {budget}."
        raise ValueError(msg)

    if required_bytes > budget:
        mib_req = required_bytes / (1024**2)
        mib_cap = budget / (1024**2)
        msg = (
            f"Array of shape {shape} and dtype {resolved_dtype} requires "
            f"{mib_req:.2f} MiB ({required_bytes} bytes), exceeding budget of "
            f"{mib_cap:.2f} MiB ({budget} bytes). Use out-of-core Parquet/DuckDB "
            "scans or batching."
        )
        raise MemoryError(msg)

    return required_bytes


def set_global_seed(seed: int = 42) -> np.random.Generator:
    """Seed Python `random`, legacy `np.random`, PyTorch (if installed), and return a Generator.

    Parameters
    ----------
    seed : int, default 42
        Non-negative integer seed.

    Returns
    -------
    np.random.Generator
        A freshly seeded `np.random.default_rng(seed)` instance.
    """
    if seed < 0:
        msg = f"Seed must be non-negative, got {seed}."
        raise ValueError(msg)

    random.seed(seed)
    np.random.seed(seed % (2**32))

    try:
        torch_mod: Any = importlib.import_module("torch")
        if callable(getattr(torch_mod, "manual_seed", None)):
            torch_mod.manual_seed(seed)
        cuda_mod: Any = getattr(torch_mod, "cuda", None)
        if (
            cuda_mod is not None
            and callable(getattr(cuda_mod, "is_available", None))
            and cuda_mod.is_available()
            and callable(getattr(cuda_mod, "manual_seed_all", None))
        ):
            cuda_mod.manual_seed_all(seed)
    except ImportError:
        pass

    return np.random.default_rng(seed)
