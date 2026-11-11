"""quantlab: Quantitative finance foundations, US equity portfolio system, and research platform."""

from quantlab.compute import (
    DEFAULT_M2_ARRAY_BUDGET_BYTES,
    DeviceType,
    check_array_memory,
    get_device,
    set_global_seed,
)

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_M2_ARRAY_BUDGET_BYTES",
    "DeviceType",
    "__version__",
    "check_array_memory",
    "get_device",
    "set_global_seed",
]
