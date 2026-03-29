"""
Array backend abstraction: seamless NumPy / CuPy switching.

Usage:
    from pyfector.backend import get_backend, set_device
    set_device("gpu")          # or "cpu" (default)
    xp = get_backend()         # returns numpy or cupy module
    A = xp.random.randn(100, 50)

All pyfector internals call ``get_backend()`` to obtain the active array
module, so every matrix operation automatically runs on CPU or GPU.
"""

from __future__ import annotations

import numpy as np
from typing import Literal

_DEVICE: Literal["cpu", "gpu"] = "cpu"
_CUPY_AVAILABLE: bool | None = None


def _check_cupy() -> bool:
    global _CUPY_AVAILABLE
    if _CUPY_AVAILABLE is None:
        try:
            import cupy  # noqa: F401
            _CUPY_AVAILABLE = True
        except ImportError:
            _CUPY_AVAILABLE = False
    return _CUPY_AVAILABLE


def set_device(device: Literal["cpu", "gpu"]) -> None:
    """Set the compute device globally."""
    global _DEVICE
    if device == "gpu" and not _check_cupy():
        raise ImportError(
            "CuPy is required for GPU support. Install it with: "
            "pip install cupy-cuda12x  (adjust for your CUDA version)"
        )
    _DEVICE = device


def get_device() -> Literal["cpu", "gpu"]:
    """Return the current device."""
    return _DEVICE


def get_backend():
    """Return the active array module (numpy or cupy)."""
    if _DEVICE == "gpu":
        import cupy
        return cupy
    return np


def to_numpy(arr) -> np.ndarray:
    """Ensure *arr* is a host numpy array (move from GPU if needed)."""
    if _DEVICE == "gpu":
        import cupy
        if isinstance(arr, cupy.ndarray):
            return cupy.asnumpy(arr)
    return np.asarray(arr)


def to_device(arr):
    """Move a numpy array to the current device."""
    if _DEVICE == "gpu":
        import cupy
        return cupy.asarray(arr)
    return np.asarray(arr)


def make_rng(seed: int | None = None):
    """Create a seeded RNG for the active backend.

    Returns a ``numpy.random.Generator`` or ``cupy.random.Generator``
    depending on the current device.
    """
    if _DEVICE == "gpu":
        import cupy
        if seed is not None:
            return cupy.random.default_rng(seed)
        return cupy.random.default_rng()
    if seed is not None:
        return np.random.default_rng(seed)
    return np.random.default_rng()
