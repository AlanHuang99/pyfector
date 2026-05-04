"""
pyfector — Fast counterfactual estimators for panel data in Python.

A high-performance Python reimplementation of the R ``fect`` package,
featuring randomized SVD, GPU acceleration (CuPy), parallel computing
(joblib), Polars data ingestion, and seeded reproducibility.

Usage::

    import pyfector

    result = pyfector.fect(
        data=df,
        Y="outcome", D="treat",
        index=("unit", "year"),
        method="ife",
        r=(0, 5),
        se=True,
        seed=42,
    )
    result.summary()
    result.plot()

"""

__version__ = "0.1.6"

from .fect import fect, FectResult
from .backend import set_device, get_device
from .diagnostics import run_diagnostics, DiagnosticResult
from .plotting import plot

__all__ = [
    "fect",
    "FectResult",
    "set_device",
    "get_device",
    "run_diagnostics",
    "DiagnosticResult",
    "plot",
]
