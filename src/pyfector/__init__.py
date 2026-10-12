"""
pyfector: counterfactual estimators for panel data in Python.

A Python implementation of counterfactual estimators for panel data,
following Liu, Wang and Xu (2024). See the
[R fect package](https://github.com/xuyiqing/fect) for the R implementation.

Usage
-----

```python

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
```

API modules
-----------
`pyfector.diagnostics` provides pre-trend tests, holdout/refit placebo tests,
and post-treatment sensitivity diagnostics. `pyfector.inference` provides
bootstrap, jackknife, and permutation inference. `pyfector.panel`,
`pyfector.estimators`, and `pyfector.cv` describe panel construction,
estimation, and cross-validation.

"""

__version__ = "0.2.3"

from .fect import fect, FectResult
from .backend import set_device, get_device
from .diagnostics import (
    run_diagnostics,
    Diagnostics,
    DiagnosticResult,
    TostResult,
    PretrendFResult,
    EquivFResult,
    PlaceboResult,
    CarryoverResult,
    LooResult,
)
from .plotting import plot

__all__ = [
    "fect",
    "FectResult",
    "set_device",
    "get_device",
    "run_diagnostics",
    "Diagnostics",
    "DiagnosticResult",
    "TostResult",
    "PretrendFResult",
    "EquivFResult",
    "PlaceboResult",
    "CarryoverResult",
    "LooResult",
    "plot",
]
