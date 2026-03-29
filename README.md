# pyfect

> **Alpha (v0.1.x)** -- This package is in early development. Results have been validated against R fect on synthetic data, but edge cases may remain. **APIs are not stable and may change without notice.** Please verify critical results independently. Bug reports, feature requests, and contributions are welcome via [GitHub Issues](https://github.com/AlanHuang99/pyfect/issues).

Fast counterfactual estimators for panel data in Python.

A high-performance reimplementation of the R [fect](https://github.com/xuyiqing/fect) package (Liu, Wang & Xu, 2024, *AJPS*), featuring GPU acceleration, parallel computing, and Polars data handling.

## Installation

```bash
pip install fectpy
```

For GPU support:
```bash
pip install fectpy[gpu]
```

> **Note**: The PyPI package is `fectpy`, but the import name is `pyfect` — i.e. `import pyfect`.

## Quick Start

```python
import pyfect

result = pyfect.fect(
    data=df,                    # Polars or Pandas DataFrame
    Y="outcome",                # outcome column
    D="treatment",              # binary treatment indicator
    index=("unit_id", "year"),  # (unit, time) columns
    X=["gdp", "population"],   # covariates (optional)
    method="ife",               # "fe", "ife", "mc", "cfe"
    r=(0, 5),                   # cross-validate over 0..5 factors
    se=True,                    # bootstrap standard errors
    nboots=500,                 # bootstrap replications
    device="cpu",               # "cpu" or "gpu"
    n_jobs=4,                   # parallel workers
    seed=42,                    # full reproducibility
)

# Results
print(result.summary())

# Plotting (gap plot, status heatmap, etc.)
result.plot(kind="gap")
result.plot(kind="status")

# Diagnostic tests
diag = result.diagnose(f_threshold=0.5, tost_threshold=0.36, loo=True)
print(diag.summary())
```

## Methods

| Method | Description | Key Parameter | When to Use |
|--------|-------------|---------------|-------------|
| `"fe"` | Fixed effects counterfactual | `force` | Parallel trends holds |
| `"ife"` | Interactive fixed effects | `r` (number of factors) | Unobserved time-varying confounders |
| `"mc"` | Matrix completion | `lam` (nuclear norm penalty) | Alternative to IFE |
| `"cfe"` | Complex fixed effects | `Z`, `Q` | Unit/time-varying interactions |

## API Reference

### `pyfect.fect()`

Main estimation function. Returns a `FectResult` object.

```python
pyfect.fect(
    data,                           # DataFrame (Polars or Pandas)
    Y: str,                         # outcome column name
    D: str,                         # treatment column name (0/1)
    index: tuple[str, str],         # (unit_id, time) column names
    X: list[str] = None,            # covariate column names
    W: str = None,                  # weight column name
    method: str = "ife",            # "fe", "ife", "mc", "cfe", "both"
    force: str = "two-way",         # "none", "unit", "time", "two-way"
    r: int | tuple = 0,             # factors; tuple (min, max) for CV
    lam: float = None,              # lambda for MC; None = auto CV
    nlambda: int = 10,              # lambda grid size for MC CV
    CV: bool = True,                # cross-validate r or lambda
    k: int = 10,                    # CV folds
    cv_prop: float = 0.1,           # fraction of control obs to mask
    criterion: str = "mspe",        # CV criterion: "mspe", "gmspe", "mad"
    se: bool = False,               # compute standard errors
    vartype: str = "bootstrap",     # "bootstrap" or "jackknife"
    nboots: int = 200,              # bootstrap replications
    alpha: float = 0.05,            # significance level
    tol: float = 1e-7,              # convergence tolerance
    max_iter: int = 5000,           # max EM iterations
    min_T0: int = 1,               # min pre-treatment periods per unit
    normalize: bool = False,        # normalize outcome by SD
    device: str = "cpu",            # "cpu" or "gpu"
    n_jobs: int = 1,                # parallel workers
    seed: int = None,               # random seed
)
```

### `FectResult` object

| Attribute | Type | Description |
|-----------|------|-------------|
| `att_avg` | `float` | Overall average treatment effect on the treated |
| `att_avg_unit` | `float` | Unit-averaged ATT |
| `att_on` | `ndarray` | Dynamic ATT by relative time to treatment |
| `time_on` | `ndarray` | Relative time indices |
| `count_on` | `ndarray` | Observation counts per relative time |
| `beta` | `ndarray` | Covariate coefficients |
| `Y_ct` | `ndarray` | T x N counterfactual outcome matrix |
| `eff` | `ndarray` | T x N treatment effect matrix |
| `factors` | `ndarray` | T x r estimated factors (IFE only) |
| `loadings` | `ndarray` | N x r estimated loadings (IFE only) |
| `sigma2` | `float` | Error variance estimate |
| `r_cv` | `int` | CV-selected number of factors |
| `lambda_cv` | `float` | CV-selected lambda |
| `inference` | `InferenceResult` | Bootstrap/jackknife results (if `se=True`) |

Methods:
- `result.summary()` — formatted summary table
- `result.plot(kind="gap")` — dynamic effects plot
- `result.plot(kind="status")` — treatment status heatmap
- `result.plot(kind="factors")` — latent factors
- `result.plot(kind="counterfactual", units=[1, 5])` — actual vs counterfactual
- `result.diagnose(...)` — run diagnostic tests

### `InferenceResult` (when `se=True`)

| Attribute | Description |
|-----------|-------------|
| `att_avg_se` | Standard error of overall ATT |
| `att_avg_ci` | 95% confidence interval (tuple) |
| `att_avg_pval` | p-value for H0: ATT=0 |
| `att_on_se` | Per-period standard errors |
| `att_on_ci_lower` | Per-period CI lower bounds |
| `att_on_ci_upper` | Per-period CI upper bounds |
| `att_on_pval` | Per-period p-values |
| `att_avg_boot` | Bootstrap distribution of overall ATT |
| `att_on_boot` | Bootstrap distribution of dynamic ATTs |

### Diagnostic Tests

```python
diag = result.diagnose(
    f_threshold=0.5,         # equivalence F-test threshold
    tost_threshold=0.36,     # TOST equivalence bound
    placebo_period=(-5, -1), # placebo test window
    loo=True,                # leave-one-period-out
)
print(diag.summary())
```

| Test | What it Tests | Key Output |
|------|---------------|------------|
| **Pre-trend F-test** | Joint significance of pre-treatment ATTs | `f_stat`, `f_pval` |
| **Equivalence F-test** | Pre-trends within equivalence bounds | `equiv_f_pval` |
| **TOST** | Per-period equivalence | `tost_pvals` |
| **Placebo test** | No effect in specified pre-period window | `placebo_att`, `placebo_pval` |
| **Carryover test** | No lingering effect after treatment ends | `carryover_att`, `carryover_pval` |
| **Leave-one-out** | Sensitivity of ATT to dropping periods | `loo_max_change` |

## Validation Against R fect

pyfect is validated against the original R fect package on identical simulated data. The table below shows exact comparison results (N=200, T=50):

### Point Estimates

| Scenario | pyfect ATT | R fect ATT | Difference | True ATT |
|----------|------------|------------|------------|----------|
| FE (no factors) | 4.995583 | 4.995640 | -0.000057 | 5.0 |
| FE + covariates | 4.975683 | 4.975809 | -0.000126 | 5.0 |
| IFE r=2 | 3.010223 | 3.013046 | -0.002822 | 3.0 |
| IFE r=2 + covariates | 2.993155 | 2.996099 | -0.002944 | 3.0 |
| MC lambda=0.01 | 3.176671 | 3.176721 | -0.000050 | 3.0 |

**Point estimates agree to 4-6 decimal places** for FE and MC. IFE differences (~0.003) are due to different SVD implementations converging to slightly different factor rotations — both are equally valid and equally close to the true ATT.

### Standard Errors (500 bootstrap replications)

| Scenario | pyfect SE | R fect SE | Ratio |
|----------|-----------|-----------|-------|
| FE (500 boots) | 0.020011 | 0.021291 | 0.94 |
| IFE r=2 (500 boots) | 0.017128 | 0.018382 | 0.93 |

**SEs agree within 7%** despite using different random draws and different SVD implementations. Per-period SE ratios range from 0.7 to 1.25.

### Covariate Coefficients

| Covariate | pyfect | R fect | Difference |
|-----------|--------|--------|------------|
| X1 | 0.817041 | 0.817042 | -0.000001 |
| X2 | 1.173748 | 1.173747 | +0.000000 |
| X3 | 1.857722 | 1.857723 | -0.000001 |

**Coefficients agree to 6 decimal places.**

## Performance

The key bottleneck in R fect is full SVD inside the EM loop: O(NT min(N,T)) per iteration. pyfect uses randomized truncated SVD for large matrices: O(NTr) where r << min(N,T).

| Scenario | pyfect | R fect | Speedup |
|----------|--------|--------|---------|
| FE, N=200 T=50 | 0.01s | 0.05s | 5x |
| IFE, N=200 T=50 | 0.08s | 0.06s | 0.7x |
| FE + 500 boots, N=200 T=50 | 0.54s | 5.52s | **10x** |
| IFE + 500 boots, N=200 T=50 | 3.61s | 22.83s | **6x** |
| IFE, N=1000 T=50 | 0.17s | 0.17s | 1x |

The main speedup comes from **bootstrap/CV parallelization** (`n_jobs`) and will increase further with GPU acceleration on large panels.

## Design Principles

- **Reproducible**: every random operation is seeded; same seed = identical results
- **Idempotent**: calling `fect()` twice with the same inputs produces the same output
- **GPU-optional**: toggle `device="gpu"` for CuPy acceleration (no code changes needed)
- **Parallel**: CV folds and bootstrap replications run in parallel via `n_jobs`
- **Polars-first**: native Polars support; Pandas also accepted

## References

- Liu, L., Wang, Y., & Xu, Y. (2024). A Practical Guide to Counterfactual Estimators for Causal Inference with Time-Series Cross-Sectional Data. *American Journal of Political Science*, 68(1), 160-176.
- Xu, Y. (2017). Generalized Synthetic Control Method. *Political Analysis*, 25(1), 57-76.
- Athey, S., et al. (2021). Matrix Completion Methods for Causal Panel Data Models. *JASA*, 116(536), 1716-1730.
- Bai, J. (2009). Panel Data Models with Interactive Fixed Effects. *Econometrica*, 77(4), 1229-1279.

## License

MIT
