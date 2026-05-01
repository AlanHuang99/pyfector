# pyfector

> **Alpha (v0.1.x).** Results have been checked against R `fect` on synthetic data, but edge cases may remain. APIs may change. Please verify critical results independently. Issues and pull requests are welcome at [GitHub](https://github.com/AlanHuang99/pyfector/issues).

Counterfactual estimators for panel data in Python. Port of the R [fect](https://github.com/xuyiqing/fect) package (Liu, Wang & Xu, 2024, *AJPS*), built on numpy, scipy, polars, and joblib, with optional CuPy GPU support.

## Installation

```bash
pip install pyfector
```

**From source:**
```bash
git clone https://github.com/AlanHuang99/pyfector.git
cd pyfector
pip install -e ".[dev]"
```

Optional extras: `pip install pyfector[gpu]` (CuPy), `pip install pyfector[pandas]`.

## What's New in 0.1.4

- Fixed bootstrap inference for panels where treated observations can have missing outcomes. Overall bootstrap ATT, confidence intervals, and p-values now use the same observed-cell mask as point estimation.
- Improved matrix-completion performance on rectangular panels with a Gram-matrix SVD shortcut. On the GHA benchmark panel, the full MC pipeline with 500 bootstraps completed in 10.38 seconds on the test machine.
- Changed the default `n_jobs` to `-1` so CV and bootstrap use all available CPUs unless a worker count is supplied.
- Made `cv_nobs` affect CV masking through within-unit block masks.
- Unsupported `group`, `Z`, and `Q` arguments now raise `NotImplementedError` instead of being silently ignored.

## Quick Start

```python
import polars as pl
import pyfector

data = pl.read_parquet("panel_data.parquet")

result = pyfector.fect(
    data=data,
    Y="outcome",
    D="treatment",
    index=("unit_id", "year"),
    X=["gdp", "population"],
    method="ife",
    r=(0, 5),
    se=True,
    nboots=500,
    seed=42,
)

result.summary()
result.plot(kind="gap")
result.diagnose()
```

## Estimators

### `method="fe"` -- Two-way fixed effects

Unit and time fixed effects on the control sample, counterfactuals imputed for the treated sample. Assumes parallel trends.

### `method="ife"` -- Interactive fixed effects (Bai 2009, Xu 2017)

Adds latent factors with unit loadings on top of fixed effects. Number of factors `r` can be fixed or chosen by cross-validation over `r=(r_min, r_max)`.

### `method="mc"` -- Matrix completion (Athey et al. 2021)

Nuclear-norm-penalized matrix completion of the counterfactual matrix. Penalty `lam` can be fixed or selected by cross-validation.

### `method="cfe"` -- Complex fixed effects

Experimental complex fixed effects. Public `Z`/`Q` interaction arguments are not implemented yet and raise `NotImplementedError` instead of being silently ignored.

Cross-validation masks a fraction `cv_prop` of control cells and picks `r` (IFE) or `lam` (MC) by prediction error (`mspe`, `gmspe`, or `mad`).

---

## Parameters

### `fect()`

```python
fect(
    data, Y, D, index,
    *, X=None, W=None, method="ife", force="two-way",
    r=0, lam=None, nlambda=10, CV=True, k=10, cv_prop=0.1, criterion="mspe",
    se=False, vartype="bootstrap", nboots=200, alpha=0.05,
    tol=1e-7, max_iter=5000, min_T0=1, normalize=False,
    device="cpu", n_jobs=-1, seed=None,
) -> FectResult
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `pl.DataFrame` or `pd.DataFrame` | required | Panel data in long format |
| `Y` | `str` | required | Outcome column |
| `D` | `str` | required | Treatment indicator (0/1) |
| `index` | `(str, str)` | required | `(unit_id, time)` column names |
| `X` | `list[str]` | `None` | Time-varying covariate columns |
| `W` | `str` | `None` | Observation weight column |
| `method` | `str` | `"ife"` | `"fe"`, `"ife"`, `"mc"`, `"cfe"` |
| `force` | `str` | `"two-way"` | `"none"`, `"unit"`, `"time"`, `"two-way"` |
| `r` | `int` or `(int, int)` | `0` | Number of factors; tuple triggers CV over range |
| `lam` | `float` | `None` | MC nuclear-norm penalty; `None` selects by CV |
| `CV` | `bool` | `True` | Cross-validate over `r` or `lam` |
| `k` | `int` | `10` | CV folds |
| `cv_prop` | `float` | `0.1` | Fraction of control cells masked per CV fold |
| `criterion` | `str` | `"mspe"` | CV loss: `"mspe"`, `"gmspe"`, `"mad"` |
| `se` | `bool` | `False` | Compute standard errors |
| `vartype` | `str` | `"bootstrap"` | `"bootstrap"` or `"jackknife"` |
| `nboots` | `int` | `200` | Bootstrap replications |
| `tol` | `float` | `1e-7` | EM convergence tolerance |
| `max_iter` | `int` | `5000` | Max EM iterations |
| `min_T0` | `int` | `1` | Minimum pre-treatment periods required per unit |
| `normalize` | `bool` | `False` | Normalize outcome by standard deviation |
| `device` | `str` | `"cpu"` | `"cpu"` or `"gpu"` (requires `pyfector[gpu]`) |
| `n_jobs` | `int` | `-1` | Parallel workers for CV / bootstrap; `-1` uses all CPUs |
| `seed` | `int` | `None` | Random seed |

### Data format

Long-format panel with one row per unit per period.

| Column | Type | Description |
|--------|------|-------------|
| outcome | float | Outcome variable |
| unit id | int/str | Unit identifier |
| time | int | Time period |
| treatment | int | 0/1 treatment indicator |

---

## Output

`fect()` returns a `FectResult` with these fields:

| Field | Type | Description |
|-------|------|-------------|
| `att_avg` | `float` | Overall ATT (weighted by cell counts) |
| `att_avg_unit` | `float` | Unit-averaged ATT |
| `att_on` | `ndarray` | Dynamic ATT by relative time |
| `time_on` | `ndarray` | Relative time indices |
| `count_on` | `ndarray` | Observation count per relative time |
| `beta` | `ndarray` | Covariate coefficients |
| `Y_ct` | `ndarray` | `(T, N)` counterfactual outcome matrix |
| `eff` | `ndarray` | `(T, N)` treatment effect matrix |
| `factors` | `ndarray` | `(T, r)` estimated factors (IFE) |
| `loadings` | `ndarray` | `(N, r)` estimated loadings (IFE) |
| `sigma2` | `float` | Residual variance |
| `r_cv` | `int` | CV-selected `r` |
| `lambda_cv` | `float` | CV-selected `lam` |
| `inference` | `InferenceResult` | Bootstrap or jackknife results (if `se=True`) |

### Methods

| Method | Description |
|--------|-------------|
| `summary()` | Formatted text summary |
| `plot(kind, **kwargs)` | Matplotlib figure. Kinds: `gap`, `status`, `factors`, `counterfactual` |
| `diagnose(...)` | Diagnostic tests (see below) |

### Inference

When `se=True`, `result.inference` carries bootstrap or jackknife SEs and CIs for the overall ATT and per-period effects, plus the full bootstrap distribution (`att_avg_boot`, `att_on_boot`).

### Diagnostics

```python
diag = result.diagnose(
    f_threshold=0.5,
    tost_threshold=0.36,
    placebo_period=(-5, -1),
    loo=True,
)
diag.summary()
```

| Test | Output |
|------|--------|
| Pre-trend F-test | `f_stat`, `f_pval` |
| Equivalence F-test | `equiv_f_pval` |
| TOST | `tost_pvals` |
| Placebo | `placebo_att`, `placebo_pval` |
| Carryover | `carryover_att`, `carryover_pval` |
| Leave-one-out | `loo_max_change` |

---

## Validation against R fect

Synthetic DGP, N=200, T=50. Point estimates:

| Scenario | pyfector | R fect | Difference |
|----------|----------|--------|------------|
| FE | 4.995583 | 4.995640 | -0.000057 |
| FE + X | 4.975683 | 4.975809 | -0.000126 |
| IFE r=2 | 3.010223 | 3.013046 | -0.002822 |
| IFE r=2 + X | 2.993155 | 2.996099 | -0.002944 |
| MC lambda=0.01 | 3.176671 | 3.176721 | -0.000050 |

FE and MC agree to 4-6 decimal places. IFE differences reflect SVD rotation non-uniqueness.

Bootstrap SEs (500 reps):

| Scenario | pyfector | R fect | Ratio |
|----------|----------|--------|-------|
| FE | 0.020011 | 0.021291 | 0.94 |
| IFE r=2 | 0.017128 | 0.018382 | 0.93 |

Covariate coefficients agree to 6 decimal places. Comparison scripts are in `benchmarks/`.

---

## Performance notes

Hot paths use vectorised numpy, a randomised truncated SVD for the interactive-FE update when `r << min(T, N)`, and a Gram-matrix SVD shortcut for rectangular matrix-completion panels. Cross-validation and bootstrap replications parallelise over `n_jobs` via joblib. GPU execution is available through `device="gpu"` when CuPy is installed. See [benchmarks/](benchmarks/) for reproducible timing scripts.

---

## Testing

```bash
uv run pytest tests/                    # full suite
uv run pytest tests/test_vs_r.py -v -s  # R validation (requires R + fect)
```

---

## API reference

- **[REFERENCE.md](REFERENCE.md)** -- parameter tables, return types
- **[API docs](https://alanhuang99.github.io/pyfector/)** -- searchable HTML reference (pdoc)

---

## References

- Liu, L., Wang, Y., & Xu, Y. (2024). "A Practical Guide to Counterfactual Estimators for Causal Inference with Time-Series Cross-Sectional Data." *American Journal of Political Science*, 68(1), 160-176.
- Xu, Y. (2017). "Generalized Synthetic Control Method." *Political Analysis*, 25(1), 57-76.
- Athey, S., Bayati, M., Doudchenko, N., Imbens, G., & Khosravi, K. (2021). "Matrix Completion Methods for Causal Panel Data Models." *JASA*, 116(536), 1716-1730.
- Bai, J. (2009). "Panel Data Models with Interactive Fixed Effects." *Econometrica*, 77(4), 1229-1279.

## License

MIT
