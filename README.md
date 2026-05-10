# pyfector

> **Alpha (v0.2.x).** Results have been checked against R `fect` on synthetic data, but edge cases may remain. APIs may change. Please verify critical results independently. Issues and pull requests are welcome at [GitHub](https://github.com/AlanHuang99/pyfector/issues).

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

## What's New in 0.2.1

- Corrected `diag.placebo` to use the standardized Liu-Wang-Xu/R `fect` holdout/refit placebo test instead of averaging existing pre-period event-study coefficients.
- Added `PlaceboResult.n_obs` and `PlaceboResult.n_boot` so downstream caches can audit the withheld-cell count and valid placebo bootstrap refits.
- Matched R `fect`'s default normal approximation for the placebo point-null p-value and normal TOST formula for placebo equivalence.

## What's New in 0.2.0

- Added `sigma2_fect`, the additive fixed-effect baseline residual variance used for Liu et al. (2024)'s TOST bound.
- Changed the default `tost_threshold=0.36` to mean `0.36 * sqrt(sigma2_fect)`, matching the published specification.
- Added fit-time diagnostics via `diagnostics="full"` or an explicit list, with slim-safe cached results.
- Replaced the old flat diagnostics output with a future-proof `Diagnostics` registry while keeping built-in convenience properties such as `diag.tost`.
- Fixed diagnostic F-tests with partially missing bootstrap draws and tightened diagnostics option validation before expensive fits start.

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

Cross-validation masks a fraction `cv_prop` of control cells and scores candidates by prediction error (`mspe`, `gmspe`, or `mad`). By default, `cv_rule="min"` follows the papers and selects the strict minimum-score candidate. Use `cv_rule="onepct"` to select the simpler candidate within 1% of the best score (lower `r` for IFE, higher `lam` for MC).

## Design choices

pyfector is paper-faithful first. When the published counterfactual-estimator papers and historical R package behavior differ, the default follows the paper and compatibility behavior is exposed through explicit options.

### Cross-validation rule

The papers describe choosing `r` or `lam` by a prespecified validation metric such as MSPE. Therefore `cv_rule="min"` is the default. The `onepct` rule is useful as a sensitivity check because it avoids adding flexibility for tiny validation gains, but it is a heuristic rather than the paper's core selection criterion.

### Missing outcomes and the estimand

pyfector distinguishes two kinds of missingness:

| State | How pyfector treats it |
|-------|------------------------|
| Treated cells, where `D == 1` | Treated status makes untreated potential outcomes unobserved; pyfector imputes `Y_ct = Y(0)` for these cells. |
| Raw missing outcome, where input `Y` is null | The outcome is not observed. The cell is excluded from fitting and from ATT aggregation. |

Internally, missing `Y` cells are stored as `0.0` only as a dense-array placeholder. The observation mask `I` carries missingness, and estimators use `I`/`II` to decide which cells are observed. A missing treated outcome may receive a model counterfactual, but it does not enter `att_avg`, `att_on`, or bootstrap/jackknife ATT calculations because the treated potential outcome `Y(1)` was not observed.

For unit filtering, the default estimand is the ATT over the matched panel's observed treated outcome cells, using all available observed untreated outcomes to fit the counterfactual model. Controls are not required to satisfy `min_T0` by default, so sparse controls can still help estimate the response surface. Units with no observed outcome at all are always dropped. Set `min_T0_strict=True` to require controls to satisfy `min_T0` as well; this is the conservative R-package-style sparse-panel behavior and changes the estimand toward always-active units.

---

## Parameters

### `fect()`

```python
fect(
    data, Y, D, index,
    *, X=None, W=None, group=None, method="ife", force="two-way",
    r=0, lam=None, nlambda=10, lambda_candidates=None,
    CV=True, k=10, cv_prop=0.1,
    cv_nobs=3, cv_treat=True, cv_donut=0, criterion="mspe",
    cv_rule="min",
    se=False, vartype="bootstrap", nboots=200, alpha=0.05,
    tol=1e-7, max_iter=5000, min_T0=1, min_T0_strict=False,
    max_missing=1.0, normalize=False,
    Z=None, Q=None,
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
| `group` | `str` | `None` | Not implemented; raises `NotImplementedError` when supplied |
| `method` | `str` | `"ife"` | `"fe"`, `"ife"`, `"mc"`, `"cfe"`, `"both"` |
| `force` | `str` | `"two-way"` | `"none"`, `"unit"`, `"time"`, `"two-way"` |
| `r` | `int` or `(int, int)` | `0` | Number of factors; tuple triggers CV over range |
| `lam` | `float` | `None` | MC nuclear-norm penalty; `None` selects by CV |
| `nlambda` | `int` | `10` | Grid size for MC CV |
| `lambda_candidates` | array-like | `None` | Explicit non-negative lambda candidates for MC CV; overrides `nlambda` |
| `CV` | `bool` | `True` | Cross-validate over `r` or `lam` |
| `k` | `int` | `10` | CV folds |
| `cv_prop` | `float` | `0.1` | Fraction of control cells masked per CV fold |
| `cv_nobs` | `int` | `3` | Consecutive within-unit observations to mask together during CV |
| `cv_treat` | `bool` | `True` | If `True`, CV masks eligible pre-treatment cells from ever-treated units; if `False`, it masks all observed control cells |
| `cv_donut` | `int` | `0` | Exclude this many periods around treatment onset from CV evaluation |
| `criterion` | `str` | `"mspe"` | CV loss: `"mspe"`, `"gmspe"`, `"mad"` |
| `cv_rule` | `str` | `"min"` | `"min"` selects the strict minimum validation score; `"onepct"` selects the simpler candidate within 1% of the best score |
| `se` | `bool` | `False` | Compute standard errors |
| `vartype` | `str` | `"bootstrap"` | `"bootstrap"` or `"jackknife"` |
| `nboots` | `int` | `200` | Bootstrap replications |
| `alpha` | `float` | `0.05` | Significance level for intervals and tests |
| `tol` | `float` | `1e-7` | EM convergence tolerance |
| `max_iter` | `int` | `5000` | Max EM iterations |
| `min_T0` | `int` | `1` | Minimum untreated/pre-treatment observed periods; default applies to treated/reversal units |
| `min_T0_strict` | `bool` | `False` | If `True`, apply `min_T0` to controls too, matching conservative R `fect` sparse-panel filtering |
| `max_missing` | `float` | `1.0` | Maximum missing-outcome fraction per unit; units with zero observed outcomes are always dropped |
| `normalize` | `bool` | `False` | Normalize outcome by standard deviation |
| `Z`, `Q` | `list[str]` | `None` | Not implemented; raises `NotImplementedError` when supplied |
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
| `att_avg` | `float` | Overall ATT over observed treated outcome cells |
| `att_avg_unit` | `float` | Unit-averaged ATT over units with observed treated outcome cells |
| `att_on` | `ndarray` | Dynamic ATT by relative time, using observed cells only |
| `time_on` | `ndarray` | Relative time indices |
| `count_on` | `ndarray` | Observation count per relative time |
| `beta` | `ndarray` | Covariate coefficients |
| `Y_ct` | `ndarray` | `(T, N)` counterfactual outcome matrix |
| `eff` | `ndarray` | `(T, N)` raw effect matrix `Y - Y_ct`; use `panel.I` to identify observed cells |
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
# Default: omit tost_threshold to use Liu et al. (2024)'s 0.36 * sqrt(sigma2_fect)
diag = result.diagnose(
    placebo_period=(-5, -1),
    loo=True,
)
diag.summary()

# Or run all diagnostics at fit time and store on the result object
# (slim-safe: result.diagnostics survives a downstream slim of result.panel)
result = pyfector.fect(
    df, Y="y", D="d", index=("unit", "time"),
    method="mc", se=True, nboots=200,
    diagnostics="full",
    diagnostics_options={"placebo_period": (-3, 0)},
)
result.diagnostics.tost.threshold
result.diagnostics.tests  # registry for future diagnostic outputs
```

`placebo_period` runs the standardized placebo test from Liu, Wang, and
Xu (2024) and R `fect(placeboTest=TRUE)`: pyfector removes the selected
observed pre-treatment cells from the fitting mask, refits with the
already-selected rank/lambda configuration, and computes the placebo ATT
from the withheld cells. For example, `placebo_period=(-3, 0)` withholds
relative periods `-3`, `-2`, and `-1`; period `0` is the treatment-onset
period and is not treated as pre-treatment. The point-null
`diag.placebo.p_value` follows R `fect`'s default normal approximation
using the placebo bootstrap SE; `diag.placebo.equiv_p_value` is the
corresponding normal TOST p-value.

The `tost_threshold` argument has three modes:

- Omitted or `0.36`: uses Liu's auto-scaled bound `0.36 * sqrt(result.sigma2_fect)`
- Any other positive float: absolute outcome-scale bound (e.g. `tost_threshold=0.1`)
- `None`: invalid; raises `ValueError`

`sigma2_fect` is computed from the additive fixed-effect baseline using
the `force` structure requested for the fit. With the default
`force="two-way"`, its denominator uses residual degrees of freedom for
the identified unit/time FE design on the observed untreated cells.

| Test | Outputs |
|------|---------|
| Pre-trend F-test | `diag.pretrend_f.f_stat`, `.p_value`, `.df1`, `.df2` |
| Equivalence F-test | `diag.equiv_f.p_value`, `.f_threshold` |
| TOST | `diag.tost.pvals`, `.threshold`, `.threshold_source`, `.max_pval`, `.all_pass` |
| Placebo | `diag.placebo.estimate`, `.se`, `.p_value`, `.equiv_p_value`, `.n_obs`, `.n_boot` |
| Carryover | `diag.carryover.estimate` |
| Leave-one-out | `diag.loo.atts`, `.max_change` |
| Resolved options | `diag.options` (records what was actually used) |

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
