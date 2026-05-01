# pyfector API Reference

**Version 0.1.5**

## Overview

pyfector implements counterfactual estimators for panel data, closely following the R [fect](https://github.com/xuyiqing/fect) package (Liu, Wang & Xu, 2024). Built on numpy, scipy, polars, joblib, and (optionally) CuPy.

**Pipeline:** `prepare_panel()` -> `initial_fit()` -> CV (optional) -> `estimate_*()` -> effects -> inference -> `FectResult`

---

## Design policy

pyfector defaults to the published counterfactual-estimator papers when paper language and historical R-package behavior diverge.

### Cross-validation

The paper-faithful default is `cv_rule="min"`: select the candidate with the lowest validation score under `criterion`. `cv_rule="onepct"` is available as an explicit slack heuristic that selects the simpler candidate within 1% of the best score.

### Missing outcomes

Raw missing outcomes are distinct from counterfactual missingness induced by treatment. Observed untreated cells fit the response surface. Observed treated cells contribute effects as `Y - Y_ct`. Raw missing `Y` cells are excluded from fitting and from ATT aggregation. Internally, dense matrices use `0.0` as a placeholder for missing `Y`, but `panel.I` is the authoritative observation mask.

The default estimand is the ATT over observed treated outcome cells in the matched panel, using all available observed untreated outcomes for counterfactual fitting. Sparse controls are retained if they have at least one observed outcome and satisfy `max_missing`; they do not need to satisfy `min_T0` unless `min_T0_strict=True`. Units with no observed outcome are always dropped.

---

## `fect()`

Main entry point. Returns a `FectResult`.

```python
fect(
    data, Y, D, index,
    *, X=None, W=None, group=None,
    method="ife", force="two-way",
    r=0, lam=None, nlambda=10,
    CV=True, k=10, cv_prop=0.1, cv_nobs=3, cv_treat=True, cv_donut=0,
    criterion="mspe", cv_rule="min",
    se=False, vartype="bootstrap", nboots=200, alpha=0.05,
    tol=1e-7, max_iter=5000,
    min_T0=1, min_T0_strict=False, max_missing=1.0, normalize=False,
    Z=None, Q=None,
    device="cpu", n_jobs=-1, seed=None,
) -> FectResult
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `pl.DataFrame` / `pd.DataFrame` | required | Long-format panel |
| `Y` | `str` | required | Outcome column |
| `D` | `str` | required | Treatment indicator (0/1) |
| `index` | `(str, str)` | required | `(unit_id, time)` column names |
| `X` | `list[str]` | `None` | Time-varying covariates |
| `W` | `str` | `None` | Observation weight column |
| `group` | `str` | `None` | Not implemented; raises `NotImplementedError` when supplied |
| `method` | `str` | `"ife"` | `"fe"`, `"ife"`, `"mc"`, `"cfe"`, `"both"` |
| `force` | `str` | `"two-way"` | `"none"`, `"unit"`, `"time"`, `"two-way"` |
| `r` | `int` or `(int, int)` | `0` | Number of factors; tuple triggers CV |
| `lam` | `float` | `None` | MC nuclear-norm penalty; `None` = CV |
| `nlambda` | `int` | `10` | Grid size for MC CV |
| `CV` | `bool` | `True` | Cross-validate |
| `k` | `int` | `10` | CV folds |
| `cv_prop` | `float` | `0.1` | Fraction of cells masked per CV fold |
| `cv_nobs` | `int` | `3` | Consecutive within-unit observations to mask together during CV |
| `cv_treat` | `bool` | `True` | If `True`, CV masks eligible pre-treatment cells from ever-treated units; if `False`, it masks all observed control cells |
| `cv_donut` | `int` | `0` | Exclude this many periods around treatment onset from CV evaluation |
| `criterion` | `str` | `"mspe"` | `"mspe"`, `"gmspe"`, `"mad"` |
| `cv_rule` | `str` | `"min"` | `"min"` selects the strict minimum validation score; `"onepct"` selects the simpler candidate within 1% of best score |
| `se` | `bool` | `False` | Compute standard errors |
| `vartype` | `str` | `"bootstrap"` | `"bootstrap"` or `"jackknife"` |
| `nboots` | `int` | `200` | Bootstrap replications |
| `alpha` | `float` | `0.05` | Significance level |
| `tol` | `float` | `1e-7` | EM convergence tolerance |
| `max_iter` | `int` | `5000` | Max EM iterations |
| `min_T0` | `int` | `1` | Minimum untreated/pre-treatment observed periods; default applies to treated/reversal units |
| `min_T0_strict` | `bool` | `False` | If `True`, apply `min_T0` to controls too, matching conservative R `fect` sparse-panel filtering |
| `max_missing` | `float` | `1.0` | Max fraction missing per unit; units with zero observed outcomes are always dropped |
| `normalize` | `bool` | `False` | Normalize outcome by its standard deviation |
| `Z`, `Q` | `list[str]` | `None` | Not implemented; raises `NotImplementedError` when supplied |
| `device` | `str` | `"cpu"` | `"cpu"` or `"gpu"` (requires CuPy) |
| `n_jobs` | `int` | `-1` | Parallel workers; `-1` uses all CPUs |
| `seed` | `int` | `None` | Random seed |

### Methods

| Method | Description |
|--------|-------------|
| `"fe"` | Two-way fixed effects (Liu, Wang & Xu 2024) |
| `"ife"` | Interactive fixed effects (Bai 2009, Xu 2017) |
| `"mc"` | Nuclear-norm matrix completion (Athey et al. 2021) |
| `"cfe"` | Complex fixed effects with unit/time interactions |

---

## `FectResult`

Returned by `fect()`.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `method` | `str` | Estimation method used |
| `r_cv` | `int \| None` | Selected number of factors |
| `lambda_cv` | `float \| None` | Selected MC penalty |
| `att_avg` | `float` | Overall ATT over observed treated outcome cells |
| `att_avg_unit` | `float` | Unit-averaged ATT over units with observed treated outcome cells |
| `att_on` | `np.ndarray` | Dynamic ATT by relative time, using observed cells only |
| `time_on` | `np.ndarray` | Relative time indices |
| `count_on` | `np.ndarray` | Observations per relative time |
| `att_off` | `np.ndarray \| None` | Exit-effect ATT (treatment reversal) |
| `time_off` | `np.ndarray \| None` | Exit-effect time indices |
| `beta` | `np.ndarray \| None` | Covariate coefficients |
| `covariate_names` | `list[str]` | Names aligned with `beta` |
| `mu` | `float` | Intercept |
| `alpha` | `np.ndarray \| None` | Unit fixed effects |
| `xi` | `np.ndarray \| None` | Time fixed effects |
| `factors` | `np.ndarray \| None` | `(T, r)` factors (IFE) |
| `loadings` | `np.ndarray \| None` | `(N, r)` loadings (IFE) |
| `Y_ct` | `np.ndarray` | `(T, N)` counterfactual outcome matrix |
| `eff` | `np.ndarray` | `(T, N)` raw effect matrix `Y - Y_ct`; use `panel.I` to identify observed cells |
| `residuals` | `np.ndarray` | `(T, N)` residual matrix on observed fitting cells |
| `sigma2` | `float` | Residual variance |
| `IC`, `PC` | `float` | Information / Panel criterion |
| `rmse` | `float` | Root mean squared error |
| `niter` | `int` | EM iterations used |
| `converged` | `bool` | Whether EM converged |
| `inference` | `InferenceResult \| None` | Bootstrap / jackknife results |
| `cv_result` | `CVResult \| None` | CV diagnostics |
| `panel` | `PanelData` | Processed panel metadata |
| `seed` | `int \| None` | Seed used |

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `summary()` | `str` | Formatted text summary |
| `plot(kind, **kwargs)` | `Figure` | Matplotlib figure |
| `diagnose(**kwargs)` | `DiagnosticResult` | Diagnostic tests |

`plot()` kinds: `"gap"`, `"status"`, `"factors"`, `"counterfactual"`.

---

## `InferenceResult`

Populated when `se=True`.

| Attribute | Type | Description |
|-----------|------|-------------|
| `att_avg_se` | `float` | SE of overall ATT |
| `att_avg_ci` | `(float, float)` | `1 - alpha` CI |
| `att_avg_pval` | `float` | p-value (H0: ATT=0) |
| `att_on_se` | `np.ndarray` | Per-period SEs |
| `att_on_ci_lower` | `np.ndarray` | Per-period CI lower |
| `att_on_ci_upper` | `np.ndarray` | Per-period CI upper |
| `att_on_pval` | `np.ndarray` | Per-period p-values |
| `att_avg_boot` | `np.ndarray` | Bootstrap distribution of ATT |
| `att_on_boot` | `np.ndarray` | Bootstrap distribution of dynamic ATT |

---

## `DiagnosticResult`

Returned by `result.diagnose()`.

```python
result.diagnose(
    f_threshold=0.5,
    tost_threshold=0.36,
    placebo_period=(-5, -1),
    loo=True,
)
```

| Test | Outputs |
|------|---------|
| Pre-trend F-test | `f_stat`, `f_pval`, `f_df` |
| Equivalence F-test | `equiv_f_pval` |
| TOST equivalence | `tost_pvals` |
| Placebo | `placebo_att`, `placebo_pval` |
| Carryover | `carryover_att`, `carryover_pval` |
| Leave-one-out | `loo_max_change` |

---

## Internal modules

| Module | Purpose |
|--------|---------|
| `fect.py` | Public `fect()` entry point and `FectResult` |
| `panel.py` | DataFrame -> `PanelData`, initial fit |
| `estimators.py` | IFE / MC / CFE EM loops |
| `cv.py` | Cross-validation for `r` and `lam` |
| `inference.py` | Bootstrap and jackknife |
| `diagnostics.py` | F-test, TOST, placebo, carryover, LOO |
| `plotting.py` | Matplotlib plots |
| `linalg.py` | Randomized SVD, projections, solves |
| `backend.py` | CPU/GPU backend dispatch (numpy/cupy) |
