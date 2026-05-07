# Changelog

## 0.2.0 - 2026-05-06

**Breaking change** (diagnostics API). The field-by-field migration table is
included below.

### Behavior change: TOST default now matches Liu et al. (2024)

`run_diagnostics(result, tost_threshold=0.36)` previously treated
`0.36` as an absolute equivalence bound. It now triggers Liu's
scale-aware default `0.36 * sqrt(result.sigma2_fect)`, where
`sigma2_fect` is the residual variance of the additive fixed-effect
baseline implied by the fit's `force` option. With the default
`force="two-way"`, this is the two-way fixed-effects baseline from the
AJPS paper; pyfector uses residual degrees of freedom for the identified
FE design on observed untreated cells.

For sparse log-count outcomes fit by MC, the new default can differ
from the old one by an order of magnitude. Example (developer-month
panel, MC fit):

| Outcome | Old (abs 0.36) | New (Liu auto) |
|---|---:|---:|
| `log_pushes_own` | 0.360 | 0.347 |
| `log_pushes_other` | 0.360 | 0.203 |
| `log_prs_opened_own` | 0.360 | 0.105 |
| `log_active_days` | 0.360 | 0.075 |

To preserve the old absolute behavior, pass a literal other than
`0.36` (e.g. `tost_threshold=0.36000001`). To opt out of TOST
auto-scaling entirely, pass an explicit positive float for an
absolute bound.

### New: `result.sigma2_fect`

`FectResult` now carries `sigma2_fect`, the residual variance of the
additive-FE baseline (one extra r=0 IFE pass on the main fit, never
inside bootstrap iterations). For `method="fe"`, `sigma2_fect` equals
`sigma2` since the chosen estimator is the additive-FE pass.

### New: `pyfector.fect(..., diagnostics=...)`

```python
result = pyfector.fect(
    df, Y="y", D="d", index=("u", "t"),
    method="mc", se=True, nboots=200,
    diagnostics="full",
    diagnostics_options={"placebo_period": (-3, 0)},
)
result.diagnostics.tost.threshold      # auto-scaled by sigma2_fect
result.diagnostics.placebo.equiv_p_value
```

`diagnostics` accepts `"none"` (default), `"full"`, or a list of test
names (`["tost", "pretrend_f"]`). The full path is best-effort:
unconfigured optional tests are silently skipped. An explicit list is
strict: missing `placebo_period` for a listed `placebo` raises
`ValueError` at fit start. `diagnostics="full"` requires `se=True`.

### New: `result.diagnostics`

When `diagnostics="full"` (or a list) is set on `fect()`, the result
has `result.diagnostics` populated with `Diagnostics`. The container is
**slim-safe**: it holds only primitives, tuples, dicts of
primitives, and bounded numpy arrays. No references to the panel,
inference matrices, or estimator state. Survives a downstream slim of
`result.panel` and pickle round-trips.

`Diagnostics` stores populated tests in a `diag.tests` registry so new
diagnostics can be added without changing the envelope schema.
Convenience properties (`diag.tost`, `diag.pretrend_f`, etc.) are kept
for built-in tests. `DiagnosticResult` remains as a compatibility alias.

### Breaking: diagnostics output is now hierarchical

Each test owns its own sub-dataclass. Field renames:

| Old (flat) | New (hierarchical) |
|---|---|
| `diag.f_stat`, `f_pval`, `f_df1`, `f_df2` | `diag.pretrend_f.f_stat`, `.p_value`, `.df1`, `.df2` |
| `diag.equiv_f_pval`, `diag.equiv_threshold` | `diag.equiv_f.p_value`, `diag.equiv_f.f_threshold` |
| `diag.tost_pvals`, `diag.tost_periods`, `diag.tost_threshold` | `diag.tost.pvals`, `.periods`, `.threshold` |
| `diag.placebo_att`, `diag.placebo_pval` | `diag.placebo.estimate`, `diag.placebo.p_value` |
| (new) | `diag.placebo.equiv_p_value` (TOST equivalence on the placebo window) |
| `diag.carryover_att` | `diag.carryover.estimate` |
| `diag.loo_atts`, `diag.loo_periods`, `diag.loo_max_change` | `diag.loo.atts`, `.periods`, `.max_change` |

`run_diagnostics` also gains:
- `tost_threshold=None` raises `ValueError` (was silently propagated).
- `tost_threshold` non-positive or non-finite raises `ValueError`.
- `alpha` parameter (default 0.05) controls `TostResult.all_pass` cutoff.
- `diag.tests` registry records populated diagnostic result objects.
- `diag.options` dict records the resolved configuration used (for
  reproducibility from a pickled diagnostics object alone).
- Fit-time diagnostics validate option values before estimation begins.
- Pre-trend F diagnostics drop bootstrap columns with missing selected
  pre-period coefficients instead of silently omitting the F-test.

### Deferred to future releases

- Carryover TOST equivalence p-value (requires plumbing `att_off_boot`
  through the inference layer).
- `result.slim()` helper to drop heavy state. Slim-safety is contracted
  on `Diagnostics` so the future helper can leave it untouched.

## 0.1.7 - 2026-05-04

- Added explicit `lambda_candidates` support for targeted matrix-completion CV grids.
- Reused the existing full-sample point estimate during bootstrap setup, avoiding duplicate point re-estimation while preserving all bootstrap resampled fits.
- Added MC CV diagnostics that warn when the selected lambda is on the candidate-grid boundary.
- Expanded release wheel smoke checks to Python 3.10, 3.11, 3.12, and 3.13.

## 0.1.6 - 2026-05-04

- Fixed a large-panel cross-validation fold-construction bottleneck by tracking masked cells incrementally instead of repeatedly summing the full mask.
- Added regression coverage for exact CV-fold mask preservation and for avoiding full-mask rescans.
- Improved optional GPU backend robustness by using host-side RNG for index sampling, avoiding unnecessary sort kernels after Gram eigendecomposition, and serializing Python-level jobs on GPU runs.
- Moved CV scoring reductions to host arrays for more stable optional GPU execution.
- Updated the PyPI publish workflow to disable attestations unless the PyPI project is configured for them.

## 0.1.5 - 2026-05-01

- Made cross-validation paper-faithful by default: `cv_rule="min"` now selects the strict lowest-score candidate.
- Added `cv_rule="onepct"` as the explicit 1% slack heuristic for conservative model-complexity sensitivity checks.
- Removed obsolete CV-rule compatibility spelling so rule names reflect the implemented criteria exactly.
- Clarified and enforced missing-outcome semantics: missing raw `Y` cells are masked by `I`, all-missing units are dropped, and missing treated outcomes do not enter ATT or inference aggregation.
- Expanded README, reference docs, pdoc API output, and regression tests for CV-rule and sparse-panel behavior.

## 0.1.4 - 2026-05-01

- Fixed bootstrap inference when treated outcomes are missing. Overall bootstrap ATT, confidence intervals, and p-values now respect the observed-cell mask.
- Improved matrix-completion performance on rectangular panels by using a Gram-matrix SVD shortcut in the soft-thresholding step.
- Changed the default `n_jobs` to `-1`, using all available CPUs for CV and bootstrap unless specified.
- Made `cv_nobs` create within-unit block masks during cross-validation.
- Changed unsupported `group`, `Z`, and `Q` arguments to raise `NotImplementedError` instead of being ignored.
- Added regression tests for missing treated outcomes, rectangular MC panels, CV block masking, and unsupported argument handling.

## 0.1.3 - 2026-04-09

- Improved bootstrap runtime with relaxed replicate tolerance, warm-started replicates, and EM buffer reuse.
- Kept R comparison tests available for local validation while removing the slow R comparison job from CI.
