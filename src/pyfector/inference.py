"""
Non-parametric inference: bootstrap, jackknife, and permutation.

This module builds confidence intervals and standard errors around the
point estimates returned by :func:`pyfector.fect`.  All three routines
resample *units* (not cells), so they naturally respect within-unit
dependence:

* :func:`bootstrap` — draws ``nboots`` bootstrap samples of unit indices
  with replacement, re-estimates on each sample, and returns pivotal
  confidence intervals, bootstrap SEs, and two-sided p-values.  Treated,
  control, and reversal units are resampled separately so the treated
  share stays fixed across replicates (stratified bootstrap); when
  ``cluster`` is supplied we fall back to a cluster bootstrap.
* :func:`jackknife` — delete-one-unit jackknife with the standard
  ``(N-1)/N`` variance correction and t-distribution CIs.
* :func:`permutation_test` — shuffles treatment assignment and compares
  the observed ATT to the null distribution of permuted ATTs.

Implementation notes
--------------------
* Replications are parallelised via joblib.  The default backend is
  ``processes`` to avoid GIL contention on long-running fits; for
  repeated small fits ``prefer="threads"`` can be cheaper because
  numpy releases the GIL inside BLAS.
* Every replicate receives a deterministic seed derived from the
  master ``seed`` so results are reproducible.
* The dynamic-ATT grid is pinned to the point-estimate's ``time_on``
  so replicate outputs can be stacked into a ``(n_periods, nboots)``
  matrix even when a resampled draw happens to miss a relative-time
  value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from joblib import Parallel, delayed

from .backend import get_backend, to_numpy, to_device, make_rng


@dataclass
class InferenceResult:
    """Inference results container."""
    att_avg: float                          # point estimate of overall ATT
    att_avg_se: float                       # standard error
    att_avg_ci: tuple[float, float]         # confidence interval
    att_avg_pval: float                     # p-value

    att_on: np.ndarray                      # (n_periods,) dynamic ATT
    att_on_se: np.ndarray                   # (n_periods,) SE
    att_on_ci_lower: np.ndarray             # (n_periods,)
    att_on_ci_upper: np.ndarray             # (n_periods,)
    att_on_pval: np.ndarray                 # (n_periods,)
    time_on: np.ndarray                     # (n_periods,) relative time indices

    # Bootstrap distributions (for advanced use)
    att_avg_boot: np.ndarray | None = None  # (nboots,)
    att_on_boot: np.ndarray | None = None   # (n_periods, nboots)


def bootstrap(
    estimate_fn,
    Y: np.ndarray,
    D: np.ndarray,
    I: np.ndarray,
    T_on: np.ndarray,
    unit_type: np.ndarray,
    nboots: int = 200,
    alpha: float = 0.05,
    n_jobs: int = 1,
    seed: int | None = None,
    cluster: np.ndarray | None = None,
) -> InferenceResult:
    """Non-parametric cluster bootstrap.

    Parameters
    ----------
    estimate_fn : callable
        Function that takes (unit_indices,) and returns:
        - eff: (T, N_sub) treatment effects
        - D_sub: (T, N_sub) treatment indicators
        - T_on_sub: (T, N_sub) relative time
        - I_sub: (T, N_sub) observation indicators
    unit_type : (N,) array
        1=control, 2=treated, 3=reversal
    cluster : (N,) array, optional
        Cluster assignments. If None, each unit is its own cluster.
    """
    rng = np.random.default_rng(seed)
    N = Y.shape[1]

    # Identify treated and control units
    treated_idx = np.where(unit_type == 2)[0]
    control_idx = np.where(unit_type == 1)[0]
    reversal_idx = np.where(unit_type == 3)[0]

    N_tr = len(treated_idx)
    N_co = len(control_idx)
    N_rev = len(reversal_idx)

    # Point estimate
    eff_point, D_point, T_on_point, I_point = _unpack_estimate(
        estimate_fn(np.arange(N)), I
    )
    att_avg_point, att_on_point, time_on = _compute_att(eff_point, D_point, T_on_point, I_point)

    n_periods = len(time_on)

    def _one_boot(boot_seed):
        boot_rng = np.random.default_rng(boot_seed)

        if cluster is not None:
            # Cluster bootstrap
            unique_clusters = np.unique(cluster)
            sampled_clusters = boot_rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
            unit_idx = np.concatenate([np.where(cluster == c)[0] for c in sampled_clusters])
        else:
            # Resample treated and control separately
            tr_sample = boot_rng.choice(treated_idx, size=N_tr, replace=True) if N_tr > 0 else np.array([], dtype=int)
            co_sample = boot_rng.choice(control_idx, size=N_co, replace=True) if N_co > 0 else np.array([], dtype=int)
            rev_sample = boot_rng.choice(reversal_idx, size=N_rev, replace=True) if N_rev > 0 else np.array([], dtype=int)
            unit_idx = np.concatenate([tr_sample, co_sample, rev_sample]).astype(int)

        try:
            eff_b, D_b, T_on_b, I_b = _unpack_estimate(
                estimate_fn(unit_idx), I[:, unit_idx]
            )
            att_avg_b, att_on_b, _ = _compute_att(eff_b, D_b, T_on_b, I_b, time_on)
            return att_avg_b, att_on_b
        except Exception:
            return None, None

    # Generate deterministic seeds for each replicate
    boot_seeds = rng.integers(0, 2**31, size=nboots)

    if n_jobs == 1:
        results = [_one_boot(s) for s in boot_seeds]
    else:
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_one_boot)(s) for s in boot_seeds
        )

    # Filter failed boots
    valid = [(avg, on) for avg, on in results if avg is not None]
    if len(valid) == 0:
        raise RuntimeError("All bootstrap replications failed")

    att_avg_boot = np.array([v[0] for v in valid])
    att_on_boot = np.column_stack([v[1] for v in valid])  # (n_periods, nboots_valid)

    # Pivotal confidence intervals: 2*theta - quantile(1-alpha/2) to 2*theta - quantile(alpha/2)
    att_avg_se = float(np.std(att_avg_boot, ddof=1))
    att_avg_ci = (
        2 * att_avg_point - float(np.quantile(att_avg_boot, 1 - alpha / 2)),
        2 * att_avg_point - float(np.quantile(att_avg_boot, alpha / 2)),
    )
    att_avg_pval = min(
        2 * np.mean(att_avg_boot >= 0),
        2 * np.mean(att_avg_boot <= 0),
        1.0,
    )

    att_on_se = np.nanstd(att_on_boot, axis=1, ddof=1)
    att_on_ci_lower = 2 * att_on_point - np.nanquantile(att_on_boot, 1 - alpha / 2, axis=1)
    att_on_ci_upper = 2 * att_on_point - np.nanquantile(att_on_boot, alpha / 2, axis=1)
    valid_on = np.isfinite(att_on_boot)
    valid_counts = valid_on.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        boot_ge_0 = np.sum((att_on_boot >= 0) & valid_on, axis=1) / valid_counts
        boot_le_0 = np.sum((att_on_boot <= 0) & valid_on, axis=1) / valid_counts
    att_on_pval = np.minimum(2 * boot_ge_0, np.minimum(2 * boot_le_0, 1.0))
    att_on_pval[valid_counts == 0] = np.nan

    return InferenceResult(
        att_avg=att_avg_point,
        att_avg_se=att_avg_se,
        att_avg_ci=att_avg_ci,
        att_avg_pval=att_avg_pval,
        att_on=att_on_point,
        att_on_se=att_on_se,
        att_on_ci_lower=att_on_ci_lower,
        att_on_ci_upper=att_on_ci_upper,
        att_on_pval=att_on_pval,
        time_on=time_on,
        att_avg_boot=att_avg_boot,
        att_on_boot=att_on_boot,
    )


def jackknife(
    estimate_fn,
    Y: np.ndarray,
    D: np.ndarray,
    I: np.ndarray,
    T_on: np.ndarray,
    unit_type: np.ndarray,
    alpha: float = 0.05,
    n_jobs: int = 1,
) -> InferenceResult:
    """Delete-one-unit jackknife inference.

    Drops one unit at a time, re-estimates, computes jackknife SE
    with the standard (N-1)/N correction.
    """
    from scipy import stats

    N = Y.shape[1]

    # Point estimate
    eff_point, D_point, T_on_point, I_point = _unpack_estimate(
        estimate_fn(np.arange(N)), I
    )
    att_avg_point, att_on_point, time_on = _compute_att(eff_point, D_point, T_on_point, I_point)
    n_periods = len(time_on)

    def _one_jack(j):
        idx = np.concatenate([np.arange(j), np.arange(j + 1, N)])
        try:
            eff_j, D_j, T_on_j, I_j = _unpack_estimate(
                estimate_fn(idx), I[:, idx]
            )
            att_avg_j, att_on_j, _ = _compute_att(eff_j, D_j, T_on_j, I_j, time_on)
            return att_avg_j, att_on_j
        except Exception:
            return None, None

    if n_jobs == 1:
        results = [_one_jack(j) for j in range(N)]
    else:
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_one_jack)(j) for j in range(N)
        )

    valid = [(avg, on) for avg, on in results if avg is not None]
    n_valid = len(valid)
    if n_valid < 2:
        raise RuntimeError("Too few valid jackknife replications")

    att_avg_jack = np.array([v[0] for v in valid])
    att_on_jack = np.column_stack([v[1] for v in valid])

    # Jackknife variance: (N-1)/N * sum((theta_j - theta_bar)^2)
    att_avg_se = float(np.sqrt((n_valid - 1) / n_valid * np.sum((att_avg_jack - np.mean(att_avg_jack)) ** 2)))
    att_on_se = np.sqrt(
        (n_valid - 1) / n_valid * np.sum((att_on_jack - att_on_jack.mean(axis=1, keepdims=True)) ** 2, axis=1)
    )

    # t-distribution CI
    t_crit = stats.t.ppf(1 - alpha / 2, df=n_valid - 1)
    att_avg_ci = (att_avg_point - t_crit * att_avg_se, att_avg_point + t_crit * att_avg_se)
    att_on_ci_lower = att_on_point - t_crit * att_on_se
    att_on_ci_upper = att_on_point + t_crit * att_on_se

    # p-values from t-distribution
    t_avg = att_avg_point / max(att_avg_se, 1e-10)
    att_avg_pval = float(2 * stats.t.sf(abs(t_avg), df=n_valid - 1))

    t_on = att_on_point / np.maximum(att_on_se, 1e-10)
    att_on_pval = 2 * stats.t.sf(np.abs(t_on), df=n_valid - 1)

    return InferenceResult(
        att_avg=att_avg_point,
        att_avg_se=att_avg_se,
        att_avg_ci=att_avg_ci,
        att_avg_pval=att_avg_pval,
        att_on=att_on_point,
        att_on_se=att_on_se,
        att_on_ci_lower=att_on_ci_lower,
        att_on_ci_upper=att_on_ci_upper,
        att_on_pval=att_on_pval,
        time_on=time_on,
        att_avg_boot=att_avg_jack,
        att_on_boot=att_on_jack,
    )


def permutation_test(
    estimate_fn,
    att_avg_observed: float,
    N: int,
    T: int,
    n_perms: int = 200,
    block_size: int = 1,
    seed: int | None = None,
    n_jobs: int = 1,
) -> tuple[float, np.ndarray]:
    """Permutation test: shuffle treatment assignment, compare ATTs.

    Returns (p_value, permuted_atts).
    """
    rng = np.random.default_rng(seed)
    perm_seeds = rng.integers(0, 2**31, size=n_perms)

    def _one_perm(perm_seed):
        try:
            att_perm = estimate_fn(perm_seed)
            return att_perm
        except Exception:
            return None

    if n_jobs == 1:
        results = [_one_perm(s) for s in perm_seeds]
    else:
        results = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_one_perm)(s) for s in perm_seeds
        )

    valid = [r for r in results if r is not None]
    if len(valid) == 0:
        return 1.0, np.array([])

    permuted_atts = np.array(valid)
    p_value = float(np.mean(np.abs(permuted_atts) >= abs(att_avg_observed)))
    return p_value, permuted_atts


# ---------------------------------------------------------------------------
# ATT computation helpers
# ---------------------------------------------------------------------------

def _unpack_estimate(result, fallback_I: np.ndarray):
    """Accept old 3-tuple estimate closures and new 4-tuple closures."""
    if len(result) == 3:
        eff, D, T_on = result
        return eff, D, T_on, fallback_I
    if len(result) == 4:
        return result
    raise ValueError("estimate_fn must return (eff, D, T_on) or (eff, D, T_on, I)")


def _compute_att(
    eff: np.ndarray,     # (T, N) effects
    D: np.ndarray,       # (T, N) treatment indicator
    T_on: np.ndarray,    # (T, N) relative time
    I: np.ndarray | None = None,  # (T, N) observation indicator
    time_on: np.ndarray | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute overall and dynamic ATT from the effect matrix.

    Uses ``np.bincount`` for the event-time grouping so the cost is
    linear in the number of observed cells.  When ``time_on`` is
    supplied the caller's grid is preserved exactly so bootstrap
    replicates can be stacked into a ``(n_periods, nboots)`` matrix even
    if some replicates happen to not observe every event time.
    """
    if I is not None and np.shape(I) != np.shape(D) and time_on is None:
        time_on = np.asarray(I)
        I = None

    if I is None:
        observed_mask = np.ones_like(D, dtype=bool)
    else:
        observed_mask = I > 0

    treated_mask = (D > 0) & observed_mask
    n_treated = int(np.sum(treated_mask))
    att_avg = 0.0 if n_treated == 0 else float(np.sum(eff[treated_mask]) / n_treated)

    ever_treated = np.any(D > 0, axis=0)                            # (N,)
    all_mask = observed_mask & np.broadcast_to(ever_treated[np.newaxis, :], T_on.shape)
    T_on_flat = T_on[all_mask]
    eff_flat = eff[all_mask]
    valid = ~np.isnan(T_on_flat)
    T_on_flat = T_on_flat[valid]
    eff_flat = eff_flat[valid]

    if T_on_flat.size == 0:
        if time_on is None:
            return att_avg, np.array([]), np.array([])
        return att_avg, np.full(len(time_on), np.nan), time_on

    T_on_int = T_on_flat.astype(np.int64)

    if time_on is None:
        offset = int(T_on_int.min())
        minlength = int(T_on_int.max() - offset + 1)
        idx = T_on_int - offset
        sums = np.bincount(idx, weights=eff_flat, minlength=minlength)
        counts = np.bincount(idx, minlength=minlength)
        keep = counts > 0
        time_on = (offset + np.arange(minlength))[keep].astype(np.float64)
        att_on = sums[keep] / counts[keep]
        return att_avg, att_on, time_on

    # Caller supplied a grid: produce att_on aligned to time_on so that
    # bootstrap replicates can be stacked with a fixed shape.
    time_on_int = np.asarray(time_on, dtype=np.int64)
    grid_offset = int(time_on_int.min())
    grid_len = int(time_on_int.max() - grid_offset + 1)
    # Bin flat values that fall inside the grid range.
    in_range = (T_on_int >= grid_offset) & (T_on_int < grid_offset + grid_len)
    T_clip = T_on_int[in_range] - grid_offset
    e_clip = eff_flat[in_range]
    sums = np.bincount(T_clip, weights=e_clip, minlength=grid_len)
    counts = np.bincount(T_clip, minlength=grid_len)
    gather = time_on_int - grid_offset                              # (n_periods,)
    att_on = np.full(len(time_on), np.nan)
    counts_at_grid = counts[gather]
    valid_grid = counts_at_grid > 0
    att_on[valid_grid] = sums[gather[valid_grid]] / counts_at_grid[valid_grid]
    return att_avg, att_on, time_on
