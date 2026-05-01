"""
Cross-validation for hyperparameter selection.

This module provides two entry points used by :func:`pyfector.fect`:

* :func:`cv_ife` picks the number of latent factors ``r`` for the IFE
  estimator by grid search over ``r_range = (r_min, r_max)``.
* :func:`cv_mc` picks the nuclear-norm penalty ``lam`` for the
  matrix-completion estimator over an automatically generated
  log-spaced grid (or a user-supplied one).

CV scheme
---------
* Hold-out cells are sampled *only* from control observations
  (``II``).  When ``cv_treat=True`` (the default) we further restrict
  the hold-out cells to pre-treatment periods of ever-treated units, so
  CV mimics the prediction task the estimator faces on treated cells.
* For each candidate we fit on the non-masked cells and score on the
  masked cells using one of ``mspe`` (mean squared prediction error),
  ``gmspe`` (geometric mean of squared errors), or ``mad`` (median
  absolute deviation).  The paper-faithful default is
  ``cv_rule="min"``, which chooses the candidate with the lowest
  validation score.  ``cv_rule="onepct"`` is an explicit slack option:
  it chooses the simplest candidate within 1% of the best score (lower
  ``r`` for IFE and higher ``lam`` for MC).
* Fold parallelism uses a thread-backed joblib pool so the heavy
  numpy/BLAS work runs concurrently without pickling.
* Randomness (fold construction) is seeded via :func:`pyfector.backend.make_rng`
  so CV is reproducible given ``seed``.

The CV inner loop calls into :func:`pyfector.estimators.estimate_ife`
and :func:`pyfector.estimators.estimate_mc` with a relaxed convergence
tolerance ``max(tol, 1e-3)`` — the same trick R fect uses to keep CV
cheap without affecting the final fit, which always uses the tight
tolerance.
"""

from __future__ import annotations

import math
from typing import Literal, NamedTuple

import numpy as np
from joblib import Parallel, delayed

from .backend import get_backend, to_numpy, to_device, make_rng
from .estimators import estimate_ife, estimate_mc


class CVResult(NamedTuple):
    best_r: int | None          # selected number of factors (IFE)
    best_lambda: float | None   # selected lambda (MC)
    scores: dict                # {candidate: {criterion: score}}
    all_residuals: dict         # {candidate: residuals array}


# ---------------------------------------------------------------------------
# CV sampling
# ---------------------------------------------------------------------------

def _make_cv_folds(
    II: np.ndarray,   # (T, N) control indicator
    D: np.ndarray,    # (T, N) treatment indicator
    k: int,
    cv_prop: float,
    cv_nobs: int,
    cv_treat: bool,
    cv_donut: int,
    rng,
) -> list[dict]:
    """Generate k CV folds by masking control observations.

    Returns list of dicts with keys 'mask' (cells to hide) and 'eval' (cells to score).
    """
    xp = get_backend()
    folds = []
    T, N = II.shape

    # Eligible cells: observed control cells
    if cv_treat:
        # Only mask pre-treatment cells of treated units
        ever_treated = xp.any(D > 0, axis=0)  # (N,)
        eligible = II.copy()
        eligible[:, ~ever_treated] = 0
    else:
        eligible = II.copy()

    # Total control obs to mask per fold
    total_eligible = int(xp.sum(eligible))
    rm_count = max(1, int(total_eligible * cv_prop))

    # Get indices of eligible cells
    eligible_np = to_numpy(eligible)
    elig_rows, elig_cols = np.where(eligible_np > 0)
    n_eligible = len(elig_rows)

    rng_np = np.random.default_rng(int(rng.integers(2**31)) if hasattr(rng, 'integers') else 42)

    if n_eligible == 0:
        raise ValueError("No eligible control observations for cross-validation")

    for fold_i in range(k):
        mask = np.zeros((T, N), dtype=bool)

        if cv_nobs <= 1 or rm_count <= 2:
            chosen = rng_np.choice(n_eligible, size=min(rm_count, n_eligible), replace=False)
            mask[elig_rows[chosen], elig_cols[chosen]] = True
        else:
            # Match R fect's block-style CV more closely: hide short
            # within-unit runs rather than only isolated cells.
            while int(mask.sum()) < min(rm_count, n_eligible):
                base = int(rng_np.integers(n_eligible))
                row = int(elig_rows[base])
                col = int(elig_cols[base])
                stop = min(T, row + cv_nobs)
                rows = np.arange(row, stop)
                rows = rows[eligible_np[rows, col] > 0]
                mask[rows, col] = True
                if len(rows) == 0:
                    mask[row, col] = True

            selected = np.flatnonzero(mask.ravel())
            if len(selected) > rm_count:
                drop = rng_np.choice(selected, size=len(selected) - rm_count, replace=False)
                mask.ravel()[drop] = False

        # Keep every row and usable unit identifiable after masking.
        II_after = to_numpy(II).copy()
        II_after[mask] = 0
        bad_rows = np.where(II_after.sum(axis=1) < 1)[0]
        if len(bad_rows) > 0:
            mask[bad_rows, :] = False
            II_after = to_numpy(II).copy()
            II_after[mask] = 0
        orig_col_counts = to_numpy(II).sum(axis=0)
        bad_cols = np.where((orig_col_counts > 0) & (II_after.sum(axis=0) < 1))[0]
        if len(bad_cols) > 0:
            mask[:, bad_cols] = False

        # Eval indices (same as mask, optionally excluding donut around treatment)
        eval_mask = mask.copy()
        if cv_donut > 0:
            for j in range(N):
                treat_times = np.where(to_numpy(D[:, j]) > 0)[0]
                if len(treat_times) > 0:
                    first_treat = treat_times[0]
                    donut_start = max(0, first_treat - cv_donut)
                    donut_end = min(T, first_treat + cv_donut + 1)
                    eval_mask[donut_start:donut_end, j] = False

        folds.append({
            "mask": to_device(mask),
            "eval": to_device(eval_mask),
        })

    return folds


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_residuals(
    residuals: np.ndarray,
    eval_mask: np.ndarray,
    count_weights: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute CV scoring criteria from residuals at eval cells."""
    xp = get_backend()

    e = residuals[eval_mask]
    n = max(int(xp.sum(eval_mask)), 1)

    mspe = float(xp.sum(e ** 2) / n)

    # GMSPE: geometric mean of squared errors
    e2 = e ** 2
    e2_pos = e2[e2 > 0]
    if len(e2_pos) > 0:
        gmspe = float(xp.exp(xp.mean(xp.log(e2_pos))))
    else:
        gmspe = float("inf")

    # MAD: median absolute deviation
    mad = float(xp.median(xp.abs(e))) if len(e) > 0 else float("inf")

    return {
        "mspe": mspe,
        "gmspe": gmspe,
        "mad": mad,
    }


# ---------------------------------------------------------------------------
# CV for IFE (selecting r)
# ---------------------------------------------------------------------------

def cv_ife(
    Y: np.ndarray,
    Y0: np.ndarray,
    X: np.ndarray | None,
    I: np.ndarray,
    II: np.ndarray,
    D: np.ndarray,
    W: np.ndarray | None,
    beta0: np.ndarray | None,
    force: int,
    r_range: tuple[int, int],
    k: int = 10,
    cv_prop: float = 0.1,
    cv_nobs: int = 3,
    cv_treat: bool = True,
    cv_donut: int = 0,
    criterion: str = "mspe",
    cv_rule: Literal["min", "onepct"] = "min",
    tol: float = 1e-5,
    max_iter: int = 1000,
    n_jobs: int = 1,
    seed: int | None = None,
) -> CVResult:
    """Cross-validate to select number of factors *r*.

    Parameters
    ----------
    r_range : (r_min, r_max)
        Range of candidate factor numbers.
    k : int
        Number of CV folds.
    cv_rule : {"min", "onepct"}
        ``"min"`` selects the lowest-score candidate, matching the
        paper's cross-validation description. ``"onepct"`` selects the
        lowest rank within 1% of the global best score.
    n_jobs : int
        Number of parallel workers for fold evaluation.
    """
    xp = get_backend()
    rng = make_rng(seed)

    # Looser tolerance for CV (like R fect)
    cv_tol = max(tol, 1e-3)

    # Generate folds
    folds = _make_cv_folds(II, D, k, cv_prop, cv_nobs, cv_treat, cv_donut, rng)

    r_candidates = list(range(r_range[0], r_range[1] + 1))
    scores = {}

    for r_cand in r_candidates:
        fold_residuals = []

        def _run_fold(fold):
            II_cv = II.copy()
            II_cv[fold["mask"]] = 0
            # Re-initialize Y0 for masked cells with simple mean
            Y0_cv = Y0.copy()
            if xp.sum(II_cv) > 0:
                fill_val = xp.sum(Y * II_cv) / (xp.sum(II_cv) + 1e-10)
                Y0_cv = xp.where(II_cv, Y0_cv, fill_val)

            result = estimate_ife(
                Y, Y0_cv, X, II_cv, W, beta0,
                r=r_cand, force=force, tol=cv_tol, max_iter=max_iter,
            )
            # Score on eval cells
            resid = Y - result.fit
            return _score_residuals(resid, fold["eval"])

        if n_jobs == 1:
            fold_scores = [_run_fold(f) for f in folds]
        else:
            fold_scores = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_run_fold)(f) for f in folds
            )

        # Average scores across folds
        avg_scores = {}
        for metric in fold_scores[0]:
            avg_scores[metric] = sum(fs[metric] for fs in fold_scores) / len(fold_scores)
        scores[r_cand] = avg_scores

    # r_candidates is ordered from simpler to more flexible.
    best_r = _select_best(scores, r_candidates, criterion, cv_rule=cv_rule)

    return CVResult(best_r=best_r, best_lambda=None, scores=scores, all_residuals={})


# ---------------------------------------------------------------------------
# CV for MC (selecting lambda)
# ---------------------------------------------------------------------------

def cv_mc(
    Y: np.ndarray,
    Y0: np.ndarray,
    X: np.ndarray | None,
    I: np.ndarray,
    II: np.ndarray,
    D: np.ndarray,
    W: np.ndarray | None,
    beta0: np.ndarray | None,
    force: int,
    lambda_candidates: np.ndarray | None = None,
    nlambda: int = 10,
    k: int = 10,
    cv_prop: float = 0.1,
    cv_nobs: int = 3,
    cv_treat: bool = True,
    cv_donut: int = 0,
    criterion: str = "mspe",
    cv_rule: Literal["min", "onepct"] = "min",
    tol: float = 1e-5,
    max_iter: int = 1000,
    n_jobs: int = 1,
    seed: int | None = None,
) -> CVResult:
    """Cross-validate to select lambda (nuclear norm penalty).

    If ``lambda_candidates`` is None, generates a log-spaced grid from the
    data's largest singular value down to 0. With ``cv_rule="onepct"``, the
    largest penalty within 1% of the global best score is selected.
    """
    xp = get_backend()
    rng = make_rng(seed)
    cv_tol = max(tol, 1e-3)

    # Generate lambda grid if not provided
    if lambda_candidates is None:
        lambda_candidates = _make_lambda_grid(Y, II, nlambda)

    # Generate folds
    folds = _make_cv_folds(II, D, k, cv_prop, cv_nobs, cv_treat, cv_donut, rng)

    scores = {}
    for lam in lambda_candidates:
        lam_key = float(lam)

        def _run_fold(fold):
            II_cv = II.copy()
            II_cv[fold["mask"]] = 0
            Y0_cv = Y0.copy()
            if xp.sum(II_cv) > 0:
                fill_val = xp.sum(Y * II_cv) / (xp.sum(II_cv) + 1e-10)
                Y0_cv = xp.where(II_cv, Y0_cv, fill_val)

            result = estimate_mc(
                Y, Y0_cv, X, II_cv, W, beta0,
                lam=lam_key, force=force, tol=cv_tol, max_iter=max_iter,
            )
            resid = Y - result.fit
            return _score_residuals(resid, fold["eval"])

        if n_jobs == 1:
            fold_scores = [_run_fold(f) for f in folds]
        else:
            fold_scores = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_run_fold)(f) for f in folds
            )

        avg_scores = {}
        for metric in fold_scores[0]:
            avg_scores[metric] = sum(fs[metric] for fs in fold_scores) / len(fold_scores)
        scores[lam_key] = avg_scores

    # Select from heavier to lighter regularization, i.e. from simpler
    # to more flexible. Sort here so user-supplied grids behave the same
    # way as the generated grid.
    lambda_order = sorted([float(l) for l in lambda_candidates], reverse=True)
    best_lambda = _select_best(
        scores, lambda_order, criterion,
        cv_rule=cv_rule,
    )

    return CVResult(best_r=None, best_lambda=best_lambda, scores=scores, all_residuals={})


def _make_lambda_grid(Y, II, nlambda: int):
    """Generate log-spaced lambda grid from data's singular values."""
    xp = get_backend()
    T, N = Y.shape
    Y_obs = Y * II
    # Get largest singular value
    _, s, _ = xp.linalg.svd(Y_obs / (T * N), full_matrices=False)
    lam_max = float(s[0])
    log_max = math.log10(max(lam_max, 1e-10))
    step = 3.0 / max(nlambda - 2, 1)
    grid = [10 ** (log_max - i * step) for i in range(nlambda - 1)]
    grid.append(0.0)
    return np.array(grid)


def _select_best(
    scores: dict,
    candidates: list,
    criterion: str,
    cv_rule: Literal["min", "onepct"] = "min",
):
    """Select a CV candidate by min score or an explicit 1% slack rule.

    ``"min"`` is the paper-faithful rule: select the candidate with the
    lowest validation score.  For ``cv_rule="onepct"``, ``candidates``
    must be ordered from simpler to more flexible; the selected candidate
    is the first candidate within 1% of the global best score.
    """
    if not candidates:
        raise ValueError("candidates must not be empty")
    if cv_rule not in {"min", "onepct"}:
        raise ValueError("cv_rule must be 'min' or 'onepct'")

    vals = [(c, float(scores[c][criterion])) for c in candidates]
    if cv_rule == "min":
        return min(vals, key=lambda x: x[1])[0]

    best_score = min(v for _, v in vals)
    threshold = best_score * 1.01
    for c, v in vals:
        if v <= threshold:
            return c

    return min(vals, key=lambda x: x[1])[0]
