"""
EM iteration engines for FE, IFE, MC, and CFE estimators.

Each function takes panel matrices and iterates to convergence, returning
fitted values, residuals, and estimated parameters.

Key performance improvements over R fect:
- Randomized truncated SVD (via linalg.panel_factor)
- In-place array updates to reduce allocation
- Warm-starting support (pass previous fit as Y0)
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .backend import get_backend
from .linalg import (
    ife,
    panel_beta,
    panel_factor,
    panel_FE,
    compute_xxinv,
    covar_fit,
    demean,
    fe_add,
)


# ---------------------------------------------------------------------------
# E-step helpers
# ---------------------------------------------------------------------------

def _e_adj(Y, fit, I):
    """Fill missing entries (I==0) with current fit values."""
    xp = get_backend()
    return xp.where(I, Y, fit)


def _we_adj(Y, fit, W, I):
    """Weighted E-step: observed entries shrunk toward fit by weight."""
    xp = get_backend()
    out = xp.where(I, W * Y + (1.0 - W) * fit, fit)
    return out


def _fe_adj(FE, I):
    """Zero out entries where I==0 (keep only observed residuals)."""
    xp = get_backend()
    return xp.where(I, FE, 0.0)


def _frob_ratio(new, old, W=None):
    """Relative Frobenius norm change: ||new-old||_F / (||old||_F + 1e-10)."""
    xp = get_backend()
    if W is not None:
        diff = xp.sqrt(xp.sum(W * (new - old) ** 2))
        base = xp.sqrt(xp.sum(W * old ** 2)) + 1e-10
    else:
        diff = xp.linalg.norm(new - old, "fro")
        base = xp.linalg.norm(old, "fro") + 1e-10
    return float(diff / base)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

class EstimationResult(NamedTuple):
    beta: np.ndarray | None    # (p,) covariate coefficients
    mu: float                  # grand mean
    alpha: np.ndarray | None   # (N,) unit FE
    xi: np.ndarray | None      # (T,) time FE
    factors: np.ndarray | None # (T, r) latent factors
    loadings: np.ndarray | None  # (N, r) factor loadings
    eigenvalues: np.ndarray | None  # (r,) eigenvalues
    fit: np.ndarray            # (T, N) fitted values (counterfactual)
    residuals: np.ndarray      # (T, N) residuals (observed only)
    sigma2: float              # error variance estimate
    IC: float                  # information criterion (BIC-type)
    PC: float                  # PC criterion
    niter: int                 # iterations to convergence
    converged: bool            # whether converged


# ---------------------------------------------------------------------------
# FE / IFE estimator  (inter_fe_ub equivalent)
# ---------------------------------------------------------------------------

def estimate_ife(
    Y: np.ndarray,        # (T, N) outcome
    Y0: np.ndarray,       # (T, N) initial imputation
    X: np.ndarray | None, # (T, N, p) covariates or None
    I: np.ndarray,        # (T, N) observation indicator
    W: np.ndarray | None, # (T, N) weights or None
    beta0: np.ndarray | None,  # (p,) initial beta
    r: int,               # number of factors
    force: int,           # FE type: 0/1/2/3
    tol: float = 1e-5,
    max_iter: int = 1000,
) -> EstimationResult:
    """EM algorithm for interactive fixed effects on unbalanced panels.

    Equivalent to R fect's ``inter_fe_ub()``.
    """
    xp = get_backend()
    T, N = Y.shape
    use_weight = W is not None and W.shape == Y.shape
    has_covar = X is not None and X.shape[2] > 0

    # Normalize weights
    if use_weight:
        W = W / (xp.max(W) + 1e-10)
        WI = W * I
    else:
        WI = None

    # Check covariate variation and remove invariant ones
    valid_x = []
    if has_covar:
        for k in range(X.shape[2]):
            if xp.sum(xp.abs(X[:, :, k])) > 1e-5:
                valid_x.append(k)
        if valid_x:
            X = X[:, :, valid_x]
        else:
            X = None
            has_covar = False

    p = X.shape[2] if has_covar else 0

    # Precompute (X'X)^-1
    xxinv = None
    if has_covar:
        xxinv = compute_xxinv(X, W=WI if use_weight else None)

    # Initialize
    beta = beta0[:len(valid_x)] if beta0 is not None and has_covar else (
        xp.zeros(p) if has_covar else None
    )
    fit = Y0.copy()

    # Dispatch to appropriate EM loop
    if not has_covar and r == 0 and force == 0:
        # Trivial case: just grand mean
        if use_weight:
            mu = float(xp.sum(Y * WI) / (xp.sum(WI) + 1e-10))
        else:
            mu = float(xp.sum(Y * I) / (xp.sum(I) + 1e-10))
        fit = xp.full_like(Y, mu)
        residuals = _fe_adj(Y - fit, I)
        obs = float(xp.sum(I))
        sigma2 = float(xp.sum(residuals ** 2) / max(obs - 1, 1))
        return EstimationResult(
            beta=None, mu=mu, alpha=None, xi=None,
            factors=None, loadings=None, eigenvalues=None,
            fit=fit, residuals=residuals, sigma2=sigma2,
            IC=0.0, PC=0.0, niter=0, converged=True,
        )

    if not has_covar and r > 0:
        return _em_fe_inter(Y, Y0, I, W, WI, use_weight, force, r, tol, max_iter)
    elif not has_covar and r == 0:
        return _em_fe_ad(Y, Y0, I, W, WI, use_weight, force, tol, max_iter)
    elif has_covar and r == 0:
        return _em_fe_ad_covar(Y, Y0, X, I, W, WI, use_weight, beta, xxinv, force, tol, max_iter)
    else:
        return _em_fe_inter_covar(Y, Y0, X, I, W, WI, use_weight, beta, xxinv, force, r, tol, max_iter)


# ---------------------------------------------------------------------------
# EM: additive FE only, no covariates
# ---------------------------------------------------------------------------

def _em_fe_ad(Y, Y0, I, W, WI, use_weight, force, tol, max_iter):
    xp = get_backend()
    T, N = Y.shape
    fit = Y0.copy()
    converged = False

    for niter in range(1, max_iter + 1):
        fit_old = fit.copy()

        # E-step
        if use_weight:
            YY = _we_adj(Y, fit, W, I)
        else:
            YY = _e_adj(Y, fit, I)

        # M-step: demean and reconstruct
        ife_res = ife(YY, force, mc=False, r=0)
        fit = ife_res.FE

        # Convergence
        dif = _frob_ratio(fit, fit_old, W=WI if use_weight else None)
        if dif < tol:
            converged = True
            break

    residuals = _fe_adj(Y - fit, I)
    obs = float(xp.sum(I))
    sigma2 = float(xp.sum(residuals ** 2) / max(obs - 1, 1))

    return EstimationResult(
        beta=None, mu=ife_res.mu, alpha=ife_res.alpha, xi=ife_res.xi,
        factors=None, loadings=None, eigenvalues=None,
        fit=fit, residuals=residuals, sigma2=sigma2,
        IC=0.0, PC=0.0, niter=niter, converged=converged,
    )


# ---------------------------------------------------------------------------
# EM: additive + interactive FE, no covariates  (with burn-in)
# ---------------------------------------------------------------------------

def _em_fe_inter(Y, Y0, I, W, WI, use_weight, force, r, tol, max_iter):
    xp = get_backend()
    T, N = Y.shape
    fit = Y0.copy()
    d = min(T, N)
    converged = False
    burn_in_done = False
    niter_total = 0

    for phase in range(2):
        # Phase 0: burn-in (start with high rank, reduce gradually)
        # Phase 1: final estimation with target r
        if phase == 0 and not use_weight:
            # Skip burn-in for unweighted estimation
            continue

        fit_phase = fit.copy() if phase == 0 else Y0.copy()
        fit_current = fit_phase.copy()

        for niter in range(1, max_iter + 1):
            fit_old = fit_current.copy()

            # E-step
            if use_weight:
                YY = _we_adj(Y, fit_current, W, I)
            else:
                YY = _e_adj(Y, fit_current, I)

            # Current rank (burn-in decreases from d)
            if phase == 0:
                r_current = max(r, d - niter)
            else:
                r_current = r

            # M-step
            ife_res = ife(YY, force, mc=False, r=r_current)
            fit_current = ife_res.FE

            # Convergence
            dif = _frob_ratio(fit_current, fit_old, W=WI if use_weight else None)
            niter_total += 1

            if dif < tol:
                if phase == 0:
                    burn_in_done = True
                    break
                else:
                    converged = True
                    break

        if phase == 0 and burn_in_done:
            fit = fit_current
            continue
        elif phase == 1:
            fit = fit_current

    residuals = _fe_adj(Y - fit, I)
    obs = float(xp.sum(I))
    np_param = r * (N + T) - r * r + 1
    if force in (1, 3):
        np_param += N - 1
    if force in (2, 3):
        np_param += T - 1
    denom = max(obs - np_param, 1)
    sigma2 = float(xp.sum(residuals ** 2) / denom)

    IC = _compute_ic(sigma2, np_param, obs)
    PC = _compute_pc(residuals, np_param, T, N, obs)

    return EstimationResult(
        beta=None, mu=ife_res.mu, alpha=ife_res.alpha, xi=ife_res.xi,
        factors=ife_res.factors, loadings=ife_res.loadings,
        eigenvalues=ife_res.eigenvalues,
        fit=fit, residuals=residuals, sigma2=sigma2,
        IC=IC, PC=PC, niter=niter_total, converged=converged,
    )


# ---------------------------------------------------------------------------
# EM: additive FE + covariates, no interactive FE
# ---------------------------------------------------------------------------

def _em_fe_ad_covar(Y, Y0, X, I, W, WI, use_weight, beta, xxinv, force, tol, max_iter):
    xp = get_backend()
    T, N = Y.shape
    p = X.shape[2]
    fit = Y0.copy()
    cf = covar_fit(X, beta)
    FE = fit - cf
    converged = False

    for niter in range(1, max_iter + 1):
        fit_old = fit.copy()

        # E-step
        if use_weight:
            YY = _we_adj(Y, fit, W, I)
        else:
            YY = _e_adj(Y, fit, I)

        # M1: beta
        beta = panel_beta(X, YY, FE, xxinv, W=WI if use_weight else None)
        cf = covar_fit(X, beta)

        # M2: FE
        if use_weight:
            U = _we_adj(YY - cf, FE, W, I)
        else:
            U = _e_adj(YY - cf, FE, I)
        ife_res = ife(U, force, mc=False, r=0)
        FE = ife_res.FE
        fit = cf + FE

        # Convergence
        dif = _frob_ratio(fit, fit_old, W=WI if use_weight else None)
        if dif < tol:
            converged = True
            break

    residuals = _fe_adj(Y - fit, I)
    obs = float(xp.sum(I))
    np_param = p + 1
    if force in (1, 3):
        np_param += N - 1
    if force in (2, 3):
        np_param += T - 1
    sigma2 = float(xp.sum(residuals ** 2) / max(obs - np_param, 1))

    return EstimationResult(
        beta=beta, mu=ife_res.mu, alpha=ife_res.alpha, xi=ife_res.xi,
        factors=None, loadings=None, eigenvalues=None,
        fit=fit, residuals=residuals, sigma2=sigma2,
        IC=0.0, PC=0.0, niter=niter, converged=converged,
    )


# ---------------------------------------------------------------------------
# EM: full model — additive FE + interactive FE + covariates  (with burn-in)
# ---------------------------------------------------------------------------

def _em_fe_inter_covar(Y, Y0, X, I, W, WI, use_weight, beta, xxinv, force, r, tol, max_iter):
    xp = get_backend()
    T, N = Y.shape
    p = X.shape[2]
    d = min(T, N)
    cf = covar_fit(X, beta)
    FE = Y0 - cf
    fit = Y0.copy()
    converged = False
    burn_in_done = False
    niter_total = 0

    for phase in range(2):
        if phase == 0 and not use_weight:
            continue

        fit_current = fit.copy() if phase == 0 else Y0.copy()
        cf_current = cf.copy()
        FE_current = FE.copy()
        beta_current = beta.copy()

        for niter in range(1, max_iter + 1):
            fit_old = fit_current.copy()

            # E-step
            if use_weight:
                YY = _we_adj(Y, fit_current, W, I)
            else:
                YY = _e_adj(Y, fit_current, I)

            # M1: beta
            beta_current = panel_beta(X, YY, FE_current, xxinv,
                                       W=WI if use_weight else None)
            cf_current = covar_fit(X, beta_current)

            # M2: FE (additive + interactive)
            r_current = max(r, d - niter) if phase == 0 else r
            if use_weight:
                U = _we_adj(YY - cf_current, FE_current, W, I)
            else:
                U = _e_adj(YY - cf_current, FE_current, I)
            ife_res = ife(U, force, mc=False, r=r_current)
            FE_current = ife_res.FE
            fit_current = cf_current + FE_current

            dif = _frob_ratio(fit_current, fit_old, W=WI if use_weight else None)
            niter_total += 1

            if dif < tol:
                if phase == 0:
                    burn_in_done = True
                    break
                else:
                    converged = True
                    break

        beta = beta_current
        cf = cf_current
        FE = FE_current
        fit = fit_current

    residuals = _fe_adj(Y - fit, I)
    obs = float(xp.sum(I))
    np_param = r * (N + T) - r * r + p + 1
    if force in (1, 3):
        np_param += N - 1
    if force in (2, 3):
        np_param += T - 1
    denom = max(obs - np_param, 1)
    sigma2 = float(xp.sum(residuals ** 2) / denom)

    IC = _compute_ic(sigma2, np_param, obs)
    PC = _compute_pc(residuals, np_param, T, N, obs)

    return EstimationResult(
        beta=beta, mu=ife_res.mu, alpha=ife_res.alpha, xi=ife_res.xi,
        factors=ife_res.factors, loadings=ife_res.loadings,
        eigenvalues=ife_res.eigenvalues,
        fit=fit, residuals=residuals, sigma2=sigma2,
        IC=IC, PC=PC, niter=niter_total, converged=converged,
    )


# ---------------------------------------------------------------------------
# MC estimator  (inter_fe_mc equivalent)
# ---------------------------------------------------------------------------

def estimate_mc(
    Y: np.ndarray,
    Y0: np.ndarray,
    X: np.ndarray | None,
    I: np.ndarray,
    W: np.ndarray | None,
    beta0: np.ndarray | None,
    lam: float,
    force: int,
    tol: float = 1e-5,
    max_iter: int = 1000,
) -> EstimationResult:
    """Matrix completion estimator via nuclear norm penalization.

    Equivalent to R fect's ``inter_fe_mc()``.  Uses soft singular value
    thresholding in each EM iteration.
    """
    xp = get_backend()
    T, N = Y.shape
    use_weight = W is not None and W.shape == Y.shape
    has_covar = X is not None and X.shape[2] > 0

    if use_weight:
        W = W / (xp.max(W) + 1e-10)
        WI = W * I
    else:
        WI = None

    # Check covariate variation
    valid_x = []
    if has_covar:
        for k in range(X.shape[2]):
            if xp.sum(xp.abs(X[:, :, k])) > 1e-5:
                valid_x.append(k)
        if valid_x:
            X = X[:, :, valid_x]
        else:
            X = None
            has_covar = False

    p = X.shape[2] if has_covar else 0

    xxinv = None
    if has_covar:
        xxinv = compute_xxinv(X, W=WI if use_weight else None)

    beta = beta0[:len(valid_x)] if beta0 is not None and has_covar else (
        xp.zeros(p) if has_covar else None
    )
    fit = Y0.copy()
    cf = covar_fit(X, beta) if has_covar else xp.zeros_like(Y)
    FE = fit - cf
    converged = False

    for niter in range(1, max_iter + 1):
        fit_old = fit.copy()

        # E-step
        if use_weight:
            YY = _we_adj(Y, fit, W, I)
        else:
            YY = _e_adj(Y, fit, I)

        # M1: beta (if covariates)
        if has_covar:
            beta = panel_beta(X, YY, FE, xxinv, W=WI if use_weight else None)
            cf = covar_fit(X, beta)

        # M2: FE (additive + nuclear norm penalized interactive)
        if use_weight:
            U = _we_adj(YY - cf, FE, W, I)
        else:
            U = _e_adj(YY - cf, FE, I)
        ife_res = ife(U, force, mc=True, r=1, hard=False, lam=lam)
        FE = ife_res.FE
        fit = cf + FE

        dif = _frob_ratio(fit, fit_old, W=WI if use_weight else None)
        if dif < tol:
            converged = True
            break

    residuals = _fe_adj(Y - fit, I)
    obs = float(xp.sum(I))
    sigma2 = float(xp.sum(residuals ** 2) / max(obs - 1, 1))

    return EstimationResult(
        beta=beta, mu=ife_res.mu, alpha=ife_res.alpha, xi=ife_res.xi,
        factors=ife_res.factors, loadings=ife_res.loadings,
        eigenvalues=ife_res.eigenvalues,
        fit=fit, residuals=residuals, sigma2=sigma2,
        IC=0.0, PC=0.0, niter=niter, converged=converged,
    )


# ---------------------------------------------------------------------------
# CFE estimator  (complex_fe_ub equivalent)
# ---------------------------------------------------------------------------

def estimate_cfe(
    Y: np.ndarray,
    Y0: np.ndarray,
    X: np.ndarray | None,
    I: np.ndarray,
    W: np.ndarray | None,
    beta0: np.ndarray | None,
    r: int,
    force: int,
    Z: np.ndarray | None = None,       # (N, p_z) unit-invariant covariates
    Q: np.ndarray | None = None,        # (p_q, T) time-invariant covariates
    gamma_groups: np.ndarray | None = None,  # (T,) group assignments for gamma
    kappa_groups: np.ndarray | None = None,  # (N,) group assignments for kappa
    tol: float = 1e-5,
    max_iter: int = 1000,
) -> EstimationResult:
    """Complex fixed effects estimator with Z*gamma_t and kappa_i*Q.

    Equivalent to R fect's ``complex_fe_ub()`` (simplified: no extra FE dims).
    Uses block coordinate descent / EM.
    """
    xp = get_backend()
    T, N = Y.shape
    use_weight = W is not None and W.shape == Y.shape
    has_covar = X is not None and X.shape[2] > 0
    has_Z = Z is not None and Z.shape[1] > 0
    has_Q = Q is not None and Q.shape[0] > 0

    if use_weight:
        W = W / (xp.max(W) + 1e-10)
        WI = W * I
    else:
        WI = None

    p = X.shape[2] if has_covar else 0
    xxinv = compute_xxinv(X, W=WI if use_weight else None) if has_covar else None
    beta = beta0 if beta0 is not None and has_covar else (
        xp.zeros(p) if has_covar else None
    )

    # Precompute Z'Z inverse and Q Q' inverse
    zzinv = xp.linalg.inv(Z.T @ Z) if has_Z else None
    qqinv = xp.linalg.inv(Q @ Q.T) if has_Q else None

    # Default group assignments (each period/unit is its own group)
    if gamma_groups is None and has_Z:
        gamma_groups = xp.arange(T)
    if kappa_groups is None and has_Q:
        kappa_groups = xp.arange(N)

    # Initialize fits
    fit = Y0.copy()
    cf = covar_fit(X, beta) if has_covar else xp.zeros_like(Y)
    gamma_fit = xp.zeros_like(Y)
    kappa_fit = xp.zeros_like(Y)
    FE_ad = xp.zeros_like(Y)
    FE_inter = xp.zeros_like(Y)

    converged = False
    consec_converged = 0

    for niter in range(1, max_iter + 1):
        fit_old = fit.copy()

        # E-step
        if use_weight:
            YY = _we_adj(Y, fit, W, I)
        else:
            YY = _e_adj(Y, fit, I)

        # Block 1: beta
        if has_covar:
            residual_for_beta = YY - gamma_fit - kappa_fit - FE_ad - FE_inter
            beta = panel_beta(X, residual_for_beta + cf, FE_ad + FE_inter + gamma_fit + kappa_fit,
                              xxinv, W=WI if use_weight else None)
            cf = covar_fit(X, beta)

        # Block 2: gamma (time-varying coefficients on Z)
        if has_Z:
            E_gamma = YY - cf - kappa_fit - FE_ad - FE_inter
            gamma_fit = _estimate_gamma(E_gamma, Z, zzinv, gamma_groups, T, N, xp)

        # Block 3: kappa (unit-varying coefficients on Q)
        if has_Q:
            E_kappa = YY - cf - gamma_fit - FE_ad - FE_inter
            kappa_fit = _estimate_kappa(E_kappa, Q, qqinv, kappa_groups, T, N, xp)

        # Block 4: additive FE
        E_ad = YY - cf - gamma_fit - kappa_fit - FE_inter
        dm = demean(E_ad, force)
        FE_ad = fe_add(dm.mu, dm.alpha, dm.xi, T, N, force)

        # Block 5: interactive FE
        E_inter = YY - cf - gamma_fit - kappa_fit - FE_ad
        if r > 0:
            fr = panel_factor(E_inter - xp.mean(E_inter), r)
            FE_inter = fr.FE
        else:
            FE_inter = xp.zeros_like(Y)

        fit = cf + gamma_fit + kappa_fit + FE_ad + FE_inter

        dif = _frob_ratio(fit, fit_old, W=WI if use_weight else None)
        if dif < tol:
            consec_converged += 1
            if consec_converged >= 3:
                converged = True
                break
        else:
            consec_converged = 0

    residuals = _fe_adj(Y - fit, I)
    obs = float(xp.sum(I))
    sigma2 = float(xp.sum(residuals ** 2) / max(obs - 1, 1))

    return EstimationResult(
        beta=beta, mu=float(dm.mu), alpha=dm.alpha, xi=dm.xi,
        factors=fr.factors if r > 0 else None,
        loadings=fr.loadings if r > 0 else None,
        eigenvalues=fr.eigenvalues if r > 0 else None,
        fit=fit, residuals=residuals, sigma2=sigma2,
        IC=0.0, PC=0.0, niter=niter, converged=converged,
    )


def _estimate_gamma(E, Z, zzinv, gamma_groups, T, N, xp):
    """Estimate time-varying coefficients gamma: Y ≈ Z @ gamma_t."""
    unique_groups = xp.unique(gamma_groups)
    gamma_coefs = xp.zeros((Z.shape[1], T), dtype=E.dtype)
    for g in unique_groups:
        g_int = int(g)
        mask = gamma_groups == g
        E_sum = xp.sum(E[mask, :], axis=0)  # (N,)
        count = int(xp.sum(mask))
        if count > 0:
            y_bar = E_sum / count
            gamma_g = zzinv @ Z.T @ y_bar  # (p_z,)
            gamma_coefs[:, mask] = gamma_g[:, None]
    return (Z @ gamma_coefs).T  # (T, N)


def _estimate_kappa(E, Q, qqinv, kappa_groups, T, N, xp):
    """Estimate unit-varying coefficients kappa: Y ≈ kappa_i @ Q."""
    unique_groups = xp.unique(kappa_groups)
    kappa_coefs = xp.zeros((N, Q.shape[0]), dtype=E.dtype)
    for g in unique_groups:
        g_int = int(g)
        mask = kappa_groups == g
        E_sum = xp.sum(E[:, mask], axis=1)  # (T,)
        count = int(xp.sum(mask))
        if count > 0:
            y_bar = E_sum / count
            kappa_g = qqinv @ Q @ y_bar  # (p_q,)
            kappa_coefs[mask, :] = kappa_g[None, :]
    return (kappa_coefs @ Q).T if Q.shape[0] > 0 else xp.zeros((T, N), dtype=E.dtype)


# ---------------------------------------------------------------------------
# IC / PC helpers
# ---------------------------------------------------------------------------

def _compute_ic(sigma2: float, np_param: int, obs: float) -> float:
    """BIC-type information criterion."""
    import math
    if sigma2 <= 0:
        return float("inf")
    return math.log(sigma2) + np_param * math.log(obs) / obs


def _compute_pc(residuals, np_param: int, T: int, N: int, obs: float) -> float:
    """PC criterion (Li 2018)."""
    xp = get_backend()
    mse = float(xp.sum(residuals ** 2) / max(obs, 1))
    # Penalty term
    penalty = np_param * mse * (T + N) / (T * N) if T * N > 0 else 0
    return mse + penalty
