"""
Main Fect estimator class — the public API of pyfect.

Usage:
    import pyfect

    result = pyfect.fect(
        data=df,
        Y="outcome", D="treat",
        index=("unit", "year"),
        X=["gdp", "pop"],
        method="ife",
        r=(0, 5),            # CV over 0..5 factors
        se=True,
        nboots=200,
        device="cpu",         # or "gpu"
        n_jobs=4,
        seed=42,
    )

    result.summary()
    result.plot()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .backend import set_device, get_backend, to_numpy, to_device, make_rng
from .panel import PanelData, prepare_panel, initial_fit
from .estimators import estimate_ife, estimate_mc, estimate_cfe, EstimationResult
from .cv import cv_ife, cv_mc, CVResult
from .inference import bootstrap, jackknife, InferenceResult


@dataclass
class FectResult:
    """Container for all fect estimation results."""
    # Method info
    method: str
    r_cv: int | None = None
    lambda_cv: float | None = None

    # Point estimates
    att_avg: float = 0.0
    att_avg_unit: float = 0.0

    # Dynamic effects
    att_on: np.ndarray | None = None
    time_on: np.ndarray | None = None
    count_on: np.ndarray | None = None

    # Exit effects (treatment reversal)
    att_off: np.ndarray | None = None
    time_off: np.ndarray | None = None

    # Coefficients
    beta: np.ndarray | None = None
    covariate_names: list[str] = field(default_factory=list)

    # Fixed effects
    mu: float = 0.0
    alpha: np.ndarray | None = None   # unit FE
    xi: np.ndarray | None = None      # time FE
    factors: np.ndarray | None = None
    loadings: np.ndarray | None = None

    # Counterfactual and effects matrices
    Y_ct: np.ndarray | None = None    # T×N counterfactual
    eff: np.ndarray | None = None     # T×N treatment effects
    residuals: np.ndarray | None = None

    # Model fit
    sigma2: float = 0.0
    IC: float = 0.0
    PC: float = 0.0
    rmse: float = 0.0
    niter: int = 0
    converged: bool = False

    # Inference
    inference: InferenceResult | None = None

    # CV
    cv_result: CVResult | None = None

    # Panel metadata
    panel: PanelData | None = None

    # Reproducibility
    seed: int | None = None

    def summary(self) -> str:
        """Print summary table of results."""
        lines = []
        lines.append(f"pyfect estimation results")
        lines.append(f"{'='*60}")
        lines.append(f"Method: {self.method}")
        if self.r_cv is not None:
            lines.append(f"Number of factors (CV): {self.r_cv}")
        if self.lambda_cv is not None:
            lines.append(f"Lambda (CV): {self.lambda_cv:.6f}")
        lines.append(f"Converged: {self.converged} (iter={self.niter})")
        lines.append(f"Sigma^2: {self.sigma2:.6f}")
        lines.append(f"")
        lines.append(f"ATT (average): {self.att_avg:.6f}")
        if self.inference is not None:
            inf = self.inference
            lines.append(f"  SE:     {inf.att_avg_se:.6f}")
            lines.append(f"  CI:     [{inf.att_avg_ci[0]:.6f}, {inf.att_avg_ci[1]:.6f}]")
            lines.append(f"  p-val:  {inf.att_avg_pval:.4f}")

        if self.beta is not None and len(self.beta) > 0:
            lines.append(f"")
            lines.append(f"Coefficients:")
            for i, name in enumerate(self.covariate_names):
                lines.append(f"  {name}: {self.beta[i]:.6f}")

        if self.att_on is not None and self.time_on is not None:
            lines.append(f"")
            lines.append(f"Dynamic effects (ATT by relative time):")
            lines.append(f"  {'Time':>6s}  {'ATT':>10s}  {'Count':>6s}", )
            for i, t in enumerate(self.time_on):
                count = self.count_on[i] if self.count_on is not None else ""
                att = self.att_on[i]
                if self.inference is not None:
                    se = self.inference.att_on_se[i]
                    lines.append(f"  {t:>6.0f}  {att:>10.4f}  ({se:.4f})  {count}")
                else:
                    lines.append(f"  {t:>6.0f}  {att:>10.4f}  {count}")

        lines.append(f"{'='*60}")
        if self.panel is not None:
            lines.append(f"N={self.panel.N}, T={self.panel.T}")
        if self.seed is not None:
            lines.append(f"Seed: {self.seed}")
        return "\n".join(lines)

    def __repr__(self):
        return self.summary()

    def plot(self, kind="gap", **kwargs):
        """Plot results. Shortcut for ``pyfect.plot(self, kind, ...)``."""
        from .plotting import plot as _plot
        return _plot(self, kind=kind, **kwargs)

    def diagnose(self, **kwargs):
        """Run diagnostic tests. Shortcut for ``pyfect.run_diagnostics(self, ...)``."""
        from .diagnostics import run_diagnostics
        return run_diagnostics(self, **kwargs)


def fect(
    data,
    Y: str,
    D: str,
    index: tuple[str, str],
    X: list[str] | None = None,
    W: str | None = None,
    group: str | None = None,
    method: Literal["fe", "ife", "mc", "cfe", "both"] = "ife",
    force: Literal["none", "unit", "time", "two-way"] = "two-way",
    r: int | tuple[int, int] = 0,
    lam: float | None = None,
    nlambda: int = 10,
    CV: bool = True,
    k: int = 10,
    cv_prop: float = 0.1,
    cv_nobs: int = 3,
    cv_treat: bool = True,
    cv_donut: int = 0,
    criterion: str = "mspe",
    se: bool = False,
    vartype: Literal["bootstrap", "jackknife"] = "bootstrap",
    nboots: int = 200,
    alpha: float = 0.05,
    tol: float = 1e-5,
    max_iter: int = 1000,
    min_T0: int = 1,
    max_missing: float = 0.5,
    normalize: bool = False,
    # CFE-specific
    Z: list[str] | None = None,
    Q: list[str] | None = None,
    # Performance
    device: Literal["cpu", "gpu"] = "cpu",
    n_jobs: int = 1,
    seed: int | None = None,
) -> FectResult:
    """Estimate counterfactual treatment effects for panel data.

    This is the main entry point — equivalent to R fect's ``fect()`` function.

    Parameters
    ----------
    data : polars.DataFrame, pandas.DataFrame
        Long-format panel data.
    Y, D : str
        Column names for outcome and binary treatment indicator.
    index : (str, str)
        Column names for (unit_id, time_period).
    X : list of str, optional
        Time-varying covariates.
    method : {"fe", "ife", "mc", "cfe", "both"}
        Estimation method.
    force : {"none", "unit", "time", "two-way"}
        Fixed effects specification.
    r : int or (int, int)
        Number of factors.  If tuple, CV selects from range.
    lam : float, optional
        Nuclear norm penalty for MC.  If None with CV=True, auto-selected.
    se : bool
        Compute standard errors via bootstrap/jackknife.
    device : {"cpu", "gpu"}
        Compute device.
    n_jobs : int
        Parallel workers for CV and bootstrap.
    seed : int, optional
        Random seed for full reproducibility.
    """
    # Set device
    set_device(device)
    xp = get_backend()

    # Map force string to int
    force_map = {"none": 0, "unit": 1, "time": 2, "two-way": 3}
    force_int = force_map[force]

    # Prepare panel data
    panel = prepare_panel(
        data, Y=Y, D=D, index=index, X=X, W=W,
        group=group, min_T0=min_T0, max_missing=max_missing,
    )

    # Move to device
    Y_mat = to_device(panel.Y)
    D_mat = to_device(panel.D)
    I_mat = to_device(panel.I)
    II_mat = to_device(panel.II)
    X_mat = to_device(panel.X) if panel.X is not None else None
    W_mat = to_device(panel.W) if panel.W is not None else None

    # Normalize
    norm_factor = 1.0
    if normalize:
        sd_y = float(xp.std(Y_mat[I_mat > 0]))
        if sd_y > 0:
            Y_mat = Y_mat / sd_y
            norm_factor = sd_y

    # Initial fit
    Y0, beta0 = initial_fit(Y_mat, X_mat, II_mat, force_int)

    # Determine r and lambda
    r_cv = None
    lambda_cv = None
    cv_result = None

    if method == "ife":
        if isinstance(r, tuple) and CV:
            cv_result = cv_ife(
                Y_mat, Y0, X_mat, I_mat, II_mat, D_mat, W_mat, beta0,
                force=force_int, r_range=r, k=k, cv_prop=cv_prop,
                cv_nobs=cv_nobs, cv_treat=cv_treat, cv_donut=cv_donut,
                criterion=criterion, tol=tol, max_iter=max_iter,
                n_jobs=n_jobs, seed=seed,
            )
            r_cv = cv_result.best_r
        else:
            r_cv = r if isinstance(r, int) else r[0]

    elif method == "mc":
        if lam is None and CV:
            cv_result = cv_mc(
                Y_mat, Y0, X_mat, I_mat, II_mat, D_mat, W_mat, beta0,
                force=force_int, nlambda=nlambda, k=k, cv_prop=cv_prop,
                cv_nobs=cv_nobs, cv_treat=cv_treat, cv_donut=cv_donut,
                criterion=criterion, tol=tol, max_iter=max_iter,
                n_jobs=n_jobs, seed=seed,
            )
            lambda_cv = cv_result.best_lambda
        else:
            lambda_cv = lam if lam is not None else 0.0

    elif method == "fe":
        r_cv = 0

    elif method == "cfe":
        r_cv = r if isinstance(r, int) else r[0]

    # Point estimation
    if method in ("fe", "ife"):
        est = estimate_ife(
            Y_mat, Y0, X_mat, II_mat, W_mat, beta0,
            r=r_cv, force=force_int, tol=tol, max_iter=max_iter,
        )
    elif method == "mc":
        est = estimate_mc(
            Y_mat, Y0, X_mat, II_mat, W_mat, beta0,
            lam=lambda_cv, force=force_int, tol=tol, max_iter=max_iter,
        )
    elif method == "cfe":
        est = estimate_cfe(
            Y_mat, Y0, X_mat, II_mat, W_mat, beta0,
            r=r_cv, force=force_int, tol=tol, max_iter=max_iter,
        )
    elif method == "both":
        # Run both IFE and MC, return IFE results with MC comparison
        if isinstance(r, tuple) and CV:
            cv_result = cv_ife(
                Y_mat, Y0, X_mat, I_mat, II_mat, D_mat, W_mat, beta0,
                force=force_int, r_range=r, k=k, cv_prop=cv_prop,
                cv_nobs=cv_nobs, cv_treat=cv_treat, cv_donut=cv_donut,
                criterion=criterion, tol=tol, max_iter=max_iter,
                n_jobs=n_jobs, seed=seed,
            )
            r_cv = cv_result.best_r
        else:
            r_cv = r if isinstance(r, int) else r[0]
        est = estimate_ife(
            Y_mat, Y0, X_mat, II_mat, W_mat, beta0,
            r=r_cv, force=force_int, tol=tol, max_iter=max_iter,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    # Compute effects
    eff = Y_mat - est.fit
    Y_ct = est.fit

    # Denormalize
    if normalize and norm_factor != 1.0:
        eff = eff * norm_factor
        Y_ct = Y_ct * norm_factor
        Y_mat = Y_mat * norm_factor
        if est.beta is not None:
            est = est._replace(beta=est.beta * norm_factor)

    # ATT computation
    T_on = to_device(panel.T_on)
    att_avg, att_on, time_on, count_on, att_avg_unit = _compute_effects(
        to_numpy(eff), to_numpy(D_mat), to_numpy(panel.T_on), to_numpy(I_mat),
    )

    # Build result
    result = FectResult(
        method=method,
        r_cv=r_cv,
        lambda_cv=lambda_cv,
        att_avg=att_avg,
        att_avg_unit=att_avg_unit,
        att_on=att_on,
        time_on=time_on,
        count_on=count_on,
        beta=to_numpy(est.beta) if est.beta is not None else None,
        covariate_names=panel.covariate_names,
        mu=est.mu,
        alpha=to_numpy(est.alpha) if est.alpha is not None else None,
        xi=to_numpy(est.xi) if est.xi is not None else None,
        factors=to_numpy(est.factors) if est.factors is not None else None,
        loadings=to_numpy(est.loadings) if est.loadings is not None else None,
        Y_ct=to_numpy(Y_ct),
        eff=to_numpy(eff),
        residuals=to_numpy(est.residuals),
        sigma2=est.sigma2,
        IC=est.IC,
        PC=est.PC,
        niter=est.niter,
        converged=est.converged,
        cv_result=cv_result,
        panel=panel,
        seed=seed,
    )

    # Inference
    if se:
        result.inference = _run_inference(
            result, panel, Y_mat, X_mat, W_mat, beta0, Y0,
            method=method, r_cv=r_cv, lambda_cv=lambda_cv,
            force_int=force_int, tol=tol, max_iter=max_iter,
            vartype=vartype, nboots=nboots, alpha=alpha,
            n_jobs=n_jobs, seed=seed, normalize=normalize,
            norm_factor=norm_factor,
        )

    return result


def _compute_effects(eff, D, T_on, I):
    """Compute ATT, dynamic ATT, and counts."""
    treated = (D > 0) & (I > 0)
    n_treated = np.sum(treated)

    att_avg = float(np.sum(eff[treated]) / max(n_treated, 1))

    # Per-unit ATT
    N = eff.shape[1]
    unit_atts = []
    for j in range(N):
        mask_j = treated[:, j]
        if np.any(mask_j):
            unit_atts.append(float(np.mean(eff[mask_j, j])))
    att_avg_unit = float(np.mean(unit_atts)) if unit_atts else 0.0

    # Dynamic ATT by relative time
    T_on_flat = T_on[treated].ravel()
    eff_flat = eff[treated].ravel()
    valid = ~np.isnan(T_on_flat)
    T_on_flat = T_on_flat[valid]
    eff_flat = eff_flat[valid]

    time_on = np.sort(np.unique(T_on_flat))
    att_on = np.array([float(np.mean(eff_flat[T_on_flat == t])) for t in time_on])
    count_on = np.array([int(np.sum(T_on_flat == t)) for t in time_on])

    return att_avg, att_on, time_on, count_on, att_avg_unit


def _run_inference(
    result, panel, Y_mat, X_mat, W_mat, beta0, Y0,
    method, r_cv, lambda_cv, force_int, tol, max_iter,
    vartype, nboots, alpha, n_jobs, seed, normalize, norm_factor,
):
    """Run bootstrap or jackknife inference."""
    xp = get_backend()

    # Pre-move panel data to device once (avoids CPU→GPU copy per bootstrap rep)
    II_dev = to_device(panel.II)
    D_dev = to_device(panel.D)
    I_dev = to_device(panel.I)
    T_on_np = panel.T_on  # keep on CPU for ATT computation

    def _estimate_fn(unit_idx):
        """Re-estimate on a subset of units."""
        Y_sub = Y_mat[:, unit_idx]
        II_sub = II_dev[:, unit_idx]
        D_sub = D_dev[:, unit_idx]
        I_sub = I_dev[:, unit_idx]
        X_sub = X_mat[:, unit_idx, :] if X_mat is not None else None
        W_sub = W_mat[:, unit_idx] if W_mat is not None else None
        T_on_sub = T_on_np[:, unit_idx]
        beta0_sub = beta0

        # Re-compute initial fit for this subset
        Y0_sub, beta0_sub = initial_fit(Y_sub, X_sub, II_sub, force_int)

        if method in ("fe", "ife"):
            est = estimate_ife(
                Y_sub, Y0_sub, X_sub, II_sub, W_sub, beta0_sub,
                r=r_cv, force=force_int, tol=tol, max_iter=max_iter,
            )
        elif method == "mc":
            est = estimate_mc(
                Y_sub, Y0_sub, X_sub, II_sub, W_sub, beta0_sub,
                lam=lambda_cv, force=force_int, tol=tol, max_iter=max_iter,
            )
        else:
            est = estimate_ife(
                Y_sub, Y0_sub, X_sub, II_sub, W_sub, beta0_sub,
                r=r_cv or 0, force=force_int, tol=tol, max_iter=max_iter,
            )

        eff = to_numpy(Y_sub - est.fit)
        if normalize and norm_factor != 1.0:
            eff = eff * norm_factor

        return eff, to_numpy(D_sub), T_on_sub

    if vartype == "bootstrap":
        return bootstrap(
            _estimate_fn, to_numpy(Y_mat), to_numpy(panel.D),
            to_numpy(panel.I), panel.T_on, panel.unit_type,
            nboots=nboots, alpha=alpha, n_jobs=n_jobs, seed=seed,
        )
    else:
        return jackknife(
            _estimate_fn, to_numpy(Y_mat), to_numpy(panel.D),
            to_numpy(panel.I), panel.T_on, panel.unit_type,
            alpha=alpha, n_jobs=n_jobs,
        )
