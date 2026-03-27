"""
Shared fixtures for pyfect tests.

Generates simulated panel data with known DGP (data generating process)
matching fect's simulation approach:
  Y_it = mu + alpha_i + xi_t + lambda_i' f_t + X_it beta + D_it * delta + eps_it
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest


@pytest.fixture
def small_panel():
    """Small balanced panel: N=30, T=20, staggered adoption, r=2 factors."""
    return _simulate_panel(N=30, T=20, N_treated=10, r=2, seed=42)


@pytest.fixture
def medium_panel():
    """Medium balanced panel: N=200, T=50."""
    return _simulate_panel(N=200, T=50, N_treated=80, r=2, seed=123)


@pytest.fixture
def large_panel():
    """Large panel: N=2000, T=100 (pressure test)."""
    return _simulate_panel(N=2000, T=100, N_treated=800, r=2, seed=456)


@pytest.fixture
def unbalanced_panel():
    """Unbalanced panel with 20% missing observations."""
    return _simulate_panel(N=50, T=30, N_treated=20, r=2, seed=789,
                           missing_frac=0.2)


@pytest.fixture
def panel_with_covariates():
    """Panel with 3 covariates."""
    return _simulate_panel(N=50, T=30, N_treated=20, r=2, p=3, seed=101)


@pytest.fixture
def panel_with_reversals():
    """Panel where some treated units revert to control."""
    return _simulate_panel(N=40, T=25, N_treated=15, r=1, seed=202,
                           reversals=True)


@pytest.fixture
def zero_effect_panel():
    """Panel with zero treatment effect (for placebo tests)."""
    return _simulate_panel(N=50, T=30, N_treated=20, r=2, seed=303,
                           delta=0.0)


def _simulate_panel(
    N: int = 50,
    T: int = 30,
    N_treated: int = 20,
    r: int = 2,
    p: int = 0,
    delta: float = 3.0,
    seed: int = 42,
    missing_frac: float = 0.0,
    reversals: bool = False,
) -> dict:
    """Simulate panel data with interactive fixed effects DGP.

    DGP: Y_it = 1 + alpha_i + xi_t + lambda_i' f_t + X_it beta + D_it * delta + eps_it

    Returns dict with 'data' (pl.DataFrame), 'true_att' (float), and DGP params.
    """
    rng = np.random.default_rng(seed)

    # Fixed effects
    mu = 1.0
    alpha = rng.normal(0, 1, N)  # unit FE
    xi = rng.normal(0, 0.5, T)   # time FE

    # Factors and loadings
    factors = rng.normal(0, 1, (T, r))  # (T, r)
    loadings = rng.normal(0, 1, (N, r))  # (N, r)

    # Covariates
    beta = rng.normal(1, 0.5, p) if p > 0 else np.array([])
    X = rng.normal(0, 1, (T, N, p)) if p > 0 else None

    # Error
    eps = rng.normal(0, 0.5, (T, N))

    # Treatment assignment (staggered adoption)
    D = np.zeros((T, N))
    treat_start = np.full(N, T + 1)  # never-treated by default
    for j in range(N_treated):
        # Treatment starts between T/3 and 2T/3
        t0 = rng.integers(T // 3, 2 * T // 3 + 1)
        treat_start[j] = t0
        D[t0:, j] = 1.0

        if reversals and j < N_treated // 3:
            # Some units revert
            t_off = min(t0 + rng.integers(3, 8), T)
            D[t_off:, j] = 0.0

    # Outcome
    Y = mu + alpha[None, :] + xi[:, None] + factors @ loadings.T
    if p > 0:
        for k in range(p):
            Y = Y + X[:, :, k] * beta[k]
    Y = Y + D * delta + eps

    # Introduce missing data
    I = np.ones((T, N))
    if missing_frac > 0:
        n_missing = int(T * N * missing_frac)
        missing_idx = rng.choice(T * N, size=n_missing, replace=False)
        I.ravel()[missing_idx] = 0.0
        # Don't let treatment cells be missing
        for j in range(N):
            for t in range(T):
                if D[t, j] > 0:
                    I[t, j] = 1.0

    # Build long-format DataFrame
    rows = []
    for j in range(N):
        for t in range(T):
            if I[t, j] > 0:
                row = {
                    "unit": j,
                    "time": t,
                    "Y": float(Y[t, j]),
                    "D": int(D[t, j]),
                }
                if p > 0:
                    for k in range(p):
                        row[f"X{k+1}"] = float(X[t, j, k])
                rows.append(row)

    df = pl.DataFrame(rows)

    # True ATT
    treated_cells = D > 0
    true_att = delta  # homogeneous effect

    return {
        "data": df,
        "true_att": true_att,
        "delta": delta,
        "N": N,
        "T": T,
        "N_treated": N_treated,
        "r": r,
        "p": p,
        "beta": beta,
        "seed": seed,
        "covariate_names": [f"X{k+1}" for k in range(p)],
    }
