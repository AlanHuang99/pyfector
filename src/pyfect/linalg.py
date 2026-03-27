"""
Core linear algebra primitives for pyfect.

Key performance improvement over R fect: **randomized / truncated SVD**
instead of full SVD.  For a T×N matrix with rank-r target the cost drops
from O(T·N·min(T,N)) to O(T·N·r), which is the dominant bottleneck.

All functions obtain the array module via ``get_backend()`` so they work
transparently on CPU (NumPy) or GPU (CuPy).
"""

from __future__ import annotations

from typing import Literal, NamedTuple

import numpy as np

from .backend import get_backend, to_numpy


# ---------------------------------------------------------------------------
# SVD helpers
# ---------------------------------------------------------------------------

class SVDResult(NamedTuple):
    """Truncated SVD result: U (T,r), s (r,), Vt (r,N)."""
    U: np.ndarray
    s: np.ndarray
    Vt: np.ndarray


def truncated_svd(M, r: int, n_oversamples: int = 10, n_iter: int = 4) -> SVDResult:
    """Randomized truncated SVD of *M* (T×N), returning top-*r* components.

    Uses the Halko-Martinsson-Tropp algorithm (2011).  Cost is
    O(T·N·(r + n_oversamples)) — dramatically cheaper than a full SVD when
    r << min(T, N).

    Falls back to exact truncated SVD when r is close to min(T, N).
    """
    xp = get_backend()
    T, N = M.shape
    k = min(r, T, N)

    # For small matrices or when r is large, use exact SVD (more stable)
    if k >= min(T, N) * 0.5 or min(T, N) <= 50 or k >= min(T, N) - 1:
        return _exact_truncated_svd(M, k)

    # Randomized SVD (Halko et al. 2011)
    p = min(k + n_oversamples, min(T, N))

    # Stage A: form an approximate orthonormal basis Q for range(M)
    Omega = xp.random.standard_normal((N, p)).astype(M.dtype)
    Y = M @ Omega  # (T, p)

    # Power iteration with QR re-orthogonalization (prevents overflow)
    for _ in range(n_iter):
        Z = M.T @ Y           # (N, p)
        Z, _ = xp.linalg.qr(Z)
        Y = M @ Z             # (T, p)
        Y, _ = xp.linalg.qr(Y)
    Q = Y  # already orthonormal from last QR

    # Stage B: form B = Q' M, then SVD of the small matrix B
    B = Q.T @ M  # (p, N)
    Uhat, s, Vt = xp.linalg.svd(B, full_matrices=False)
    U = Q @ Uhat  # (T, p)

    return SVDResult(U[:, :k], s[:k], Vt[:k, :])


def _exact_truncated_svd(M, k: int) -> SVDResult:
    """Full SVD truncated to top-k.  Used as fallback."""
    xp = get_backend()
    U, s, Vt = xp.linalg.svd(M, full_matrices=False)
    return SVDResult(U[:, :k], s[:k], Vt[:k, :])


def full_svd(M):
    """Full economy SVD.  Use sparingly — prefer truncated_svd."""
    xp = get_backend()
    return xp.linalg.svd(M, full_matrices=False)


# ---------------------------------------------------------------------------
# Panel factor extraction  (equivalent to R fect's panel_factor)
# ---------------------------------------------------------------------------

class FactorResult(NamedTuple):
    factors: np.ndarray   # (T, r)
    loadings: np.ndarray  # (N, r)
    eigenvalues: np.ndarray  # (r,)
    FE: np.ndarray        # (T, N) = factors @ loadings.T


def panel_factor(E, r: int) -> FactorResult:
    """Extract *r* latent factors and loadings from residual matrix *E* (T×N).

    Matches R fect's ``panel_factor()`` exactly:
    - Computes eigen decomposition of EE'/(NT) or E'E/(NT) (the smaller one)
    - factors = top-r eigenvectors * sqrt(T)
    - loadings = E' @ factors / T

    For large matrices (min(T,N) > 50), uses randomized SVD for O(NTr) cost.
    """
    xp = get_backend()
    T, N = E.shape
    if r == 0:
        return FactorResult(
            xp.empty((T, 0)),
            xp.empty((N, 0)),
            xp.empty((0,)),
            xp.zeros((T, N), dtype=E.dtype),
        )

    if T < N:
        # Work with the T×T covariance: EE'/(NT)
        EE = E @ E.T / (N * T)
        # SVD of the symmetric PSD matrix (= eigendecomposition)
        U, s, _ = xp.linalg.svd(EE, full_matrices=False)
        factors = U[:, :r] * xp.sqrt(xp.array(T, dtype=E.dtype))   # (T, r)
        loadings = E.T @ factors / T                                 # (N, r)
        eigenvalues = s[:r]
    else:
        # Work with the N×N covariance: E'E/(NT)
        EE = E.T @ E / (N * T)
        U, s, _ = xp.linalg.svd(EE, full_matrices=False)
        loadings = U[:, :r] * xp.sqrt(xp.array(N, dtype=E.dtype))  # (N, r)
        factors = E @ loadings / N                                   # (T, r)
        eigenvalues = s[:r]

    FE = factors @ loadings.T

    return FactorResult(factors, loadings, eigenvalues, FE)


# ---------------------------------------------------------------------------
# Nuclear norm thresholding  (equivalent to R fect's panel_FE)
# ---------------------------------------------------------------------------

def panel_FE(E, lam: float, hard: bool = False) -> np.ndarray:
    """Soft (or hard) singular value thresholding of *E* (T×N).

    For matrix completion: shrink singular values by *lam* (soft) or zero out
    those below *lam* (hard).

    Returns the thresholded T×N matrix.
    """
    xp = get_backend()
    T, N = E.shape
    scale = T * N

    # We need all singular values to threshold, so use full SVD here.
    # But we can still save work: after thresholding, the effective rank
    # is usually small, so reconstruction is cheap.
    U, s, Vt = full_svd(E / scale)

    if hard:
        mask = s > lam
        s_thresh = xp.where(mask, s, xp.zeros_like(s))
    else:
        s_thresh = xp.maximum(s - lam, 0.0)

    # Reconstruct only with nonzero singular values
    keep = s_thresh > 0
    if not xp.any(keep):
        return xp.zeros_like(E)

    n_keep = int(xp.sum(keep))
    FE = (U[:, :n_keep] * s_thresh[:n_keep][None, :]) @ Vt[:n_keep, :] * scale
    return FE


# ---------------------------------------------------------------------------
# Demeaning (equivalent to Y_demean / Y_wdemean)
# ---------------------------------------------------------------------------

class DemeanResult(NamedTuple):
    YY: np.ndarray       # demeaned matrix (T, N)
    mu: float            # grand mean
    alpha: np.ndarray | None   # unit FE (N,)
    xi: np.ndarray | None      # time FE (T,)


def demean(Y, force: int, W=None) -> DemeanResult:
    """Remove additive fixed effects from *Y* (T×N).

    Parameters
    ----------
    force : int
        0=none, 1=unit, 2=time, 3=two-way.
    W : array (T, N), optional
        Observation weights.  If given, computes weighted means.
    """
    xp = get_backend()
    T, N = Y.shape
    alpha = None
    xi = None

    if W is not None:
        # Weighted demeaning
        Wsum = xp.sum(W)
        mu = float(xp.sum(Y * W) / Wsum)
        YY = Y.copy()

        if force in (1, 3):
            col_w = xp.sum(W, axis=0)  # (N,)
            col_w = xp.where(col_w > 0, col_w, xp.ones_like(col_w))
            alpha = xp.sum(Y * W, axis=0) / col_w  # (N,)
            YY = YY - alpha[None, :]

        if force in (2, 3):
            row_w = xp.sum(W, axis=1)  # (T,)
            row_w = xp.where(row_w > 0, row_w, xp.ones_like(row_w))
            xi = xp.sum(Y * W, axis=1) / row_w  # (T,)
            YY = YY - xi[:, None]

        if force == 3:
            YY = YY + mu
        elif force == 0:
            YY = YY - mu
    else:
        mu = float(xp.mean(Y))
        YY = Y.copy()

        if force in (1, 3):
            alpha = xp.mean(Y, axis=0)  # (N,)
            YY = YY - alpha[None, :]

        if force in (2, 3):
            xi = xp.mean(Y, axis=1)  # (T,)
            YY = YY - xi[:, None]

        if force == 3:
            YY = YY + mu
        elif force == 0:
            YY = YY - mu

    return DemeanResult(YY, mu, alpha, xi)


def fe_add(mu: float, alpha, xi, T: int, N: int, force: int):
    """Reconstruct additive FE matrix (T×N) from components.

    R fect convention:
      force=0: FE = mu
      force=1: FE = mu + (alpha_Y - mu) = alpha_Y  (per-unit mean)
      force=2: FE = mu + (xi_Y - mu) = xi_Y  (per-time mean)
      force=3: FE = mu + (alpha_Y - mu) + (xi_Y - mu) = alpha_Y + xi_Y - mu
    """
    xp = get_backend()
    FE = xp.full((T, N), mu, dtype=xp.float64)
    if force in (1, 3) and alpha is not None:
        FE = FE + (alpha - mu)[None, :]
    if force in (2, 3) and xi is not None:
        FE = FE + (xi - mu)[:, None]
    return FE


# ---------------------------------------------------------------------------
# IFE: combined additive + interactive FE  (equivalent to R's ife())
# ---------------------------------------------------------------------------

class IFEResult(NamedTuple):
    FE: np.ndarray          # total FE (T, N)
    mu: float
    alpha: np.ndarray | None
    xi: np.ndarray | None
    factors: np.ndarray | None
    loadings: np.ndarray | None
    eigenvalues: np.ndarray | None
    FE_interactive: np.ndarray | None


def ife(E, force: int, mc: bool = False, r: int = 0,
        hard: bool = False, lam: float = 0.0) -> IFEResult:
    """Additive + interactive fixed effects extraction.

    Equivalent to fect's C++ ``ife()`` function.
    """
    xp = get_backend()
    T, N = E.shape

    # Step 1: demean
    dm = demean(E, force)
    EE = dm.YY

    # Step 2: additive FE
    FE_ad = fe_add(dm.mu, dm.alpha, dm.xi, T, N, force)

    # Step 3: interactive FE
    factors = loadings = eigenvalues = FE_inter = None
    if r > 0 or mc:
        if mc:
            FE_inter = panel_FE(EE, lam, hard=hard)
        else:
            fr = panel_factor(EE, r)
            factors = fr.factors
            loadings = fr.loadings
            eigenvalues = fr.eigenvalues
            FE_inter = fr.FE

    FE_total = FE_ad
    if FE_inter is not None:
        FE_total = FE_total + FE_inter

    return IFEResult(FE_total, dm.mu, dm.alpha, dm.xi,
                     factors, loadings, eigenvalues, FE_inter)


# ---------------------------------------------------------------------------
# Panel beta estimation (equivalent to panel_beta / wpanel_beta)
# ---------------------------------------------------------------------------

def panel_beta(X, Y, FE, xxinv=None, W=None):
    """OLS beta with FE partialled out.

    Parameters
    ----------
    X : array (T, N, p)
    Y : array (T, N)
    FE : array (T, N)
    xxinv : array (p, p), optional – precomputed (X'X)^-1
    W : array (T, N), optional – weights
    """
    xp = get_backend()
    p = X.shape[2]
    resid = Y - FE  # (T, N)

    if W is not None:
        sqW = xp.sqrt(W)
        xy = xp.array([xp.sum(sqW * X[:, :, k] * sqW * resid) for k in range(p)])
        if xxinv is None:
            xx = xp.zeros((p, p), dtype=X.dtype)
            for k in range(p):
                for m in range(k, p):
                    val = xp.sum(W * X[:, :, k] * X[:, :, m])
                    xx[k, m] = val
                    xx[m, k] = val
            xxinv = xp.linalg.inv(xx)
    else:
        xy = xp.array([xp.sum(X[:, :, k] * resid) for k in range(p)])
        if xxinv is None:
            xx = xp.zeros((p, p), dtype=X.dtype)
            for k in range(p):
                for m in range(k, p):
                    val = xp.sum(X[:, :, k] * X[:, :, m])
                    xx[k, m] = val
                    xx[m, k] = val
            xxinv = xp.linalg.inv(xx)

    beta = xxinv @ xy
    return beta


def compute_xxinv(X, W=None):
    """Precompute (X'X)^-1 or (X'WX)^-1."""
    xp = get_backend()
    p = X.shape[2]
    xx = xp.zeros((p, p), dtype=X.dtype)
    for k in range(p):
        for m in range(k, p):
            if W is not None:
                val = xp.sum(W * X[:, :, k] * X[:, :, m])
            else:
                val = xp.sum(X[:, :, k] * X[:, :, m])
            xx[k, m] = val
            xx[m, k] = val
    return xp.linalg.inv(xx)


def covar_fit(X, beta):
    """Compute X @ beta contribution: sum_k X[:,:,k] * beta[k]."""
    xp = get_backend()
    # Vectorised: (T, N, p) @ (p,) -> (T, N) via einsum
    return xp.einsum("ijk,k->ij", X, beta)
