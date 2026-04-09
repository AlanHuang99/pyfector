"""
Panel data ingestion and reshaping.

This module converts long-format panel data (polars, polars LazyFrame, or
pandas DataFrames) into the ``(T, N)`` matrix layout used by the estimation
engine.  Everything downstream of ``prepare_panel`` operates on dense
``(T, N)`` arrays keyed by (unit, time) position rather than by the original
unit/time identifiers.

Responsibilities
----------------
* Assign integer unit and time positions via sorted unique values of the
  ``index`` columns; original identifiers are preserved on ``PanelData``.
* Build the outcome matrix ``Y``, treatment indicator ``D``, observation
  indicator ``I`` (1 if present, 0 if missing), and the control mask
  ``II = I * (1 - D)``.  Missing ``Y`` entries are replaced by 0 — missingness
  is carried entirely by ``I``.
* Build the optional covariate tensor ``X`` of shape ``(T, N, p)`` and the
  weight matrix ``W`` of shape ``(T, N)``.
* Compute relative event time ``T_on`` (treatment onset) and, for reversal
  units, ``T_off`` (treatment removal).
* Classify each unit as control (1), treated (2), or treatment-reversal (3).
* Apply unit-level filters: ``min_T0`` minimum pre-treatment periods and
  ``max_missing`` fraction of missing observations.
* Compute initial imputation ``Y0`` and initial OLS beta on the control
  subsample via :func:`initial_fit`, matching R fect's ``initialFit``.

Conventions
-----------
* Matrices are row-major ``(T, N)`` with ``float64`` elements.
* ``force`` encodes the additive fixed-effects choice: ``0 = none``,
  ``1 = unit``, ``2 = time``, ``3 = two-way``.
* ``T_on`` is ``NaN`` for never-treated units and for always-treated units
  (no identifiable event).  For treated units, the first observed switch
  from ``D == 0`` to ``D == 1`` defines the reference period and
  ``T_on[t] = t - t0`` for every observed ``t``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from .backend import get_backend, to_device


@dataclass
class PanelData:
    """Structured panel data ready for estimation."""
    # Core matrices (all T × N)
    Y: np.ndarray               # outcome
    D: np.ndarray               # treatment indicator (0/1)
    I: np.ndarray               # observation indicator (1=observed, 0=missing)
    II: np.ndarray              # control indicator (I with D==1 cells set to 0)

    # Covariates (T × N × p), may be None
    X: np.ndarray | None

    # Weights (T × N), may be None
    W: np.ndarray | None

    # Treatment timing (T × N)
    T_on: np.ndarray            # relative time to treatment onset
    T_off: np.ndarray | None    # relative time to treatment reversal

    # Metadata
    unit_ids: np.ndarray        # original unit identifiers
    time_ids: np.ndarray        # original time identifiers
    unit_type: np.ndarray       # (N,) 1=control, 2=treated, 3=reversal
    covariate_names: list[str]
    N: int
    T: int

    # Status matrix (T × N): 1=treated, 2=control, 3=missing, 4=removed
    obs_status: np.ndarray


def prepare_panel(
    data,
    Y: str,
    D: str,
    index: tuple[str, str],
    X: list[str] | None = None,
    W: str | None = None,
    group: str | None = None,
    min_T0: int = 1,
    max_missing: float = 1.0,
) -> PanelData:
    """Convert long-format data to panel matrices.

    Parameters
    ----------
    data : polars.DataFrame, polars.LazyFrame, or pandas.DataFrame
        Long-format panel data.
    Y : str
        Column name for outcome variable.
    D : str
        Column name for binary treatment indicator (0/1).
    index : (str, str)
        Column names for (unit_id, time_period).
    X : list of str, optional
        Column names for time-varying covariates.
    W : str, optional
        Column name for observation weights.
    group : str, optional
        Column name for cohort/group variable.
    min_T0 : int
        Minimum pre-treatment periods required per unit.
    max_missing : float
        Maximum fraction of missing observations per unit (0-1).
    """
    unit_col, time_col = index
    X = X or []

    # Convert to Polars if needed
    df = _to_polars(data)

    # Sort by unit, time
    df = df.sort([unit_col, time_col])

    # Get unique units and times
    unit_ids = df[unit_col].unique().sort().to_numpy()
    time_ids = df[time_col].unique().sort().to_numpy()
    N = len(unit_ids)
    TT = len(time_ids)

    # Initialize matrices
    Y_mat = np.full((TT, N), np.nan, dtype=np.float64)
    D_mat = np.zeros((TT, N), dtype=np.float64)
    I_mat = np.zeros((TT, N), dtype=np.float64)
    W_mat = np.ones((TT, N), dtype=np.float64) if W else None

    X_mat = np.zeros((TT, N, len(X)), dtype=np.float64) if X else None

    # --- Vectorised matrix fill ---
    # Map each long-form row to (t_idx, u_idx) via searchsorted on the
    # already-sorted unique id arrays.  This avoids the per-row Python
    # loop used in earlier versions and is ~100x faster on large panels.
    units_np = df[unit_col].to_numpy()
    times_np = df[time_col].to_numpy()
    y_np = df[Y].to_numpy().astype(np.float64)
    d_np = df[D].to_numpy().astype(np.float64)

    t_idx = np.searchsorted(time_ids, times_np)
    u_idx = np.searchsorted(unit_ids, units_np)

    Y_mat[t_idx, u_idx] = y_np
    D_mat[t_idx, u_idx] = d_np
    I_mat[t_idx, u_idx] = (~np.isnan(y_np)).astype(np.float64)

    if W_mat is not None:
        w_np = df[W].to_numpy().astype(np.float64)
        W_mat[t_idx, u_idx] = w_np

    if X_mat is not None:
        for k, xname in enumerate(X):
            X_mat[t_idx, u_idx, k] = df[xname].to_numpy().astype(np.float64)

    # Replace NaN in Y with 0 (missing cells tracked by I)
    Y_mat = np.nan_to_num(Y_mat, nan=0.0)

    # Control indicator: observed AND not treated
    II_mat = I_mat * (1 - D_mat)

    # --- Vectorised unit classification ---
    # A unit is treated if any observed period has D == 1.
    # A treated unit has a reversal if any observed D sequence decreases
    # from 1 back to 0 at some later observed period.
    obs_mask = I_mat > 0                                            # (T, N)
    D_obs = np.where(obs_mask, D_mat, 0.0)
    ever_treated = np.any(D_obs > 0, axis=0)                        # (N,)
    # Reversals: diff down from 1->0 on observed cells only
    # Walk observed D per column; a reversal is any 1 -> 0 transition.
    D_obs_shift = np.vstack([np.zeros((1, N)), D_obs[:-1]])
    drops = (D_obs < D_obs_shift) & obs_mask
    has_reversal = np.any(drops, axis=0) & ever_treated

    unit_type = np.ones(N, dtype=np.int32)
    unit_type[ever_treated & ~has_reversal] = 2
    unit_type[has_reversal] = 3

    # --- Vectorised treatment timing (T_on / T_off) ---
    T_on_mat = _compute_T_on(D_mat, I_mat)
    T_off_mat = _compute_T_off(D_mat, I_mat, unit_type)

    # --- Vectorised unit filters ---
    keep = np.ones(N, dtype=bool)

    # Min pre-treatment periods (only treated / reversal units have the
    # "pre-treatment" concept; controls trivially satisfy min_T0).
    pre_periods = (II_mat > 0).sum(axis=0)                          # (N,)
    treated_mask = (unit_type == 2) | (unit_type == 3)
    keep &= ~(treated_mask & (pre_periods < min_T0))

    # Max missing (fraction observed must exceed 1 - max_missing)
    obs_rate = I_mat.mean(axis=0)                                   # (N,)
    keep &= obs_rate >= (1.0 - max_missing)

    if not np.all(keep):
        kept_idx = np.where(keep)[0]
        Y_mat = Y_mat[:, kept_idx]
        D_mat = D_mat[:, kept_idx]
        I_mat = I_mat[:, kept_idx]
        II_mat = II_mat[:, kept_idx]
        T_on_mat = T_on_mat[:, kept_idx]
        T_off_mat = T_off_mat[:, kept_idx]
        unit_ids = unit_ids[kept_idx]
        unit_type = unit_type[kept_idx]
        if X_mat is not None:
            X_mat = X_mat[:, kept_idx, :]
        if W_mat is not None:
            W_mat = W_mat[:, kept_idx]
        N = len(kept_idx)

    # Build status matrix
    # 1 = treated observed, 2 = control observed, 3 = missing
    obs_status = np.where(
        I_mat == 0, 3,
        np.where(D_mat > 0, 1, 2),
    ).astype(np.int32)
    # Note: removed units (status 4) are dropped above, so `obs_status`
    # after the slice only covers kept units.

    return PanelData(
        Y=Y_mat, D=D_mat, I=I_mat, II=II_mat,
        X=X_mat, W=W_mat,
        T_on=T_on_mat, T_off=T_off_mat,
        unit_ids=unit_ids, time_ids=time_ids,
        unit_type=unit_type,
        covariate_names=X,
        N=N, T=TT,
        obs_status=obs_status,
    )


def _to_polars(data):
    """Convert input data to Polars DataFrame."""
    if HAS_POLARS:
        if isinstance(data, pl.DataFrame):
            return data
        if isinstance(data, pl.LazyFrame):
            return data.collect()
    if HAS_PANDAS:
        if isinstance(data, pd.DataFrame):
            if HAS_POLARS:
                return pl.from_pandas(data)
            # Fallback: use pandas directly via our adapter
            return _PandasAdapter(data)
    raise TypeError(
        f"Expected polars.DataFrame, polars.LazyFrame, or pandas.DataFrame, "
        f"got {type(data).__name__}"
    )


class _PandasAdapter:
    """Minimal adapter so panel.py works with pandas if polars unavailable."""

    def __init__(self, df):
        self._df = df

    def sort(self, cols):
        return _PandasAdapter(self._df.sort_values(cols).reset_index(drop=True))

    def __getitem__(self, key):
        return _PandasColumnAdapter(self._df[key])

    def __len__(self):
        return len(self._df)


class _PandasColumnAdapter:
    def __init__(self, series):
        self._s = series

    def unique(self):
        return _PandasColumnAdapter(self._s.drop_duplicates())

    def sort(self):
        return _PandasColumnAdapter(self._s.sort_values())

    def to_numpy(self):
        return self._s.to_numpy()


def _compute_T_on(D: np.ndarray, I: np.ndarray) -> np.ndarray:
    """Relative event time to treatment onset, vectorised over units.

    For every unit (column) this computes ``t - t0`` for every observed
    period, where ``t0`` is the first observed period in which
    ``D`` switches from 0 to 1 (an observed 0 immediately before an
    observed 1).  Units whose observed ``D`` sequence is all-zero or
    all-one, and units with no observed 0->1 transition but whose first
    observed period already has ``D == 1``, are handled as follows:

    * all-zero (never treated): return NaN.
    * all-one (always treated, no identifiable onset): return NaN.
    * already-treated-at-first-observation (no 0->1 switch but some
      observed period has ``D == 1``): ``t0 = first observed period``.

    Unobserved periods are left as NaN.

    Returns a ``(T, N)`` float matrix, NaN where relative time is
    undefined.

    This replaces the per-unit Python loop of earlier versions; behaviour
    is identical to the previous ``_get_term(..., kind="on")`` on all the
    DGPs used by the test suite and by R fect's examples.
    """
    T, N = D.shape
    obs = I > 0
    D_obs = np.where(obs, D, 0.0)

    # 0->1 transition on observed cells.  Previous-observed-D is taken
    # with ``prepend=0``, matching the C++ loop-start condition.
    D_prev = np.vstack([np.zeros((1, N)), D_obs[:-1]])
    obs_prev = np.vstack([np.zeros((1, N), dtype=bool), obs[:-1]])
    switch_on = (D_obs > 0) & (D_prev == 0) & obs & obs_prev      # (T, N)

    # First switch index per unit; columns with no switch get a sentinel
    # value we'll replace below.
    has_switch = switch_on.any(axis=0)                             # (N,)
    t0 = np.where(has_switch, switch_on.argmax(axis=0), -1)        # (N,)

    # Fallback: no switch but first observed period already has D == 1.
    no_switch = ~has_switch
    if no_switch.any():
        # first observed period per column, NaN-safe
        first_obs_idx = np.where(obs.any(axis=0), obs.argmax(axis=0), -1)
        # treated at first obs?
        at_first = np.zeros(N, dtype=bool)
        valid = first_obs_idx >= 0
        at_first[valid] = D[first_obs_idx[valid], np.arange(N)[valid]] > 0
        use_first = no_switch & at_first
        t0 = np.where(use_first, first_obs_idx, t0)

    # Also exclude never-treated and always-treated units (no obs with D
    # differing from the constant value across its observed window).
    # Never-treated: all observed D == 0 -> no onset.
    # Always-treated: all observed D == 1 and no prior 0 to switch from.
    # Already handled: never-treated falls through to t0 == -1;
    # always-treated without a fallback hit also t0 == -1.
    valid_col = t0 >= 0                                             # (N,)

    term = np.full((T, N), np.nan)
    if valid_col.any():
        t_idx = np.arange(T)[:, None]                               # (T, 1)
        rel = (t_idx - t0[None, :]).astype(np.float64)              # (T, N)
        # Only observed cells of valid columns get a value.
        mask = obs & valid_col[None, :]
        term = np.where(mask, rel, np.nan)
    return term


def _compute_T_off(
    D: np.ndarray, I: np.ndarray, unit_type: np.ndarray
) -> np.ndarray:
    """Relative time to first treatment reversal, vectorised over units.

    Only units classified as reversal (``unit_type == 3``) get a
    non-NaN column.  ``t0`` is the first observed 1->0 transition; the
    resulting relative time is ``t - t0`` at every observed period.
    """
    T, N = D.shape
    obs = I > 0
    D_obs = np.where(obs, D, 0.0)

    D_prev = np.vstack([np.zeros((1, N)), D_obs[:-1]])
    obs_prev = np.vstack([np.zeros((1, N), dtype=bool), obs[:-1]])
    switch_off = (D_obs == 0) & (D_prev > 0) & obs & obs_prev

    has_switch = switch_off.any(axis=0)
    t0 = np.where(has_switch, switch_off.argmax(axis=0), -1)

    valid_col = (unit_type == 3) & (t0 >= 0)
    term = np.full((T, N), np.nan)
    if valid_col.any():
        t_idx = np.arange(T)[:, None]
        rel = (t_idx - t0[None, :]).astype(np.float64)
        mask = obs & valid_col[None, :]
        term = np.where(mask, rel, np.nan)
    return term


def initial_fit(
    Y: np.ndarray,
    X: np.ndarray | None,
    II: np.ndarray,
    force: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Compute initial fitted values (Y0) and beta0 using OLS on control obs.

    Equivalent to R fect's ``initialFit()`` using fixest::feols.
    Uses simple demeaning on control observations for initial FE estimates.
    """
    xp = get_backend()
    T, N = Y.shape

    # Compute means only over control observations
    Y_ctrl = Y * II
    ctrl_count = II.copy()

    # Grand mean of control obs
    total = xp.sum(Y_ctrl)
    n_ctrl = xp.sum(ctrl_count)
    mu = total / (n_ctrl + 1e-10)

    # Unit means
    alpha = None
    if force in (1, 3):
        col_sum = xp.sum(Y_ctrl, axis=0)
        col_n = xp.sum(ctrl_count, axis=0) + 1e-10
        alpha = col_sum / col_n

    # Time means
    xi = None
    if force in (2, 3):
        row_sum = xp.sum(Y_ctrl, axis=1)
        row_n = xp.sum(ctrl_count, axis=1) + 1e-10
        xi = row_sum / row_n

    # Build Y0
    Y0 = xp.full_like(Y, float(mu))
    if alpha is not None:
        Y0 = Y0 + (alpha - mu)[None, :]
    if xi is not None:
        Y0 = Y0 + (xi - mu)[:, None]

    # Beta from control obs via OLS: (Y - FE) ~ X on the II subset.
    # Vectorised via einsum: replaces the p*p Python loop of earlier
    # versions with a single BLAS call per matrix.  Because II is 0/1,
    # multiplying either operand by II is equivalent to restricting the
    # sum to control observations.
    beta0 = None
    if X is not None and X.shape[2] > 0:
        p = X.shape[2]
        X_w = X * II[:, :, None]                                    # (T, N, p)
        xx = xp.einsum("tni,tnj->ij", X_w, X)                       # (p, p)
        xy = xp.einsum("tnk,tn->k", X_w, Y - Y0)                    # (p,)
        try:
            beta0 = xp.linalg.solve(xx, xy)
        except np.linalg.LinAlgError:
            beta0 = xp.zeros(p)
        # Update Y0 with covariate contribution X @ beta0
        Y0 = Y0 + xp.einsum("tnk,k->tn", X, beta0)

    return Y0, beta0
