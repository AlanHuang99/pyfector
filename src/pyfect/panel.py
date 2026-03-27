"""
Panel data ingestion and reshaping using Polars.

Converts long-format panel data (Polars or Pandas DataFrames) into the
matrix representation used by the estimation engine: T×N outcome matrix,
T×N treatment indicator, T×N×p covariate array, etc.

Also handles:
- Staggered adoption and treatment timing (T_on, T_off)
- Unbalanced panels (missing observations tracked via indicator I)
- Unit filtering (min pre-treatment periods, max missing obs)
- Cohort identification
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
    max_missing: float = 0.5,
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

    # Create unit/time index maps
    unit_map = {uid: i for i, uid in enumerate(unit_ids)}
    time_map = {tid: i for i, tid in enumerate(time_ids)}

    # Initialize matrices
    Y_mat = np.full((TT, N), np.nan, dtype=np.float64)
    D_mat = np.zeros((TT, N), dtype=np.float64)
    I_mat = np.zeros((TT, N), dtype=np.float64)
    W_mat = np.ones((TT, N), dtype=np.float64) if W else None

    X_mat = np.zeros((TT, N, len(X)), dtype=np.float64) if X else None

    # Fill matrices from data
    # Convert relevant columns to numpy for fast iteration
    units_np = df[unit_col].to_numpy()
    times_np = df[time_col].to_numpy()
    y_np = df[Y].to_numpy().astype(np.float64)
    d_np = df[D].to_numpy().astype(np.float64)

    w_np = df[W].to_numpy().astype(np.float64) if W else None
    x_nps = [df[xname].to_numpy().astype(np.float64) for xname in X]

    for row_idx in range(len(df)):
        u = unit_map[units_np[row_idx]]
        t = time_map[times_np[row_idx]]
        Y_mat[t, u] = y_np[row_idx]
        D_mat[t, u] = d_np[row_idx]
        I_mat[t, u] = 1.0 if not np.isnan(y_np[row_idx]) else 0.0
        if W_mat is not None:
            W_mat[t, u] = w_np[row_idx]
        if X_mat is not None:
            for k, x_arr in enumerate(x_nps):
                X_mat[t, u, k] = x_arr[row_idx]

    # Replace NaN in Y with 0 (missing cells tracked by I)
    Y_mat = np.nan_to_num(Y_mat, nan=0.0)

    # Control indicator: observed AND not treated
    II_mat = I_mat * (1 - D_mat)

    # Unit classification
    unit_type = np.ones(N, dtype=np.int32)  # default: control
    for j in range(N):
        d_col = D_mat[:, j]
        obs_col = I_mat[:, j]
        ever_treated = np.any(d_col[obs_col > 0] > 0)
        if ever_treated:
            # Check for reversals
            d_obs = d_col[obs_col > 0]
            has_reversal = False
            if len(d_obs) > 1:
                diffs = np.diff(d_obs)
                has_reversal = np.any(diffs < 0)
            unit_type[j] = 3 if has_reversal else 2

    # Compute treatment timing
    T_on_mat = np.full((TT, N), np.nan, dtype=np.float64)
    T_off_mat = np.full((TT, N), np.nan, dtype=np.float64)

    for j in range(N):
        T_on_mat[:, j] = _get_term(D_mat[:, j], I_mat[:, j], kind="on")
        if unit_type[j] == 3:
            T_off_mat[:, j] = _get_term(D_mat[:, j], I_mat[:, j], kind="off")

    # Filter units
    keep = np.ones(N, dtype=bool)

    # Min pre-treatment periods
    for j in range(N):
        if unit_type[j] in (2, 3):
            pre_periods = np.sum((II_mat[:, j] > 0))
            if pre_periods < min_T0:
                keep[j] = False

    # Max missing
    for j in range(N):
        obs_rate = np.sum(I_mat[:, j]) / TT
        if obs_rate < (1 - max_missing):
            keep[j] = False

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
    obs_status = np.where(
        I_mat == 0, 3,
        np.where(D_mat > 0, 1, 2),
    ).astype(np.int32)
    obs_status[:, ~keep[keep]] = 4  # removed units

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


def _get_term(
    d: np.ndarray,
    obs: np.ndarray,
    kind: str = "on",
) -> np.ndarray:
    """Compute relative event time for a single unit.

    Parameters
    ----------
    d : (T,) treatment indicator
    obs : (T,) observation indicator
    kind : "on" or "off"

    Returns
    -------
    term : (T,) relative time (negative before event, positive after)
    """
    T = len(d)
    term = np.full(T, np.nan)

    # Find treatment switches
    d_obs = d * obs  # only consider observed periods

    if kind == "on":
        # Find first treatment onset
        switches_on = []
        for t in range(1, T):
            if d_obs[t] > 0 and d_obs[t - 1] == 0 and obs[t] > 0 and obs[t - 1] > 0:
                switches_on.append(t)
        if len(switches_on) == 0:
            # Never treated or always treated
            if np.all(d_obs[obs > 0] > 0):
                # Always treated — no relative time
                return term
            elif np.all(d_obs[obs > 0] == 0):
                # Never treated — no relative time
                return term
            # Check if treated from the start
            first_obs = np.where(obs > 0)[0]
            if len(first_obs) > 0 and d[first_obs[0]] > 0:
                switches_on = [first_obs[0]]
            else:
                return term

        # Use first switch
        t0 = switches_on[0]
        for t in range(T):
            if obs[t] > 0:
                term[t] = t - t0  # negative before, 0 at onset, positive after

    elif kind == "off":
        # Find treatment reversal
        switches_off = []
        for t in range(1, T):
            if d_obs[t] == 0 and d_obs[t - 1] > 0 and obs[t] > 0 and obs[t - 1] > 0:
                switches_off.append(t)
        if len(switches_off) == 0:
            return term

        t0 = switches_off[0]
        for t in range(T):
            if obs[t] > 0:
                term[t] = t - t0

    return term


def initial_fit(
    Y: np.ndarray,
    X: np.ndarray | None,
    II: np.ndarray,
    force: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Compute initial fitted values (Y0) and beta0 using OLS on control obs.

    Equivalent to R fect's ``initialFit()`` using fixest::feols.
    Here we use a simpler approach: demean by unit/time FE on control obs,
    then regress residuals on X.
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
    if force == 3:
        Y0 = Y0 + mu

    # Beta from control obs
    beta0 = None
    if X is not None and X.shape[2] > 0:
        p = X.shape[2]
        resid = (Y - Y0) * II
        # Simple OLS on control obs: X'X beta = X'resid
        xx = xp.zeros((p, p))
        xy = xp.zeros(p)
        for k in range(p):
            for m in range(k, p):
                val = xp.sum(II * X[:, :, k] * X[:, :, m])
                xx[k, m] = val
                xx[m, k] = val
            xy[k] = xp.sum(II * X[:, :, k] * resid)
        try:
            beta0 = xp.linalg.solve(xx, xy)
        except np.linalg.LinAlgError:
            beta0 = xp.zeros(p)
        # Update Y0 with covariate contribution
        for k in range(p):
            Y0 = Y0 + X[:, :, k] * beta0[k]

    return Y0, beta0
