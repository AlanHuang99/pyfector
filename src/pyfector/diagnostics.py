"""
Diagnostic tests for pyfector results.

Implements:
- Pre-trend F-test (Wald-type joint significance)
- Equivalence F-test (against non-central F)
- TOST (two one-sided t-tests) for each pre-treatment period
- Placebo test
- Carryover test
- Leave-one-period-out (LOO) test
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DiagnosticResult:
    """Container for diagnostic test results."""
    # F-test for pre-trends
    f_stat: float | None = None
    f_pval: float | None = None
    f_df1: int | None = None
    f_df2: int | None = None

    # Equivalence F-test
    equiv_f_pval: float | None = None
    equiv_threshold: float | None = None

    # TOST per pre-period
    tost_pvals: np.ndarray | None = None
    tost_periods: np.ndarray | None = None
    tost_threshold: float | None = None

    # Placebo
    placebo_att: float | None = None
    placebo_pval: float | None = None

    # Carryover
    carryover_att: float | None = None
    carryover_pval: float | None = None

    # Leave-one-out
    loo_atts: np.ndarray | None = None      # ATT when each pre-period is dropped
    loo_periods: np.ndarray | None = None
    loo_max_change: float | None = None     # max |ATT_full - ATT_loo|

    def summary(self) -> str:
        lines = ["Diagnostic Tests", "=" * 50]
        if self.f_stat is not None:
            lines.append(f"Pre-trend F-test:")
            lines.append(f"  F({self.f_df1},{self.f_df2}) = {self.f_stat:.4f}, p = {self.f_pval:.4f}")
        if self.equiv_f_pval is not None:
            lines.append(f"Equivalence F-test:")
            lines.append(f"  p = {self.equiv_f_pval:.4f} (threshold={self.equiv_threshold})")
        if self.tost_pvals is not None:
            lines.append("TOST per pre-period:")
            for i, t in enumerate(self.tost_periods):
                p = self.tost_pvals[i]
                status = "PASS" if p < 0.05 else "fail"
                lines.append(f"  t={t:+.0f}: p={p:.4f} [{status}]")
        if self.placebo_pval is not None:
            lines.append(f"Placebo test:")
            lines.append(f"  ATT={self.placebo_att:.4f}, p={self.placebo_pval:.4f}")
        if self.carryover_pval is not None:
            lines.append(f"Carryover test:")
            lines.append(f"  ATT={self.carryover_att:.4f}, p={self.carryover_pval:.4f}")
        if self.loo_max_change is not None:
            lines.append(f"Leave-one-out:")
            lines.append(f"  Max ATT change = {self.loo_max_change:.6f}")
        return "\n".join(lines)


def run_diagnostics(
    result,
    f_threshold: float = 0.5,
    tost_threshold: float = 0.36,
    placebo_period: tuple[int, int] | None = None,
    carryover_period: tuple[int, int] | None = None,
    loo: bool = False,
) -> DiagnosticResult:
    """Run diagnostic tests on a FectResult.

    Parameters
    ----------
    result : FectResult
        Must have inference results (``se=True``).
    f_threshold : float
        Non-centrality parameter for equivalence F-test.
    tost_threshold : float
        Equivalence bound for TOST.
    placebo_period : (start, end), optional
        Relative time window for placebo test.
    """
    from scipy import stats

    diag = DiagnosticResult()

    if result.inference is None:
        return diag

    inf = result.inference
    time_on = result.time_on
    att_on = result.att_on

    if time_on is None or att_on is None:
        return diag

    # Pre-treatment periods
    pre_mask = time_on < 0
    if not np.any(pre_mask):
        return diag

    pre_idx = np.where(pre_mask)[0]
    k = len(pre_idx)
    att_pre = att_on[pre_idx]

    # F-test from bootstrap covariance
    if inf.att_on_boot is not None and inf.att_on_boot.shape[1] > k:
        boot_pre = inf.att_on_boot[pre_idx, :]  # (k, nboots)
        S = np.cov(boot_pre)  # (k, k)
        if k == 1:
            S = S.reshape(1, 1)

        n_boot = boot_pre.shape[1]
        try:
            cond = np.linalg.cond(S)
            if cond > 1e12:
                S_inv = np.linalg.pinv(S)
            else:
                S_inv = np.linalg.inv(S)
            F_stat = float(att_pre @ S_inv @ att_pre)
            scale = (n_boot - k) / ((n_boot - 1) * k)
            F_stat_scaled = F_stat * scale

            if np.isfinite(F_stat_scaled) and F_stat_scaled >= 0:
                diag.f_stat = F_stat_scaled
                diag.f_df1 = k
                diag.f_df2 = n_boot - k
                diag.f_pval = float(1 - stats.f.cdf(F_stat_scaled, k, n_boot - k))

                # Equivalence F-test (non-central F)
                ncp = n_boot * f_threshold
                diag.equiv_f_pval = float(stats.ncf.cdf(F_stat_scaled, k, n_boot - k, ncp))
                diag.equiv_threshold = f_threshold
        except np.linalg.LinAlgError:
            pass

    # TOST
    if inf.att_on_se is not None:
        se_pre = inf.att_on_se[pre_idx]
        n_boot = inf.att_on_boot.shape[1] if inf.att_on_boot is not None else 200
        df = n_boot - 1

        tost_pvals = np.full(k, np.nan)
        for i in range(k):
            if se_pre[i] > 0:
                t_upper = (att_pre[i] - tost_threshold) / se_pre[i]
                t_lower = (att_pre[i] + tost_threshold) / se_pre[i]
                p_upper = float(stats.t.cdf(t_upper, df))
                p_lower = float(1 - stats.t.cdf(t_lower, df))
                tost_pvals[i] = max(p_upper, p_lower)

        diag.tost_pvals = tost_pvals
        diag.tost_periods = time_on[pre_idx]
        diag.tost_threshold = tost_threshold

    # Placebo test
    if placebo_period is not None:
        p_start, p_end = placebo_period
        placebo_mask = (time_on >= p_start) & (time_on <= p_end) & pre_mask
        if np.any(placebo_mask):
            p_idx = np.where(placebo_mask)[0]
            diag.placebo_att = float(np.mean(att_on[p_idx]))
            if inf.att_on_boot is not None:
                boot_placebo = np.mean(inf.att_on_boot[p_idx, :], axis=0)
                diag.placebo_pval = min(
                    2 * np.mean(boot_placebo >= 0),
                    2 * np.mean(boot_placebo <= 0),
                    1.0,
                )

    # Carryover test (for treatment reversals)
    if carryover_period is not None and hasattr(result, "att_off") and result.att_off is not None:
        c_start, c_end = carryover_period
        time_off = getattr(result, "time_off", None)
        if time_off is not None:
            off_mask = (time_off >= c_start) & (time_off <= c_end)
            if np.any(off_mask):
                diag.carryover_att = float(np.mean(result.att_off[off_mask]))

    # Leave-one-period-out test
    if loo and att_on is not None and time_on is not None:
        post_mask = time_on >= 0
        post_idx = np.where(post_mask)[0]
        if len(post_idx) > 1:
            full_att = result.att_avg
            loo_atts = []
            loo_periods = []
            for drop_i in post_idx:
                remaining = np.delete(post_idx, np.where(post_idx == drop_i))
                if len(remaining) > 0:
                    loo_att = float(np.mean(att_on[remaining]))
                    loo_atts.append(loo_att)
                    loo_periods.append(time_on[drop_i])
            diag.loo_atts = np.array(loo_atts)
            diag.loo_periods = np.array(loo_periods)
            diag.loo_max_change = float(np.max(np.abs(diag.loo_atts - full_att)))

    return diag
