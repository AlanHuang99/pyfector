"""
Diagnostic tests on a fitted :class:`~pyfector.fect.FectResult`.

:func:`run_diagnostics` (also available as ``result.diagnose(...)``)
runs complementary checks on the pre-treatment portion of the event
study and on the post-treatment fit. The F-test and TOST defaults
follow Liu, Wang & Xu (2024):

* **Pre-trend F-test.** Wald-type joint significance of the
  pre-treatment ATTs against zero; small p-values cast doubt on the
  parallel-trends assumption.
* **Equivalence F-test.** Non-central F-test for whether all
  pre-treatment ATTs lie inside a user-supplied equivalence bound
  (``f_threshold``); large p-values provide positive evidence *for* no
  pre-trends.
* **TOST** (two one-sided t-tests) on each individual pre-treatment
  coefficient with bound ``tost_threshold``. The default convention
  follows Liu et al. (2024): ``0.36 * sqrt(sigma2_fect)``, where
  ``sigma2_fect`` is the residual variance of the requested additive
  fixed-effect baseline (not of the chosen MC/IFE/CFE estimator). Pass
  an explicit positive float for an absolute outcome-scale bound.
* **Placebo test.** Select pre-treatment placebo periods, remove those
  observed cells from the model-fitting mask, refit with the selected
  model configuration, and estimate the placebo ATT from the withheld
  observations. This is the Liu-Wang-Xu/R-``fect`` holdout/refit
  placebo procedure, not an average of already-estimated pre-period
  event-study coefficients.
* **Carryover test.** Look for residual effect after treatment turns
  off (reversal units only). Equivalence p deferred until
  ``att_off_boot`` is plumbed through the inference layer.
* **Leave-one-out (LOO).** Drop each post-treatment period in turn and
  report the maximum change in overall ATT.

Result shape
------------
``Diagnostics`` is a slim-safe registry of per-test result objects.
Known tests remain available through ergonomic properties such as
``diag.tost`` and ``diag.pretrend_f``, while future tests can be added
under ``diag.tests[name]`` without changing the envelope schema. The
old ``DiagnosticResult`` name remains as a compatibility alias.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TostResult:
    """Per-period TOST equivalence test on pre-treatment coefficients."""
    pvals: np.ndarray
    periods: np.ndarray
    threshold: float
    threshold_source: str
    sigma2_fect: float | None
    max_pval: float
    all_pass: bool

    def summary(self) -> str:
        lines = [
            f"TOST per pre-period (threshold={self.threshold:.4f}, "
            f"source={self.threshold_source}):"
        ]
        for i, t in enumerate(self.periods):
            p = float(self.pvals[i])
            status = "PASS" if p < 0.05 else "fail"
            lines.append(f"  t={t:+.0f}: p={p:.4f} [{status}]")
        lines.append(
            f"  max p={self.max_pval:.4f}  all_pass={self.all_pass}"
        )
        return "\n".join(lines)


@dataclass(frozen=True)
class PretrendFResult:
    """Joint F-test for all pre-treatment ATTs equal to zero."""
    f_stat: float
    p_value: float
    df1: int
    df2: int

    def summary(self) -> str:
        return (
            f"Pre-trend F-test:\n"
            f"  F({self.df1},{self.df2}) = {self.f_stat:.4f}, "
            f"p = {self.p_value:.4f}"
        )


@dataclass(frozen=True)
class EquivFResult:
    """Equivalence F-test (non-central F) for pre-trend bound."""
    p_value: float
    f_threshold: float

    def summary(self) -> str:
        return (
            f"Equivalence F-test:\n"
            f"  p = {self.p_value:.4f} (f_threshold={self.f_threshold})"
        )


@dataclass(frozen=True)
class PlaceboResult:
    """Holdout/refit placebo ATT with bootstrap and TOST p-values."""
    estimate: float
    se: float
    p_value: float
    equiv_p_value: float
    period: tuple[int, int]
    n_obs: int
    n_boot: int

    def summary(self) -> str:
        return (
            f"Placebo test (holdout/refit, window {self.period}):\n"
            f"  ATT={self.estimate:.4f} (SE={self.se:.4f}) "
            f"p={self.p_value:.4f} equiv_p={self.equiv_p_value:.4f}"
        )


@dataclass(frozen=True)
class CarryoverResult:
    """Carryover-window mean ATT (post-reversal). SE/p deferred."""
    estimate: float
    period: tuple[int, int]

    def summary(self) -> str:
        return (
            f"Carryover test (window {self.period}):\n"
            f"  ATT={self.estimate:.4f}"
        )


@dataclass(frozen=True)
class LooResult:
    """Leave-one-out post-period sensitivity."""
    atts: np.ndarray
    periods: np.ndarray
    max_change: float

    def summary(self) -> str:
        return (
            f"Leave-one-out:\n"
            f"  Max ATT change = {self.max_change:.6f}"
        )


_DIAGNOSTIC_SUMMARY_ORDER = (
    "pretrend_f", "equiv_f", "tost", "placebo", "carryover", "loo",
)


@dataclass
class Diagnostics:
    """Slim-safe registry of diagnostic test results.

    ``tests`` is the future-proof extension point. Built-in tests expose
    convenience properties for stable user code, but the container itself
    does not need a new field every time pyfector adds a diagnostic.
    """
    options: dict = field(default_factory=dict)
    tests: dict[str, Any] = field(default_factory=dict)

    @property
    def available(self) -> tuple[str, ...]:
        """Names of populated diagnostics, with built-ins first."""
        known = [name for name in _DIAGNOSTIC_SUMMARY_ORDER if name in self.tests]
        extra = sorted(name for name in self.tests if name not in set(known))
        return tuple(known + extra)

    def get(self, name: str, default: Any = None) -> Any:
        """Return a diagnostic by name, or ``default`` if absent."""
        return self.tests.get(name, default)

    def set_test(self, name: str, value: Any | None) -> None:
        """Set or remove a diagnostic result by name."""
        if not name or not isinstance(name, str):
            raise ValueError("diagnostic name must be a non-empty string")
        if value is None:
            self.tests.pop(name, None)
        else:
            self.tests[name] = value

    def set(self, name: str, value: Any | None) -> None:
        """Compatibility shortcut for :meth:`set_test`."""
        self.set_test(name, value)

    def __contains__(self, name: str) -> bool:
        return name in self.tests

    def __getitem__(self, name: str) -> Any:
        return self.tests[name]

    def _typed(self, name: str, typ: type) -> Any | None:
        value = self.tests.get(name)
        return value if isinstance(value, typ) else None

    @property
    def tost(self) -> TostResult | None:
        return self._typed("tost", TostResult)

    @tost.setter
    def tost(self, value: TostResult | None) -> None:
        self.set_test("tost", value)

    @property
    def pretrend_f(self) -> PretrendFResult | None:
        return self._typed("pretrend_f", PretrendFResult)

    @pretrend_f.setter
    def pretrend_f(self, value: PretrendFResult | None) -> None:
        self.set_test("pretrend_f", value)

    @property
    def equiv_f(self) -> EquivFResult | None:
        return self._typed("equiv_f", EquivFResult)

    @equiv_f.setter
    def equiv_f(self, value: EquivFResult | None) -> None:
        self.set_test("equiv_f", value)

    @property
    def placebo(self) -> PlaceboResult | None:
        return self._typed("placebo", PlaceboResult)

    @placebo.setter
    def placebo(self, value: PlaceboResult | None) -> None:
        self.set_test("placebo", value)

    @property
    def carryover(self) -> CarryoverResult | None:
        return self._typed("carryover", CarryoverResult)

    @carryover.setter
    def carryover(self, value: CarryoverResult | None) -> None:
        self.set_test("carryover", value)

    @property
    def loo(self) -> LooResult | None:
        return self._typed("loo", LooResult)

    @loo.setter
    def loo(self, value: LooResult | None) -> None:
        self.set_test("loo", value)

    def summary(self) -> str:
        lines = ["Diagnostic Tests", "=" * 50]
        for name in self.available:
            sub = self.tests[name]
            if sub is not None:
                if hasattr(sub, "summary"):
                    lines.append(sub.summary())
                else:
                    lines.append(f"{name}: {sub!r}")
        return "\n".join(lines)


DiagnosticResult = Diagnostics


_VALID_DIAG_NAMES = {"tost", "pretrend_f", "equiv_f", "placebo", "carryover", "loo"}
_VALID_OPTION_KEYS = {
    "tost_threshold", "f_threshold", "alpha",
    "placebo_period", "carryover_period", "loo",
}
_REQUIRED_TESTS = ("tost", "pretrend_f", "equiv_f")


def _is_positive_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and np.isfinite(value)
        and value > 0
    )


def _validate_tost_threshold_value(tost_threshold: Any) -> None:
    if tost_threshold is None:
        raise ValueError(
            "tost_threshold=None is not a valid value. Omit the argument "
            "for the Liu et al. (2024) default (0.36 * sqrt(sigma2_fect)) "
            "or pass a positive float for an explicit absolute bound."
        )
    if not _is_positive_finite_number(tost_threshold):
        raise ValueError(
            f"tost_threshold must be a positive finite float, got "
            f"{tost_threshold!r}"
        )


def _validate_period_option(name: str, value: Any) -> tuple[int, int]:
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or isinstance(value[0], bool)
        or isinstance(value[1], bool)
        or not np.isfinite(value[0])
        or not np.isfinite(value[1])
    ):
        raise ValueError(
            f"diagnostics_options['{name}'] must be a (start, end) tuple "
            "of finite numeric periods."
        )
    start = int(value[0])
    end = int(value[1])
    if start > end:
        raise ValueError(
            f"diagnostics_options['{name}'] must have start <= end."
        )
    return start, end


def _validate_placebo_period(value: Any) -> tuple[int, int]:
    start, end = _validate_period_option("placebo_period", value)
    if start >= 0:
        raise ValueError(
            "diagnostics_options['placebo_period'] must include at least "
            "one pre-treatment relative period (< 0)."
        )
    if end > 0:
        raise ValueError(
            "diagnostics_options['placebo_period'] must not extend into "
            "post-treatment periods (> 0)."
        )
    return start, end


def _resolve_tost_threshold(result, tost_threshold) -> tuple[float, str, float | None]:
    """Resolve the TOST equivalence bound.

    Returns (resolved_absolute_value, source_label, sigma2_fect_used).
    """
    if tost_threshold == 0.36:
        sigma2_fect = getattr(result, "sigma2_fect", 0.0) or 0.0
        if sigma2_fect <= 0.0:
            raise ValueError(
                "result.sigma2_fect is missing or zero; refit with "
                "pyfector >= 0.2.0 so sigma2_fect is populated, or pass "
                "tost_threshold=<positive float> for an explicit absolute "
                "bound."
            )
        return 0.36 * float(np.sqrt(sigma2_fect)), "auto", float(sigma2_fect)

    _validate_tost_threshold_value(tost_threshold)

    return float(tost_threshold), "explicit", None


def run_diagnostics(
    result,
    f_threshold: float = 0.5,
    tost_threshold: float = 0.36,
    placebo_period: tuple[int, int] | None = None,
    carryover_period: tuple[int, int] | None = None,
    loo: bool = False,
    alpha: float = 0.05,
    *,
    _requested: list[str] | None = None,
) -> Diagnostics:
    """Run diagnostic tests on a FectResult.

    Parameters
    ----------
    result : FectResult
        Must have inference results (``se=True``).
    f_threshold : float
        Non-centrality parameter for equivalence F-test.
    tost_threshold : float
        Equivalence bound for TOST. The literal ``0.36`` (default or
        explicit) triggers Liu et al. (2024)'s scale-aware bound,
        ``0.36 * sqrt(result.sigma2_fect)``. Any other positive float
        is taken as an absolute outcome-scale bound. ``None`` is
        invalid.
    placebo_period : (start, end), optional
        Relative pre-treatment window for the holdout/refit placebo test.
        Observed cells with ``start <= time_on <= end`` and ``time_on < 0``
        are removed from the fitting mask, the model is refit using the
        selected rank/lambda configuration, and the placebo ATT is
        computed from those withheld cells. For example, ``(-3, 0)``
        withholds relative periods ``-3, -2, -1``.
    carryover_period : (start, end), optional
        Relative time window for carryover test.
    loo : bool
        If True, run leave-one-out post-period sensitivity.
    alpha : float
        Significance cutoff used for ``TostResult.all_pass``.
    """
    from scipy import stats

    diag = Diagnostics()

    if result.inference is None:
        return diag

    inf = result.inference
    time_on = result.time_on
    att_on = result.att_on

    if time_on is None or att_on is None:
        return diag

    pre_mask = time_on < 0
    if not np.any(pre_mask):
        return diag

    pre_idx = np.where(pre_mask)[0]
    k = len(pre_idx)
    att_pre = att_on[pre_idx]

    requested = set(_requested) if _requested is not None else _VALID_DIAG_NAMES.copy()
    if placebo_period is not None:
        placebo_period = _validate_placebo_period(placebo_period)

    needs_tost_threshold = bool({"tost", "placebo"} & requested)
    threshold_abs = float("nan")
    threshold_source: str | None = None
    sigma2_fect_used: float | None = None
    if needs_tost_threshold:
        threshold_abs, threshold_source, sigma2_fect_used = _resolve_tost_threshold(
            result, tost_threshold
        )

    # Pre-trend / equivalence F-tests share the bootstrap covariance pass.
    if (
        ("pretrend_f" in requested or "equiv_f" in requested)
        and inf.att_on_boot is not None
    ):
        boot_pre_all = inf.att_on_boot[pre_idx, :]
        valid_boot = np.all(np.isfinite(boot_pre_all), axis=0)
        boot_pre = boot_pre_all[:, valid_boot]
        n_boot = boot_pre.shape[1]
        if n_boot <= k:
            boot_pre = None

    else:
        boot_pre = None

    if boot_pre is not None:
        S = np.cov(boot_pre)
        if k == 1:
            S = np.asarray(S).reshape(1, 1)

        try:
            cond = np.linalg.cond(S)
            S_inv = np.linalg.pinv(S) if cond > 1e12 else np.linalg.inv(S)
            F_raw = float(att_pre @ S_inv @ att_pre)
            scale = (n_boot - k) / ((n_boot - 1) * k)
            F_stat = F_raw * scale

            if np.isfinite(F_stat) and F_stat >= 0:
                if "pretrend_f" in requested:
                    diag.pretrend_f = PretrendFResult(
                        f_stat=F_stat,
                        p_value=float(1 - stats.f.cdf(F_stat, k, n_boot - k)),
                        df1=k,
                        df2=n_boot - k,
                    )
                if "equiv_f" in requested:
                    ncp = n_boot * f_threshold
                    diag.equiv_f = EquivFResult(
                        p_value=float(stats.ncf.cdf(F_stat, k, n_boot - k, ncp)),
                        f_threshold=f_threshold,
                    )
        except np.linalg.LinAlgError:
            pass

    # Per-period TOST.
    if "tost" in requested and inf.att_on_se is not None:
        se_pre = inf.att_on_se[pre_idx]

        tost_pvals = np.full(k, np.nan)
        for i in range(k):
            if se_pre[i] > 0:
                if inf.att_on_boot is not None:
                    n_boot_i = int(np.isfinite(inf.att_on_boot[pre_idx[i], :]).sum())
                else:
                    n_boot_i = 200
                df = max(n_boot_i - 1, 1)
                t_upper = (att_pre[i] - threshold_abs) / se_pre[i]
                t_lower = (att_pre[i] + threshold_abs) / se_pre[i]
                p_upper = float(stats.t.cdf(t_upper, df))
                p_lower = float(1 - stats.t.cdf(t_lower, df))
                tost_pvals[i] = max(p_upper, p_lower)

        finite = tost_pvals[np.isfinite(tost_pvals)]
        max_p = float(finite.max()) if finite.size else float("nan")
        all_pass = bool(finite.size and (finite < alpha).all())

        diag.tost = TostResult(
            pvals=tost_pvals,
            periods=time_on[pre_idx].copy(),
            threshold=threshold_abs,
            threshold_source=threshold_source,
            sigma2_fect=sigma2_fect_used,
            max_pval=max_p,
            all_pass=all_pass,
        )

    # Holdout/refit placebo test. This follows R fect's placeboTest path:
    # selected pre-treatment cells are removed from II before fitting.
    if "placebo" in requested and placebo_period is not None:
        diag.placebo = _run_placebo_test(
            result,
            placebo_period=placebo_period,
            threshold_abs=threshold_abs,
        )

    # Carryover (estimate only; bootstrap p deferred).
    if (
        "carryover" in requested
        and carryover_period is not None
        and hasattr(result, "att_off")
        and result.att_off is not None
    ):
        c_start, c_end = carryover_period
        time_off = getattr(result, "time_off", None)
        if time_off is not None:
            off_mask = (time_off >= c_start) & (time_off <= c_end)
            if np.any(off_mask):
                diag.carryover = CarryoverResult(
                    estimate=float(np.mean(result.att_off[off_mask])),
                    period=(int(c_start), int(c_end)),
                )

    # Leave-one-out post-period.
    if "loo" in requested and loo:
        post_mask = time_on >= 0
        post_idx = np.where(post_mask)[0]
        if len(post_idx) > 1:
            full_att = result.att_avg
            loo_atts: list[float] = []
            loo_periods: list[float] = []
            for drop_i in post_idx:
                remaining = np.delete(post_idx, np.where(post_idx == drop_i))
                if len(remaining) > 0:
                    loo_atts.append(float(np.mean(att_on[remaining])))
                    loo_periods.append(float(time_on[drop_i]))
            atts_arr = np.array(loo_atts)
            diag.loo = LooResult(
                atts=atts_arr,
                periods=np.array(loo_periods),
                max_change=float(np.max(np.abs(atts_arr - full_att))),
            )

    diag.options = {
        "tost_threshold": threshold_abs if needs_tost_threshold else None,
        "tost_threshold_source": threshold_source,
        "f_threshold": f_threshold,
        "alpha": alpha,
        "placebo_period": placebo_period,
        "carryover_period": carryover_period,
        "loo": loo,
        "sigma2_fect": sigma2_fect_used if sigma2_fect_used is not None
                       else getattr(result, "sigma2_fect", None),
    }
    try:
        from . import __version__ as _ver
        diag.options["pyfector_version"] = _ver
    except Exception:
        pass

    return diag


def _selected_boot_count(result) -> int:
    inf = getattr(result, "inference", None)
    boot = getattr(inf, "att_on_boot", None)
    if boot is not None and getattr(boot, "ndim", 0) == 2 and boot.shape[1] > 0:
        return int(boot.shape[1])
    avg_boot = getattr(inf, "att_avg_boot", None)
    if avg_boot is not None and len(avg_boot) > 0:
        return int(len(avg_boot))
    opts = getattr(result, "fit_options", {}) or {}
    return int(opts.get("nboots", 200))


def _estimate_with_selected_config(
    method: str,
    Y,
    Y0,
    X,
    II,
    W,
    beta0,
    *,
    r_cv: int | None,
    lambda_cv: float | None,
    force_int: int,
    tol: float,
    max_iter: int,
):
    from .estimators import estimate_cfe, estimate_ife, estimate_mc

    if method in ("fe", "ife", "both"):
        return estimate_ife(
            Y, Y0, X, II, W, beta0,
            r=r_cv or 0, force=force_int, tol=tol, max_iter=max_iter,
        )
    if method == "mc":
        return estimate_mc(
            Y, Y0, X, II, W, beta0,
            lam=0.0 if lambda_cv is None else lambda_cv,
            force=force_int, tol=tol, max_iter=max_iter,
        )
    if method == "cfe":
        return estimate_cfe(
            Y, Y0, X, II, W, beta0,
            r=r_cv or 0, force=force_int, tol=tol, max_iter=max_iter,
        )
    raise ValueError(f"Unknown method for placebo test: {method!r}")


def _run_placebo_test(
    result,
    *,
    placebo_period: tuple[int, int],
    threshold_abs: float,
) -> PlaceboResult:
    """Run the holdout/refit placebo test used by Liu-Wang-Xu/R fect."""
    from joblib import Parallel, delayed
    from scipy import stats

    from .backend import to_device, to_numpy
    from .panel import initial_fit

    panel = getattr(result, "panel", None)
    if panel is None:
        raise ValueError(
            "placebo_period requires result.panel because the placebo "
            "test refits on a modified fitting mask."
        )

    opts = getattr(result, "fit_options", {}) or {}
    if "force_int" not in opts:
        raise ValueError(
            "placebo_period requires fit metadata from pyfector.fect(). "
            "Refit the model before running the placebo test."
        )

    p_start, p_end = placebo_period
    holdout_mask = (
        (panel.I > 0)
        & np.isfinite(panel.T_on)
        & (panel.T_on >= p_start)
        & (panel.T_on <= p_end)
        & (panel.T_on < 0)
    )
    n_obs = int(np.sum(holdout_mask))
    if n_obs == 0:
        raise ValueError(
            "placebo_period selects no observed pre-treatment cells."
        )

    II_holdout = panel.II.copy()
    II_holdout[holdout_mask] = 0.0

    normalize = bool(opts.get("normalize", False))
    norm_factor = float(opts.get("norm_factor", 1.0) or 1.0)
    if normalize and norm_factor != 1.0:
        Y_np = panel.Y / norm_factor
    else:
        Y_np = panel.Y

    force_int = int(opts["force_int"])
    tol = float(opts.get("tol", 1e-7))
    boot_tol = max(tol, 1e-3)
    max_iter = int(opts.get("max_iter", 5000))
    method = getattr(result, "method")
    r_cv = getattr(result, "r_cv", None)
    lambda_cv = getattr(result, "lambda_cv", None)

    def _fit_placebo(unit_idx, tol_value):
        Y_sub_np = Y_np[:, unit_idx]
        II_sub_np = II_holdout[:, unit_idx]
        X_sub_np = panel.X[:, unit_idx, :] if panel.X is not None else None
        W_sub_np = panel.W[:, unit_idx] if panel.W is not None else None
        mask_sub = holdout_mask[:, unit_idx]
        if not np.any(mask_sub):
            return None

        Y_sub = to_device(Y_sub_np)
        II_sub = to_device(II_sub_np)
        X_sub = to_device(X_sub_np) if X_sub_np is not None else None
        W_sub = to_device(W_sub_np) if W_sub_np is not None else None

        Y0_sub, beta0_sub = initial_fit(Y_sub, X_sub, II_sub, force_int)
        est = _estimate_with_selected_config(
            method,
            Y_sub,
            Y0_sub,
            X_sub,
            II_sub,
            W_sub,
            beta0_sub,
            r_cv=r_cv,
            lambda_cv=lambda_cv,
            force_int=force_int,
            tol=tol_value,
            max_iter=max_iter,
        )
        eff_sub = to_numpy(Y_sub - est.fit)
        if normalize and norm_factor != 1.0:
            eff_sub = eff_sub * norm_factor
        return float(np.mean(eff_sub[mask_sub]))

    point = _fit_placebo(np.arange(panel.N), tol)
    if point is None:
        raise ValueError(
            "placebo_period selects no observed pre-treatment cells."
        )

    nboots = _selected_boot_count(result)
    n_jobs = int(opts.get("n_jobs", 1) or 1)
    if n_jobs == 0:
        n_jobs = 1

    seed = getattr(result, "seed", None)
    rng = np.random.default_rng(seed)
    treated_idx = np.where(panel.unit_type == 2)[0]
    control_idx = np.where(panel.unit_type == 1)[0]
    reversal_idx = np.where(panel.unit_type == 3)[0]
    n_tr = len(treated_idx)
    n_co = len(control_idx)
    n_rev = len(reversal_idx)

    def _one_boot(boot_seed):
        boot_rng = np.random.default_rng(boot_seed)
        tr_sample = (
            boot_rng.choice(treated_idx, size=n_tr, replace=True)
            if n_tr > 0 else np.array([], dtype=int)
        )
        co_sample = (
            boot_rng.choice(control_idx, size=n_co, replace=True)
            if n_co > 0 else np.array([], dtype=int)
        )
        rev_sample = (
            boot_rng.choice(reversal_idx, size=n_rev, replace=True)
            if n_rev > 0 else np.array([], dtype=int)
        )
        unit_idx = np.concatenate([tr_sample, co_sample, rev_sample]).astype(int)
        try:
            return _fit_placebo(unit_idx, boot_tol)
        except Exception:
            return None

    boot_seeds = rng.integers(0, 2**31, size=nboots)
    if n_jobs == 1:
        boot_results = [_one_boot(s) for s in boot_seeds]
    else:
        boot_results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_one_boot)(s) for s in boot_seeds
        )
    boot = np.array(
        [b for b in boot_results if b is not None and np.isfinite(b)],
        dtype=float,
    )

    se = float("nan")
    p_value = float("nan")
    equiv_p = float("nan")
    if boot.size > 1:
        se = float(np.std(boot, ddof=1))
        if se > 0:
            # Match R fect's default placebo-test summary: a normal
            # approximation from the bootstrap SE, not a percentile p-value.
            p_value = float(2 * stats.norm.sf(abs(point / se)))
            z_upper = (point - threshold_abs) / se
            z_lower = (point + threshold_abs) / se
            equiv_p = max(
                float(stats.norm.cdf(z_upper)),
                float(1 - stats.norm.cdf(z_lower)),
            )

    return PlaceboResult(
        estimate=float(point),
        se=se,
        p_value=p_value,
        equiv_p_value=equiv_p,
        period=(int(p_start), int(p_end)),
        n_obs=n_obs,
        n_boot=int(boot.size),
    )


def validate_diagnostics_request(
    diagnostics: Any,
    diagnostics_options: dict | None,
    se: bool,
) -> list[str] | None:
    """Validate a fect()-side diagnostics request and return the resolved
    list of test names to run (or ``None`` if ``diagnostics='none'``).

    Raises ``ValueError`` for any invalid combination, all before any
    expensive estimation runs.
    """
    opts = diagnostics_options or {}
    bad_keys = set(opts.keys()) - _VALID_OPTION_KEYS
    if bad_keys:
        raise ValueError(
            f"unknown diagnostics_options key(s): {sorted(bad_keys)}; "
            f"valid: {sorted(_VALID_OPTION_KEYS)}"
        )

    if diagnostics == "none":
        return None

    if not se:
        raise ValueError(
            "diagnostics requires se=True; pre-trend F-test, TOST, and "
            "placebo p-values all need bootstrap covariance."
        )

    if diagnostics == "full":
        requested = list(_REQUIRED_TESTS)
        if "placebo_period" in opts:
            requested.append("placebo")
        if "carryover_period" in opts:
            requested.append("carryover")
        if opts.get("loo", False):
            requested.append("loo")
    elif isinstance(diagnostics, list):
        if len(diagnostics) == 0:
            raise ValueError(
                "diagnostics=[] is invalid; pass 'none' or a non-empty list."
            )
        bad = [d for d in diagnostics if d not in _VALID_DIAG_NAMES]
        if bad:
            raise ValueError(
                f"Unknown diagnostic(s): {bad}; valid: "
                f"{sorted(_VALID_DIAG_NAMES)}"
            )
        if "placebo" in diagnostics and "placebo_period" not in opts:
            raise ValueError(
                "diagnostics list includes 'placebo' but "
                "diagnostics_options['placebo_period'] is missing."
            )
        if "carryover" in diagnostics and "carryover_period" not in opts:
            raise ValueError(
                "diagnostics list includes 'carryover' but "
                "diagnostics_options['carryover_period'] is missing."
            )
        requested = list(diagnostics)
    else:
        raise ValueError(
            f"diagnostics must be 'none', 'full', or a list of names; got "
            f"{diagnostics!r}"
        )

    if "alpha" in opts:
        alpha = opts["alpha"]
        if (
            not isinstance(alpha, (int, float, np.integer, np.floating))
            or isinstance(alpha, (bool, np.bool_))
            or not np.isfinite(alpha)
            or not 0 < alpha < 1
        ):
            raise ValueError(
                "diagnostics_options['alpha'] must be a finite float in (0, 1)."
            )

    if "f_threshold" in opts and not _is_positive_finite_number(opts["f_threshold"]):
        raise ValueError(
            "diagnostics_options['f_threshold'] must be a positive finite float."
        )

    if "tost_threshold" in opts and {"tost", "placebo"} & set(requested):
        _validate_tost_threshold_value(opts["tost_threshold"])

    if "placebo_period" in opts:
        _validate_placebo_period(opts["placebo_period"])
    if "carryover_period" in opts:
        _validate_period_option("carryover_period", opts["carryover_period"])

    if "loo" in opts and not isinstance(opts["loo"], (bool, np.bool_)):
        raise ValueError("diagnostics_options['loo'] must be a bool.")

    return requested
