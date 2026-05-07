"""
Diagnostic tests on a fitted :class:`~pyfector.fect.FectResult`.

:func:`run_diagnostics` (also available as ``result.diagnose(...)``)
runs several complementary checks on the pre-treatment portion of the
event study and on the post-treatment fit. Tests are implemented as in
Liu, Wang & Xu (2024):

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
* **TOST on placebo.** When ``placebo_period`` is supplied, the
  bootstrap mean of pre-period ATTs in the window is also tested for
  equivalence using the same (auto or explicit) ``tost_threshold``.
* **Placebo bootstrap p-value.** Two-sided percentile p for the same
  window (kept alongside the new equivalence p).
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
    """Placebo window mean ATT plus bootstrap percentile and TOST p-values."""
    estimate: float
    se: float
    p_value: float
    equiv_p_value: float
    period: tuple[int, int]

    def summary(self) -> str:
        return (
            f"Placebo test (window {self.period}):\n"
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
        Relative time window for placebo test. Restricted to
        pre-treatment event times: ``(time_on >= start) &
        (time_on <= end) & (time_on < 0)``. So ``(-3, 0)`` selects
        rel_time ∈ {-3, -2, -1}.
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

    # Placebo: bootstrap percentile p + TOST equivalence p.
    if "placebo" in requested and placebo_period is not None:
        p_start, p_end = placebo_period
        placebo_mask = (time_on >= p_start) & (time_on <= p_end) & pre_mask
        if np.any(placebo_mask):
            p_idx = np.where(placebo_mask)[0]
            estimate = float(np.mean(att_on[p_idx]))
            p_value = float("nan")
            equiv_p = float("nan")
            se = float("nan")
            if inf.att_on_boot is not None:
                boot_window = inf.att_on_boot[p_idx, :]
                valid_boot = np.all(np.isfinite(boot_window), axis=0)
                boot_placebo = np.mean(boot_window[:, valid_boot], axis=0)
                p_value = min(
                    2 * float(np.mean(boot_placebo >= 0)),
                    2 * float(np.mean(boot_placebo <= 0)),
                    1.0,
                ) if boot_placebo.size else float("nan")
                if boot_placebo.size > 1:
                    se = float(np.std(boot_placebo, ddof=1))
                    if se > 0:
                        df_p = boot_placebo.size - 1
                        t_up = (estimate - threshold_abs) / se
                        t_lo = (estimate + threshold_abs) / se
                        equiv_p = max(
                            float(stats.t.cdf(t_up, df_p)),
                            float(1 - stats.t.cdf(t_lo, df_p)),
                        )
            diag.placebo = PlaceboResult(
                estimate=estimate,
                se=se,
                p_value=p_value,
                equiv_p_value=equiv_p,
                period=(int(p_start), int(p_end)),
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
        _validate_period_option("placebo_period", opts["placebo_period"])
    if "carryover_period" in opts:
        _validate_period_option("carryover_period", opts["carryover_period"])

    if "loo" in opts and not isinstance(opts["loo"], (bool, np.bool_)):
        raise ValueError("diagnostics_options['loo'] must be a bool.")

    return requested
