"""
Robustness tests: edge cases, unusual panel structures, input validation.

These tests sit between ``test_smoke.py`` (happy-path sanity) and
``test_stress.py`` (large-N performance).  They exercise the code paths
that are most likely to break when the DGP deviates from the default.
"""

import warnings
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
from scipy import stats

import pyfector
from pyfector.cv import _make_cv_folds, _select_best, _warn_if_lambda_on_boundary
from pyfector.fect import _compute_effects
from pyfector.panel import prepare_panel
from conftest import _simulate_panel


# ---------------------------------------------------------------------------
# Edge case panel structures
# ---------------------------------------------------------------------------

class TestEdgeCasePanels:
    """Unusual but valid panel shapes that must run correctly."""

    def test_single_treated_unit(self):
        sim = _simulate_panel(N=30, T=20, N_treated=1, r=2, seed=10)
        result = pyfector.fect(
            data=sim["data"], Y="Y", D="D", index=("unit", "time"),
            method="fe", CV=False, se=False, seed=42,
        )
        assert result.att_avg != 0
        assert result.panel.N == 30

    def test_single_control_unit(self):
        """Only 1 control unit: estimator should still run."""
        sim = _simulate_panel(N=10, T=20, N_treated=9, r=1, seed=11)
        result = pyfector.fect(
            data=sim["data"], Y="Y", D="D", index=("unit", "time"),
            method="fe", CV=False, se=False, seed=42,
        )
        assert result.att_avg != 0

    def test_short_panel_T_equals_5(self):
        sim = _simulate_panel(N=50, T=5, N_treated=20, r=1, seed=12)
        result = pyfector.fect(
            data=sim["data"], Y="Y", D="D", index=("unit", "time"),
            method="fe", CV=False, se=False, seed=42,
        )
        assert np.isfinite(result.att_avg)

    def test_long_panel_T_equals_200(self):
        sim = _simulate_panel(N=40, T=200, N_treated=15, r=2, seed=13)
        result = pyfector.fect(
            data=sim["data"], Y="Y", D="D", index=("unit", "time"),
            method="ife", r=2, CV=False, se=False, seed=42,
        )
        assert np.isfinite(result.att_avg)

    def test_highly_unbalanced_panel(self):
        """30% of cells missing, but treated cells preserved."""
        sim = _simulate_panel(
            N=60, T=30, N_treated=25, r=2, missing_frac=0.3, seed=14,
        )
        result = pyfector.fect(
            data=sim["data"], Y="Y", D="D", index=("unit", "time"),
            method="fe", CV=False, se=False, seed=42,
        )
        assert np.isfinite(result.att_avg)

    def test_treatment_reversals(self):
        sim = _simulate_panel(
            N=40, T=25, N_treated=15, r=1, reversals=True, seed=15,
        )
        result = pyfector.fect(
            data=sim["data"], Y="Y", D="D", index=("unit", "time"),
            method="fe", CV=False, se=False, seed=42,
        )
        assert np.isfinite(result.att_avg)
        # Reversal units should be classified as unit_type == 3
        assert np.any(result.panel.unit_type == 3)


# ---------------------------------------------------------------------------
# DGP variations
# ---------------------------------------------------------------------------

class TestDGPVariations:
    """Different DGPs that pyfector should handle."""

    @pytest.mark.parametrize("method", ["fe", "ife", "mc"])
    def test_zero_treatment_effect(self, method):
        """With delta=0 the ATT should be close to zero."""
        sim = _simulate_panel(N=80, T=30, N_treated=30, r=2, delta=0.0, seed=20)
        kw = dict(
            data=sim["data"], Y="Y", D="D", index=("unit", "time"),
            method=method, CV=False, se=False, seed=42,
        )
        if method == "ife":
            kw["r"] = 2
        if method == "mc":
            kw["lam"] = 0.01
        result = pyfector.fect(**kw)
        assert abs(result.att_avg) < 1.0

    @pytest.mark.parametrize("delta", [1.0, 5.0, -3.0])
    def test_recovers_various_delta(self, delta):
        sim = _simulate_panel(N=100, T=40, N_treated=40, r=2, delta=delta, seed=21)
        result = pyfector.fect(
            data=sim["data"], Y="Y", D="D", index=("unit", "time"),
            method="fe", CV=False, se=False, seed=42,
        )
        assert abs(result.att_avg - delta) < 1.0

    def test_high_rank_factors(self):
        """DGP with r=5 factors; IFE with r=5 should match."""
        sim = _simulate_panel(N=80, T=40, N_treated=30, r=5, seed=22)
        result = pyfector.fect(
            data=sim["data"], Y="Y", D="D", index=("unit", "time"),
            method="ife", r=5, CV=False, se=False, seed=42,
        )
        # Should recover delta=3.0
        assert abs(result.att_avg - 3.0) < 1.0

    def test_with_many_covariates(self):
        sim = _simulate_panel(N=80, T=30, N_treated=30, r=2, p=5, seed=23)
        result = pyfector.fect(
            data=sim["data"], Y="Y", D="D", index=("unit", "time"),
            X=sim["covariate_names"],
            method="fe", CV=False, se=False, seed=42,
        )
        assert result.beta is not None
        assert len(result.beta) == 5


# ---------------------------------------------------------------------------
# API input flexibility
# ---------------------------------------------------------------------------

class TestInputs:
    def test_pandas_input(self):
        """Pandas DataFrame should work via polars conversion."""
        pd = pytest.importorskip("pandas")
        sim = _simulate_panel(N=40, T=20, N_treated=15, r=1, seed=30)
        df_pd = sim["data"].to_pandas()
        result = pyfector.fect(
            data=df_pd, Y="Y", D="D", index=("unit", "time"),
            method="fe", CV=False, se=False, seed=42,
        )
        assert np.isfinite(result.att_avg)

    def test_polars_lazyframe_input(self):
        sim = _simulate_panel(N=40, T=20, N_treated=15, r=1, seed=31)
        lf = sim["data"].lazy()
        result = pyfector.fect(
            data=lf, Y="Y", D="D", index=("unit", "time"),
            method="fe", CV=False, se=False, seed=42,
        )
        assert np.isfinite(result.att_avg)

    def test_reproducible_seed(self):
        sim = _simulate_panel(N=60, T=25, N_treated=20, r=2, seed=40)
        r1 = pyfector.fect(
            data=sim["data"], Y="Y", D="D", index=("unit", "time"),
            method="ife", r=2, CV=False, se=True, nboots=30,
            n_jobs=1, seed=42,
        )
        r2 = pyfector.fect(
            data=sim["data"], Y="Y", D="D", index=("unit", "time"),
            method="ife", r=2, CV=False, se=True, nboots=30,
            n_jobs=1, seed=42,
        )
        assert r1.att_avg == r2.att_avg
        assert r1.inference.att_avg_se == r2.inference.att_avg_se

    def test_group_raises_until_supported(self):
        sim = _simulate_panel(N=30, T=20, N_treated=10, r=1, seed=41)
        with pytest.raises(NotImplementedError, match="group"):
            pyfector.fect(
                data=sim["data"], Y="Y", D="D", index=("unit", "time"),
                method="fe", CV=False, group="unit",
            )

    def test_cfe_z_q_raise_until_supported(self):
        sim = _simulate_panel(N=30, T=20, N_treated=10, r=1, seed=42)
        with pytest.raises(NotImplementedError, match="Z.*Q"):
            pyfector.fect(
                data=sim["data"], Y="Y", D="D", index=("unit", "time"),
                method="cfe", CV=False, Z=["Z1"], Q=["Q1"],
            )

    def test_cv_nobs_creates_block_masks(self):
        II = np.ones((12, 4))
        D = np.zeros((12, 4))
        folds = _make_cv_folds(
            II, D, k=1, cv_prop=0.5, cv_nobs=3,
            cv_treat=False, cv_donut=0,
            rng=np.random.default_rng(123),
        )
        mask = folds[0]["mask"]

        has_run = False
        for j in range(mask.shape[1]):
            run = 0
            for value in mask[:, j]:
                run = run + 1 if value else 0
                has_run = has_run or run >= 2
        assert has_run


class TestCVSelection:
    def test_default_rule_returns_lowest_score_candidate(self):
        scores = {
            0: {"mspe": 1.02},
            1: {"mspe": 1.005},
            2: {"mspe": 1.0},
        }
        assert _select_best(scores, [0, 1, 2], "mspe") == 2

    def test_min_rule_returns_lowest_score_candidate(self):
        scores = {
            0: {"mspe": 1.02},
            1: {"mspe": 1.005},
            2: {"mspe": 1.0},
        }
        assert _select_best(scores, [0, 1, 2], "mspe", cv_rule="min") == 2

    def test_onepct_rule_prefers_lower_rank_within_slack(self):
        scores = {
            0: {"mspe": 1.02},
            1: {"mspe": 1.005},
            2: {"mspe": 1.0},
            3: {"mspe": 1.2},
        }
        selected = _select_best(scores, [0, 1, 2, 3], "mspe", cv_rule="onepct")
        assert selected == 1

    def test_onepct_rule_prefers_larger_lambda_within_slack(self):
        scores = {
            10.0: {"mspe": 1.02},
            5.0: {"mspe": 1.005},
            2.0: {"mspe": 1.0},
            1.0: {"mspe": 1.2},
        }
        selected = _select_best(scores, [10.0, 5.0, 2.0, 1.0], "mspe", cv_rule="onepct")
        assert selected == 5.0

    def test_onepct_rule_uses_global_best_threshold(self):
        scores = {
            0: {"mspe": 10.0},
            1: {"mspe": 9.95},
            2: {"mspe": 9.89},
        }
        assert _select_best(scores, [0, 1, 2], "mspe", cv_rule="onepct") == 1

    def test_invalid_cv_rule_raises(self):
        scores = {0: {"mspe": 1.0}}
        with pytest.raises(ValueError, match="cv_rule"):
            _select_best(scores, [0], "mspe", cv_rule="bad")

    def test_1se_rule_is_not_accepted(self):
        scores = {0: {"mspe": 1.0}}
        with pytest.raises(ValueError, match="cv_rule"):
            _select_best(scores, [0], "mspe", cv_rule="1se")

    def test_lambda_boundary_warning_for_lowest_candidate(self):
        with pytest.warns(UserWarning, match="lowest lambda"):
            _warn_if_lambda_on_boundary(0.0, [10.0, 1.0, 0.0])

    def test_lambda_boundary_warning_for_highest_candidate(self):
        with pytest.warns(UserWarning, match="highest lambda"):
            _warn_if_lambda_on_boundary(10.0, [10.0, 1.0, 0.0])

    def test_lambda_boundary_warning_ignores_interior_candidate(self):
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            _warn_if_lambda_on_boundary(1.0, [10.0, 1.0, 0.0])
        assert len(record) == 0


class TestPanelFiltering:
    def test_min_T0_strict_filters_sparse_controls(self):
        rows = []
        y_values = {
            0: [1.0, 2.0, 4.0, 5.0],
            1: [10.0, np.nan, np.nan, np.nan],
            2: [np.nan, np.nan, np.nan, np.nan],
            3: [2.0, 2.5, np.nan, np.nan],
        }
        d_values = {
            0: [0, 0, 1, 1],
            1: [0, 0, 0, 0],
            2: [0, 0, 0, 0],
            3: [0, 0, 0, 0],
        }
        for unit in range(4):
            for time in range(4):
                rows.append({
                    "unit": unit,
                    "time": time,
                    "Y": y_values[unit][time],
                    "D": d_values[unit][time],
                })
        df = pl.DataFrame(rows)

        permissive = prepare_panel(
            df, Y="Y", D="D", index=("unit", "time"),
            min_T0=2, min_T0_strict=False,
        )
        strict = prepare_panel(
            df, Y="Y", D="D", index=("unit", "time"),
            min_T0=2, min_T0_strict=True,
        )

        assert permissive.unit_ids.tolist() == [0, 1, 3]
        assert strict.unit_ids.tolist() == [0, 3]

    def test_missing_treated_outcomes_do_not_enter_att(self):
        eff = np.array([
            [0.1],
            [2.0],
            [-99.0],
        ])
        D = np.array([
            [0.0],
            [1.0],
            [1.0],
        ])
        T_on = np.array([
            [-1.0],
            [0.0],
            [1.0],
        ])
        I = np.array([
            [1.0],
            [1.0],
            [0.0],
        ])

        att_avg, att_on, time_on, count_on, att_avg_unit = _compute_effects(eff, D, T_on, I)

        assert att_avg == 2.0
        assert att_avg_unit == 2.0
        assert time_on.tolist() == [-1.0, 0.0]
        assert att_on.tolist() == [0.1, 2.0]
        assert count_on.tolist() == [1, 1]


# ---------------------------------------------------------------------------
# Treatment timing edge cases
# ---------------------------------------------------------------------------

class TestTreatmentTiming:
    def test_T_on_undefined_for_never_treated(self):
        sim = _simulate_panel(N=50, T=20, N_treated=20, r=1, seed=50)
        result = pyfector.fect(
            data=sim["data"], Y="Y", D="D", index=("unit", "time"),
            method="fe", CV=False, se=False, seed=42,
        )
        T_on = result.panel.T_on
        # Never-treated columns should be all NaN
        never_treated = result.panel.unit_type == 1
        assert np.all(np.isnan(T_on[:, never_treated]))

    def test_T_on_zero_at_onset(self):
        """At t0, relative time should be exactly 0."""
        sim = _simulate_panel(N=30, T=20, N_treated=10, r=1, seed=51)
        result = pyfector.fect(
            data=sim["data"], Y="Y", D="D", index=("unit", "time"),
            method="fe", CV=False, se=False, seed=42,
        )
        T_on = result.panel.T_on
        D = result.panel.D
        # For each treated column, T_on should be 0 at the first D==1 observed cell
        for j in np.where(result.panel.unit_type == 2)[0]:
            treated_times = np.where(D[:, j] > 0)[0]
            if len(treated_times) > 0:
                first = treated_times[0]
                assert T_on[first, j] == 0

    def test_staggered_adoption(self):
        """Multiple treatment cohorts should all be handled."""
        sim = _simulate_panel(N=90, T=30, N_treated=60, r=2, seed=52)
        result = pyfector.fect(
            data=sim["data"], Y="Y", D="D", index=("unit", "time"),
            method="fe", CV=False, se=False, seed=42,
        )
        # Dynamic ATT should exist and include pre-treatment periods
        assert result.time_on is not None
        assert (result.time_on < 0).any()
        assert (result.time_on >= 0).any()


# ---------------------------------------------------------------------------
# sigma2_fect (Liu et al. 2024 additive-FE baseline residual variance)
# ---------------------------------------------------------------------------

class TestSigma2Fect:
    """`result.sigma2_fect` is the additive-FE baseline residual variance,
    used as the denominator of the Liu auto-scaled TOST bound."""

    def _fit(self, method, **kw):
        sim = _simulate_panel(N=60, T=20, N_treated=25, r=2, seed=11)
        return pyfector.fect(
            data=sim["data"], Y="Y", D="D", index=("unit", "time"),
            method=method, CV=False, se=False, seed=42, **kw,
        )

    def test_sigma2_fect_populated_for_fe(self):
        result = self._fit("fe")
        assert result.sigma2_fect > 0

    def test_sigma2_fect_populated_for_ife(self):
        result = self._fit("ife", r=2)
        assert result.sigma2_fect > 0

    def test_sigma2_fect_populated_for_mc(self):
        result = self._fit("mc", lam=0.5)
        assert result.sigma2_fect > 0

    def test_sigma2_fect_equals_sigma2_for_fe(self):
        """method='fe' IS the additive-FE pass; sigma2_fect must equal sigma2."""
        result = self._fit("fe")
        assert result.sigma2_fect == pytest.approx(result.sigma2)

    def test_sigma2_fect_strictly_larger_for_ife_with_factor_panel(self):
        """IFE absorbs latent factors that additive FE doesn't, so the
        chosen-estimator residual variance is smaller."""
        result = self._fit("ife", r=2)
        assert result.sigma2 < result.sigma2_fect

    def test_sigma2_fect_normalize_roundtrip(self):
        """With normalize=True, sigma2_fect must come back on the
        original outcome scale, not the standardized scale."""
        sim = _simulate_panel(N=60, T=20, N_treated=25, r=2, seed=12)
        a = pyfector.fect(
            data=sim["data"], Y="Y", D="D", index=("unit", "time"),
            method="ife", r=2, CV=False, se=False, normalize=False, seed=42,
        )
        b = pyfector.fect(
            data=sim["data"], Y="Y", D="D", index=("unit", "time"),
            method="ife", r=2, CV=False, se=False, normalize=True, seed=42,
        )
        # Should agree to within a few percent (numerics + EM ordering).
        assert b.sigma2_fect == pytest.approx(a.sigma2_fect, rel=0.1)

    def test_sigma2_fect_uses_fe_residual_degrees_of_freedom(self):
        """The r=0 baseline variance uses residual df for the FE design."""
        resid = np.array([
            [1.0, -1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ])
        T, N = resid.shape
        mu = 10.0
        alpha = np.array([0.0, 2.0, -1.0])
        xi = np.array([0.0, 1.0, -2.0, 3.0])
        Y = mu + alpha[None, :] + xi[:, None] + resid
        df = pl.DataFrame([
            {"unit": j, "time": t, "Y": float(Y[t, j]), "D": 0}
            for j in range(N)
            for t in range(T)
        ])

        result = pyfector.fect(
            data=df, Y="Y", D="D", index=("unit", "time"),
            method="ife", r=1, CV=False, se=False, force="two-way",
        )
        expected = float(np.sum(resid ** 2) / (T * N - (T + N - 1)))
        assert result.sigma2_fect == pytest.approx(expected)

    def test_two_way_fe_rank_handles_disconnected_observed_cells(self):
        from pyfector.estimators import _active_fe_rank

        mask = np.array([
            [True, False],
            [False, True],
        ])
        assert _active_fe_rank(mask, force=3) == 2


# ---------------------------------------------------------------------------
# run_diagnostics threshold semantics
# ---------------------------------------------------------------------------

@pytest.fixture
def fitted_with_se(small_panel):
    """A small fitted result with bootstrap SEs, sufficient for diagnostics."""
    return pyfector.fect(
        data=small_panel["data"], Y="Y", D="D", index=("unit", "time"),
        method="ife", r=2, CV=False, se=True, nboots=50, seed=42,
    )


class TestTostThresholdSemantics:
    """The literal 0.36 (default or explicit) auto-scales by sigma2_fect.
    Other positive floats are taken as absolute outcome-scale bounds."""

    def test_default_uses_liu_scaling(self, fitted_with_se):
        diag = fitted_with_se.diagnose()
        expected = 0.36 * np.sqrt(fitted_with_se.sigma2_fect)
        assert diag.tost.threshold == pytest.approx(expected)
        assert diag.tost.threshold_source == "auto"
        assert diag.tost.sigma2_fect == pytest.approx(fitted_with_se.sigma2_fect)

    def test_explicit_036_also_auto_scales(self, fitted_with_se):
        """Legacy-magic rule: explicit 0.36 means Liu's default convention."""
        diag = fitted_with_se.diagnose(tost_threshold=0.36)
        expected = 0.36 * np.sqrt(fitted_with_se.sigma2_fect)
        assert diag.tost.threshold == pytest.approx(expected)
        assert diag.tost.threshold_source == "auto"

    def test_explicit_other_float_is_absolute(self, fitted_with_se):
        diag = fitted_with_se.diagnose(tost_threshold=0.5)
        assert diag.tost.threshold == 0.5
        assert diag.tost.threshold_source == "explicit"

    def test_tost_threshold_none_raises(self, fitted_with_se):
        with pytest.raises(ValueError, match="tost_threshold=None"):
            fitted_with_se.diagnose(tost_threshold=None)

    def test_tost_threshold_negative_raises(self, fitted_with_se):
        with pytest.raises(ValueError, match="positive finite float"):
            fitted_with_se.diagnose(tost_threshold=-0.1)

    def test_tost_threshold_zero_raises(self, fitted_with_se):
        with pytest.raises(ValueError, match="positive finite float"):
            fitted_with_se.diagnose(tost_threshold=0.0)

    def test_tost_threshold_nonfinite_raises(self, fitted_with_se):
        with pytest.raises(ValueError, match="positive finite float"):
            fitted_with_se.diagnose(tost_threshold=float("nan"))
        with pytest.raises(ValueError, match="positive finite float"):
            fitted_with_se.diagnose(tost_threshold=float("inf"))

    def test_pretrend_only_does_not_resolve_tost_threshold(self, fitted_with_se):
        diag = fitted_with_se.diagnose(
            tost_threshold=None,
            _requested=["pretrend_f"],
        )
        assert diag.pretrend_f is not None
        assert diag.tost is None


class TestPlaceboTest:
    """Placebo diagnostics use the Liu-Wang-Xu/R-fect holdout-refit test."""

    def test_placebo_equiv_p_value_populated(self, fitted_with_se):
        fitted_with_se.inference.att_on_boot = (
            fitted_with_se.inference.att_on_boot[:, :8]
        )
        diag = fitted_with_se.diagnose(placebo_period=(-3, 0))
        assert diag.placebo is not None
        assert isinstance(diag.placebo.equiv_p_value, float)
        assert 0.0 <= diag.placebo.equiv_p_value <= 1.0
        assert diag.placebo.period == (-3, 0)
        assert diag.placebo.n_obs > 0
        assert diag.placebo.n_boot > 1
        assert 0.0 <= diag.placebo.p_value <= 1.0
        expected_p = 2 * stats.norm.sf(abs(diag.placebo.estimate / diag.placebo.se))
        assert diag.placebo.p_value == pytest.approx(expected_p)

    def test_placebo_does_not_average_existing_pre_period_atts(self, fitted_with_se):
        fitted_with_se.inference.att_on_boot = (
            fitted_with_se.inference.att_on_boot[:, :5]
        )
        pre = fitted_with_se.time_on < 0
        fitted_with_se.att_on[pre] = 1_000_000.0

        diag = pyfector.run_diagnostics(
            fitted_with_se,
            placebo_period=(-3, 0),
            _requested=["placebo"],
        )

        assert diag.placebo is not None
        assert abs(diag.placebo.estimate) < 100_000.0

    def test_f_tests_filter_nan_bootstrap_columns(self):
        inf = SimpleNamespace(
            att_on_boot=np.array([
                [0.10, 0.20, np.nan, 0.00, 0.15],
                [-0.05, 0.05, 0.10, -0.10, 0.00],
                [1.00, 1.10, 0.90, 1.20, 0.80],
            ]),
            att_on_se=np.array([0.1, 0.1, 0.1]),
        )
        result = SimpleNamespace(
            inference=inf,
            time_on=np.array([-2.0, -1.0, 0.0]),
            att_on=np.array([0.10, -0.05, 1.0]),
            att_avg=1.0,
            sigma2_fect=1.0,
        )

        diag = pyfector.run_diagnostics(
            result,
            tost_threshold=None,
            _requested=["pretrend_f", "equiv_f"],
        )
        assert diag.pretrend_f is not None
        assert diag.equiv_f is not None
        assert diag.pretrend_f.df2 == 2
        assert np.isfinite(diag.pretrend_f.p_value)


class TestDiagnosticOptionsProvenance:
    """`options` dict on Diagnostics must record what was used."""

    def test_options_recorded(self, fitted_with_se):
        fitted_with_se.inference.att_on_boot = (
            fitted_with_se.inference.att_on_boot[:, :5]
        )
        diag = fitted_with_se.diagnose(placebo_period=(-3, 0))
        assert diag.options["tost_threshold"] == diag.tost.threshold
        assert diag.options["placebo_period"] == (-3, 0)
        assert diag.options["sigma2_fect"] == pytest.approx(
            fitted_with_se.sigma2_fect
        )
        assert diag.options["pyfector_version"] == pyfector.__version__


# ---------------------------------------------------------------------------
# diagnostics= on fect()
# ---------------------------------------------------------------------------

class TestFitTimeDiagnostics:
    """The `diagnostics=...` arg on fect() runs the requested suite at
    fit time and attaches the slim-safe Diagnostics to result.diagnostics."""

    def _kw(self, small_panel, **extra):
        return dict(
            data=small_panel["data"], Y="Y", D="D", index=("unit", "time"),
            method="ife", r=2, CV=False, se=True, nboots=30, seed=42,
            **extra,
        )

    def test_full_populates_required_tests(self, small_panel):
        result = pyfector.fect(**self._kw(small_panel, diagnostics="full"))
        assert result.diagnostics is not None
        assert result.diagnostics.tost is not None
        assert result.diagnostics.pretrend_f is not None
        assert result.diagnostics.equiv_f is not None
        # No placebo/carryover/loo config supplied → silently skipped.
        assert result.diagnostics.placebo is None
        assert result.diagnostics.carryover is None
        assert result.diagnostics.loo is None

    def test_full_with_placebo_period_runs_placebo(self, small_panel):
        result = pyfector.fect(**self._kw(
            small_panel, diagnostics="full",
            diagnostics_options={"placebo_period": (-3, 0)},
        ))
        assert result.diagnostics.placebo is not None
        assert result.diagnostics.placebo.period == (-3, 0)

    def test_full_without_se_raises(self, small_panel):
        with pytest.raises(ValueError, match="diagnostics requires se=True"):
            pyfector.fect(**dict(
                self._kw(small_panel), se=False, diagnostics="full",
            ))

    def test_subset_runs_only_listed(self, small_panel):
        result = pyfector.fect(**self._kw(
            small_panel, diagnostics=["tost"],
        ))
        assert result.diagnostics.tost is not None
        assert result.diagnostics.pretrend_f is None
        assert result.diagnostics.equiv_f is None

    def test_explicit_loo_list_runs_loo(self, small_panel):
        result = pyfector.fect(**self._kw(
            small_panel, diagnostics=["loo"],
        ))
        assert result.diagnostics.loo is not None

    def test_subset_unknown_name_raises(self, small_panel):
        with pytest.raises(ValueError, match="Unknown diagnostic"):
            pyfector.fect(**self._kw(
                small_panel, diagnostics=["bogus"],
            ))

    def test_subset_empty_list_raises(self, small_panel):
        with pytest.raises(ValueError, match="diagnostics=\\[\\]"):
            pyfector.fect(**self._kw(small_panel, diagnostics=[]))

    def test_explicit_placebo_without_period_raises(self, small_panel):
        with pytest.raises(ValueError, match="placebo_period.*missing"):
            pyfector.fect(**self._kw(
                small_panel, diagnostics=["placebo"],
            ))

    def test_full_without_placebo_period_skips_silently(self, small_panel):
        """`full` is best-effort; missing placebo config silently skips."""
        result = pyfector.fect(**self._kw(
            small_panel, diagnostics="full",
        ))
        assert result.diagnostics.placebo is None  # not raised, just skipped

    def test_unknown_options_key_raises(self, small_panel):
        with pytest.raises(ValueError, match="unknown diagnostics_options key"):
            pyfector.fect(**self._kw(
                small_panel, diagnostics="full",
                diagnostics_options={"bogus_key": 1},
            ))

    def test_unknown_options_key_raises_even_when_diagnostics_none(self, small_panel):
        with pytest.raises(ValueError, match="unknown diagnostics_options key"):
            pyfector.fect(**self._kw(
                small_panel, diagnostics="none",
                diagnostics_options={"bogus_key": 1},
            ))

    def test_invalid_tost_option_raises_before_prepare_panel(self, small_panel, monkeypatch):
        import importlib

        fect_module = importlib.import_module("pyfector.fect")

        def _fail_prepare_panel(*args, **kwargs):
            raise AssertionError("prepare_panel should not be called")

        monkeypatch.setattr(fect_module, "prepare_panel", _fail_prepare_panel)
        with pytest.raises(ValueError, match="tost_threshold=None"):
            fect_module.fect(**self._kw(
                small_panel, diagnostics="full",
                diagnostics_options={"tost_threshold": None},
            ))

    @pytest.mark.parametrize(
        ("options", "match"),
        [
            ({"tost_threshold": -0.1}, "positive finite float"),
            ({"f_threshold": 0.0}, "f_threshold"),
            ({"alpha": 1.0}, "alpha"),
            ({"placebo_period": (1, -1)}, "start <= end"),
            ({"placebo_period": (0, 0)}, "pre-treatment"),
            ({"placebo_period": (-1, 1)}, "post-treatment"),
            ({"loo": "yes"}, "loo"),
        ],
    )
    def test_invalid_diagnostics_option_values_raise(self, small_panel, options, match):
        with pytest.raises(ValueError, match=match):
            pyfector.fect(**self._kw(
                small_panel, diagnostics="full",
                diagnostics_options=options,
            ))


# ---------------------------------------------------------------------------
# Slim-safety contract: Diagnostics must not hold panel/inference refs
# ---------------------------------------------------------------------------

class TestDiagnosticsContainer:
    """`result.diagnostics` must hold only primitives, tuples, dicts of
    primitives, and bounded numpy arrays. Survives a downstream slim that
    drops result.panel and result.inference.*_boot."""

    def test_registry_shape_is_future_proof(self):
        diag = pyfector.Diagnostics(
            options={"alpha": 0.05},
            tests={"future_test": {"value": 1.0}},
        )
        assert pyfector.DiagnosticResult is pyfector.Diagnostics
        assert "future_test" in diag
        assert diag["future_test"] == {"value": 1.0}
        assert diag.get("missing") is None
        assert diag.available == ("future_test",)
        diag.set_test("another_future_test", 2.0)
        assert diag.get("another_future_test") == 2.0

        diag.tost = pyfector.TostResult(
            pvals=np.array([0.01]),
            periods=np.array([-1.0]),
            threshold=0.1,
            threshold_source="explicit",
            sigma2_fect=None,
            max_pval=0.01,
            all_pass=True,
        )
        assert diag.tests["tost"] is diag.tost
        assert diag.available == ("tost", "another_future_test", "future_test")

        diag.tost = None
        assert diag.tost is None
        assert "tost" not in diag

    def test_no_heavy_references_in_diagnostics(self, small_panel):
        result = pyfector.fect(
            data=small_panel["data"], Y="Y", D="D", index=("unit", "time"),
            method="ife", r=2, CV=False, se=True, nboots=12, seed=42,
            diagnostics="full",
            diagnostics_options={"placebo_period": (-3, 0)},
        )
        from pyfector.panel import PanelData
        from pyfector.inference import InferenceResult
        from pyfector.estimators import EstimationResult

        forbidden_types = (
            PanelData, InferenceResult, EstimationResult, pyfector.FectResult,
        )

        def walk(obj, path="diagnostics"):
            if isinstance(obj, forbidden_types):
                raise AssertionError(
                    f"Diagnostics contains a {type(obj).__name__} at "
                    f"{path}; slim-safety contract violated."
                )
            if hasattr(obj, "__dataclass_fields__"):
                for f in obj.__dataclass_fields__:
                    walk(getattr(obj, f), f"{path}.{f}")
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    walk(v, f"{path}[{k!r}]")
            elif isinstance(obj, (list, tuple)):
                for i, v in enumerate(obj):
                    walk(v, f"{path}[{i}]")
            # primitives, ndarrays, None: OK

        walk(result.diagnostics)

    def test_diagnostics_pickle_roundtrip(self, small_panel):
        import pickle
        result = pyfector.fect(
            data=small_panel["data"], Y="Y", D="D", index=("unit", "time"),
            method="ife", r=2, CV=False, se=True, nboots=12, seed=42,
            diagnostics="full",
            diagnostics_options={"placebo_period": (-3, 0)},
        )
        blob = pickle.dumps(result.diagnostics)
        restored = pickle.loads(blob)
        assert restored.tost.threshold == result.diagnostics.tost.threshold
        assert restored.options == result.diagnostics.options
        np.testing.assert_array_equal(
            restored.tost.pvals, result.diagnostics.tost.pvals,
        )
