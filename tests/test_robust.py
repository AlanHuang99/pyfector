"""
Robustness tests: edge cases, unusual panel structures, input validation.

These tests sit between ``test_smoke.py`` (happy-path sanity) and
``test_stress.py`` (large-N performance).  They exercise the code paths
that are most likely to break when the DGP deviates from the default.
"""

import numpy as np
import polars as pl
import pytest

import pyfector
from pyfector.cv import _make_cv_folds, _select_best
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
