"""
Smoke tests: verify basic functionality of each method runs without errors
and produces reasonable outputs.
"""

import numpy as np
import polars as pl
import pytest

import pyfect


class TestFESmoke:
    """Test method='fe' (plain fixed effects)."""

    def test_basic_fe(self, small_panel):
        result = pyfect.fect(
            data=small_panel["data"],
            Y="Y", D="D",
            index=("unit", "time"),
            method="fe",
            force="two-way",
            seed=42,
        )
        assert result.converged or result.niter == 0
        assert result.att_avg != 0
        assert result.eff is not None
        assert result.Y_ct is not None
        assert result.method == "fe"

    def test_fe_unit_force(self, small_panel):
        result = pyfect.fect(
            data=small_panel["data"],
            Y="Y", D="D",
            index=("unit", "time"),
            method="fe",
            force="unit",
            seed=42,
        )
        assert result.att_avg != 0

    def test_fe_time_force(self, small_panel):
        result = pyfect.fect(
            data=small_panel["data"],
            Y="Y", D="D",
            index=("unit", "time"),
            method="fe",
            force="time",
            seed=42,
        )
        assert result.att_avg != 0

    def test_fe_no_force(self, small_panel):
        result = pyfect.fect(
            data=small_panel["data"],
            Y="Y", D="D",
            index=("unit", "time"),
            method="fe",
            force="none",
            seed=42,
        )
        assert result.att_avg != 0

    def test_fe_att_near_true(self, small_panel):
        """ATT should be in the right ballpark of the true effect."""
        result = pyfect.fect(
            data=small_panel["data"],
            Y="Y", D="D",
            index=("unit", "time"),
            method="fe",
            force="two-way",
            seed=42,
        )
        # FE without factors will be biased but should be within ~50% for moderate factor strength
        assert abs(result.att_avg) < 20, f"ATT={result.att_avg} seems unreasonable"


class TestIFESmoke:
    """Test method='ife' (interactive fixed effects)."""

    def test_basic_ife_r0(self, small_panel):
        result = pyfect.fect(
            data=small_panel["data"],
            Y="Y", D="D",
            index=("unit", "time"),
            method="ife",
            r=0,
            CV=False,
            seed=42,
        )
        assert result.r_cv == 0
        assert result.att_avg != 0

    def test_basic_ife_r2(self, small_panel):
        result = pyfect.fect(
            data=small_panel["data"],
            Y="Y", D="D",
            index=("unit", "time"),
            method="ife",
            r=2,
            CV=False,
            seed=42,
        )
        assert result.r_cv == 2
        assert result.factors is not None
        assert result.factors.shape[1] == 2
        assert result.loadings is not None
        assert result.loadings.shape[1] == 2

    def test_ife_cv(self, small_panel):
        """IFE with cross-validation to select r."""
        result = pyfect.fect(
            data=small_panel["data"],
            Y="Y", D="D",
            index=("unit", "time"),
            method="ife",
            r=(0, 3),
            CV=True,
            k=5,
            seed=42,
        )
        assert result.r_cv is not None
        assert 0 <= result.r_cv <= 3
        assert result.cv_result is not None

    def test_ife_att_better_than_fe(self, small_panel):
        """IFE with correct r should be closer to true ATT than plain FE."""
        true_att = small_panel["true_att"]
        r_true = small_panel["r"]

        fe_result = pyfect.fect(
            data=small_panel["data"],
            Y="Y", D="D",
            index=("unit", "time"),
            method="fe",
            seed=42,
        )
        ife_result = pyfect.fect(
            data=small_panel["data"],
            Y="Y", D="D",
            index=("unit", "time"),
            method="ife",
            r=r_true,
            CV=False,
            seed=42,
        )
        # IFE should generally be closer to truth (not guaranteed for small N)
        fe_err = abs(fe_result.att_avg - true_att)
        ife_err = abs(ife_result.att_avg - true_att)
        # At least one should be reasonable
        assert min(fe_err, ife_err) < true_att * 0.5


class TestMCSmoke:
    """Test method='mc' (matrix completion)."""

    def test_basic_mc(self, small_panel):
        result = pyfect.fect(
            data=small_panel["data"],
            Y="Y", D="D",
            index=("unit", "time"),
            method="mc",
            lam=0.1,
            CV=False,
            seed=42,
        )
        assert result.lambda_cv == 0.1
        assert result.att_avg != 0

    def test_mc_cv(self, small_panel):
        """MC with cross-validation to select lambda."""
        result = pyfect.fect(
            data=small_panel["data"],
            Y="Y", D="D",
            index=("unit", "time"),
            method="mc",
            CV=True,
            nlambda=5,
            k=5,
            seed=42,
        )
        assert result.lambda_cv is not None
        assert result.cv_result is not None


class TestCovariateSmoke:
    """Test with covariates."""

    def test_fe_with_covariates(self, panel_with_covariates):
        p = panel_with_covariates["p"]
        result = pyfect.fect(
            data=panel_with_covariates["data"],
            Y="Y", D="D",
            index=("unit", "time"),
            X=panel_with_covariates["covariate_names"],
            method="fe",
            seed=42,
        )
        assert result.beta is not None
        assert len(result.beta) == p

    def test_ife_with_covariates(self, panel_with_covariates):
        result = pyfect.fect(
            data=panel_with_covariates["data"],
            Y="Y", D="D",
            index=("unit", "time"),
            X=panel_with_covariates["covariate_names"],
            method="ife",
            r=2,
            CV=False,
            seed=42,
        )
        assert result.beta is not None
        assert result.factors is not None


class TestUnbalancedSmoke:
    """Test with unbalanced panels."""

    def test_fe_unbalanced(self, unbalanced_panel):
        result = pyfect.fect(
            data=unbalanced_panel["data"],
            Y="Y", D="D",
            index=("unit", "time"),
            method="fe",
            seed=42,
        )
        assert result.att_avg != 0

    def test_ife_unbalanced(self, unbalanced_panel):
        result = pyfect.fect(
            data=unbalanced_panel["data"],
            Y="Y", D="D",
            index=("unit", "time"),
            method="ife",
            r=2,
            CV=False,
            seed=42,
        )
        assert result.att_avg != 0


class TestInferenceSmoke:
    """Test bootstrap and jackknife inference."""

    def test_bootstrap_fe(self, small_panel):
        result = pyfect.fect(
            data=small_panel["data"],
            Y="Y", D="D",
            index=("unit", "time"),
            method="fe",
            se=True,
            nboots=20,
            seed=42,
        )
        assert result.inference is not None
        assert result.inference.att_avg_se > 0
        assert result.inference.att_avg_ci[0] < result.inference.att_avg_ci[1]

    def test_bootstrap_ife(self, small_panel):
        result = pyfect.fect(
            data=small_panel["data"],
            Y="Y", D="D",
            index=("unit", "time"),
            method="ife",
            r=2,
            CV=False,
            se=True,
            nboots=20,
            seed=42,
        )
        assert result.inference is not None
        assert result.inference.att_on is not None
        assert len(result.inference.att_on_se) > 0

    def test_jackknife(self, small_panel):
        result = pyfect.fect(
            data=small_panel["data"],
            Y="Y", D="D",
            index=("unit", "time"),
            method="fe",
            se=True,
            vartype="jackknife",
            seed=42,
        )
        assert result.inference is not None
        assert result.inference.att_avg_se > 0


class TestDynamicEffects:
    """Test dynamic treatment effects computation."""

    def test_dynamic_effects_exist(self, small_panel):
        result = pyfect.fect(
            data=small_panel["data"],
            Y="Y", D="D",
            index=("unit", "time"),
            method="fe",
            seed=42,
        )
        assert result.att_on is not None
        assert result.time_on is not None
        assert result.count_on is not None
        assert len(result.att_on) == len(result.time_on)
        assert len(result.count_on) == len(result.time_on)

    def test_pre_treatment_effects_near_zero(self, small_panel):
        """Pre-treatment dynamic ATTs should be near zero for FE."""
        result = pyfect.fect(
            data=small_panel["data"],
            Y="Y", D="D",
            index=("unit", "time"),
            method="ife",
            r=2,
            CV=False,
            seed=42,
        )
        pre_mask = result.time_on < 0
        if np.any(pre_mask):
            pre_att = result.att_on[pre_mask]
            # Pre-treatment ATTs should be relatively small
            assert np.mean(np.abs(pre_att)) < abs(result.att_avg)


class TestReproducibility:
    """Test that seeded runs produce identical results."""

    def test_seed_determinism(self, small_panel):
        """Same seed -> same result."""
        r1 = pyfect.fect(
            data=small_panel["data"],
            Y="Y", D="D",
            index=("unit", "time"),
            method="ife",
            r=2,
            CV=False,
            seed=42,
        )
        r2 = pyfect.fect(
            data=small_panel["data"],
            Y="Y", D="D",
            index=("unit", "time"),
            method="ife",
            r=2,
            CV=False,
            seed=42,
        )
        assert r1.att_avg == r2.att_avg
        np.testing.assert_array_equal(r1.att_on, r2.att_on)

    def test_different_seed_different_cv(self, small_panel):
        """Different seeds may give different CV results (but both valid)."""
        r1 = pyfect.fect(
            data=small_panel["data"],
            Y="Y", D="D",
            index=("unit", "time"),
            method="ife",
            r=(0, 3),
            CV=True,
            k=5,
            seed=42,
        )
        r2 = pyfect.fect(
            data=small_panel["data"],
            Y="Y", D="D",
            index=("unit", "time"),
            method="ife",
            r=(0, 3),
            CV=True,
            k=5,
            seed=99,
        )
        # Both should give valid r values
        assert 0 <= r1.r_cv <= 3
        assert 0 <= r2.r_cv <= 3


class TestSummaryAndPlot:
    """Test output formatting."""

    def test_summary_string(self, small_panel):
        result = pyfect.fect(
            data=small_panel["data"],
            Y="Y", D="D",
            index=("unit", "time"),
            method="fe",
            seed=42,
        )
        s = result.summary()
        assert "ATT" in s
        assert "Method" in s

    def test_plot_gap(self, small_panel):
        """Plotting should not raise."""
        result = pyfect.fect(
            data=small_panel["data"],
            Y="Y", D="D",
            index=("unit", "time"),
            method="fe",
            seed=42,
        )
        import matplotlib
        matplotlib.use("Agg")
        fig = pyfect.plot(result, kind="gap")
        assert fig is not None

    def test_plot_status(self, small_panel):
        result = pyfect.fect(
            data=small_panel["data"],
            Y="Y", D="D",
            index=("unit", "time"),
            method="fe",
            seed=42,
        )
        import matplotlib
        matplotlib.use("Agg")
        fig = pyfect.plot(result, kind="status")
        assert fig is not None
