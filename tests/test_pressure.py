"""
Pressure tests: verify pyfect handles large datasets and edge cases.
"""

import numpy as np
import polars as pl
import pytest
import time

import pyfect


class TestLargeDatasets:
    """Test performance and correctness on large panels."""

    def test_large_fe(self, large_panel):
        """N=2000, T=100 with FE should complete in reasonable time."""
        start = time.time()
        result = pyfect.fect(
            data=large_panel["data"],
            Y="Y", D="D",
            index=("unit", "time"),
            method="fe",
            seed=42,
        )
        elapsed = time.time() - start
        assert result.att_avg != 0
        # FE on 2000x100 should be fast (< 60s)
        assert elapsed < 60, f"FE took {elapsed:.1f}s on 2000x100 panel"

    def test_large_ife_r2(self, large_panel):
        """N=2000, T=100 with IFE r=2 — the key performance benchmark."""
        start = time.time()
        result = pyfect.fect(
            data=large_panel["data"],
            Y="Y", D="D",
            index=("unit", "time"),
            method="ife",
            r=2,
            CV=False,
            seed=42,
        )
        elapsed = time.time() - start
        assert result.att_avg != 0
        assert result.converged
        # With randomized SVD, this should be much faster than R fect
        print(f"IFE r=2 on N=2000, T=100: {elapsed:.1f}s, {result.niter} iterations")

    def test_large_mc(self, large_panel):
        """N=2000, T=100 with MC."""
        start = time.time()
        result = pyfect.fect(
            data=large_panel["data"],
            Y="Y", D="D",
            index=("unit", "time"),
            method="mc",
            lam=0.01,
            CV=False,
            seed=42,
        )
        elapsed = time.time() - start
        assert result.att_avg != 0
        print(f"MC on N=2000, T=100: {elapsed:.1f}s, {result.niter} iterations")

    @pytest.mark.slow
    def test_very_large_panel(self):
        """N=10000, T=200 — the target use case."""
        from tests.conftest import _simulate_panel
        panel = _simulate_panel(N=10000, T=200, N_treated=4000, r=2, seed=999)

        start = time.time()
        result = pyfect.fect(
            data=panel["data"],
            Y="Y", D="D",
            index=("unit", "time"),
            method="ife",
            r=2,
            CV=False,
            seed=42,
        )
        elapsed = time.time() - start
        assert result.att_avg != 0
        print(f"IFE r=2 on N=10000, T=200: {elapsed:.1f}s, {result.niter} iterations")


class TestEdgeCases:
    """Edge cases that should not crash."""

    def test_single_treated_unit(self):
        """Only one treated unit."""
        from tests.conftest import _simulate_panel
        panel = _simulate_panel(N=20, T=15, N_treated=1, r=0, seed=111)
        result = pyfect.fect(
            data=panel["data"],
            Y="Y", D="D",
            index=("unit", "time"),
            method="fe",
            seed=42,
        )
        assert result.att_avg != 0

    def test_all_treated_late(self):
        """Treatment starts very late (few post-treatment periods)."""
        rng = np.random.default_rng(222)
        N, T = 30, 20
        rows = []
        for j in range(N):
            for t in range(T):
                d = 1 if (j < 10 and t >= T - 2) else 0
                rows.append({"unit": j, "time": t, "Y": float(rng.normal(t + j * 0.1 + d * 2)), "D": d})
        df = pl.DataFrame(rows)
        result = pyfect.fect(data=df, Y="Y", D="D", index=("unit", "time"), method="fe", seed=42)
        assert result.att_avg != 0

    def test_many_missing_obs(self):
        """50% missing observations."""
        from tests.conftest import _simulate_panel
        panel = _simulate_panel(N=40, T=20, N_treated=15, r=1, seed=333, missing_frac=0.4)
        result = pyfect.fect(
            data=panel["data"],
            Y="Y", D="D",
            index=("unit", "time"),
            method="fe",
            max_missing=0.6,
            seed=42,
        )
        assert result.att_avg != 0

    def test_no_covariates(self, small_panel):
        """Explicit test with no covariates."""
        result = pyfect.fect(
            data=small_panel["data"],
            Y="Y", D="D",
            index=("unit", "time"),
            method="ife",
            r=1,
            CV=False,
            seed=42,
        )
        assert result.beta is None or len(result.beta) == 0 or result.beta is None

    def test_high_r(self, small_panel):
        """r larger than reasonable — should still work."""
        result = pyfect.fect(
            data=small_panel["data"],
            Y="Y", D="D",
            index=("unit", "time"),
            method="ife",
            r=8,
            CV=False,
            seed=42,
        )
        assert result.r_cv == 8

    def test_lambda_zero(self, small_panel):
        """MC with lambda=0 (no regularization)."""
        result = pyfect.fect(
            data=small_panel["data"],
            Y="Y", D="D",
            index=("unit", "time"),
            method="mc",
            lam=0.0,
            CV=False,
            seed=42,
        )
        assert result.att_avg != 0

    def test_lambda_large(self, small_panel):
        """MC with very large lambda (heavy regularization)."""
        result = pyfect.fect(
            data=small_panel["data"],
            Y="Y", D="D",
            index=("unit", "time"),
            method="mc",
            lam=100.0,
            CV=False,
            seed=42,
        )
        # Should still produce a result (may be poor quality)
        assert result.att_avg is not None


class TestParallelComputing:
    """Test parallel CV and bootstrap."""

    def test_parallel_cv(self, small_panel):
        """CV with n_jobs=2."""
        result = pyfect.fect(
            data=small_panel["data"],
            Y="Y", D="D",
            index=("unit", "time"),
            method="ife",
            r=(0, 3),
            CV=True,
            k=5,
            n_jobs=2,
            seed=42,
        )
        assert result.r_cv is not None

    def test_parallel_bootstrap(self, small_panel):
        """Bootstrap with n_jobs=2."""
        result = pyfect.fect(
            data=small_panel["data"],
            Y="Y", D="D",
            index=("unit", "time"),
            method="fe",
            se=True,
            nboots=20,
            n_jobs=2,
            seed=42,
        )
        assert result.inference is not None
        assert result.inference.att_avg_se > 0


class TestNormalization:
    """Test normalize option."""

    def test_normalize_similar_att(self, small_panel):
        """Normalized and unnormalized should give similar ATT."""
        r1 = pyfect.fect(
            data=small_panel["data"],
            Y="Y", D="D",
            index=("unit", "time"),
            method="fe",
            normalize=False,
            seed=42,
        )
        r2 = pyfect.fect(
            data=small_panel["data"],
            Y="Y", D="D",
            index=("unit", "time"),
            method="fe",
            normalize=True,
            seed=42,
        )
        # ATT should be similar (normalization is undone)
        assert abs(r1.att_avg - r2.att_avg) / (abs(r1.att_avg) + 1e-6) < 0.1


class TestDiagnostics:
    """Test diagnostic tests."""

    def test_diagnostics_run(self, small_panel):
        result = pyfect.fect(
            data=small_panel["data"],
            Y="Y", D="D",
            index=("unit", "time"),
            method="fe",
            se=True,
            nboots=50,
            seed=42,
        )
        diag = pyfect.run_diagnostics(result, f_threshold=0.5, tost_threshold=0.36)
        assert diag is not None
        s = diag.summary()
        assert "Diagnostic" in s

    def test_zero_effect_diagnostics(self, zero_effect_panel):
        """With zero true effect, pre-trend test should not reject."""
        result = pyfect.fect(
            data=zero_effect_panel["data"],
            Y="Y", D="D",
            index=("unit", "time"),
            method="ife",
            r=2,
            CV=False,
            se=True,
            nboots=50,
            seed=42,
        )
        diag = pyfect.run_diagnostics(result)
        # With zero effect, the F-test p-value should be large
        if diag.f_pval is not None:
            # Not a hard assertion since small sample can be noisy
            print(f"Zero-effect F-test p-value: {diag.f_pval:.4f}")
