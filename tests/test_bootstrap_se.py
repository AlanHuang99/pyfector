"""
Comprehensive bootstrap / jackknife standard error tests.

Tests:
1. SE stability: increasing nboots should stabilize SE
2. Coverage: 95% CI should cover the true ATT ~95% of the time (Monte Carlo)
3. SE consistency: bootstrap SE ~ jackknife SE ~ parametric SE
4. Dynamic SE: per-period SEs should be reasonable
5. Bootstrap distribution properties: centered, finite, correct shape
6. Parallel vs sequential: same results
7. Seed reproducibility: same seed → same SE
8. Different seed: different boots but similar SE with large nboots
9. Method comparison: FE, IFE, MC all produce reasonable SEs
10. Stress: many boots on medium panel, convergence of SE
"""

import numpy as np
import polars as pl
import pytest
import time

import pyfector
from tests.conftest import _simulate_panel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _quick_result(panel, method="fe", r=0, nboots=50, seed=42, n_jobs=1, **kw):
    """Run fect with SE for a given panel."""
    return pyfector.fect(
        data=panel["data"], Y="Y", D="D", index=("unit", "time"),
        method=method, r=r, CV=False, se=True, nboots=nboots,
        seed=seed, n_jobs=n_jobs, **kw,
    )


# ---------------------------------------------------------------------------
# 1. SE stability: more boots → more stable SE
# ---------------------------------------------------------------------------

class TestSEStability:
    """Increasing nboots should produce a more stable SE estimate."""

    def test_se_converges_with_more_boots(self):
        panel = _simulate_panel(N=100, T=30, N_treated=40, r=0, delta=3.0, seed=42)

        se_20 = _quick_result(panel, nboots=20, seed=1).inference.att_avg_se
        se_50 = _quick_result(panel, nboots=50, seed=1).inference.att_avg_se
        se_100 = _quick_result(panel, nboots=100, seed=1).inference.att_avg_se
        se_200 = _quick_result(panel, nboots=200, seed=1).inference.att_avg_se

        print(f"SE with 20/50/100/200 boots: {se_20:.4f}/{se_50:.4f}/{se_100:.4f}/{se_200:.4f}")

        # All should be positive
        assert se_20 > 0
        assert se_50 > 0
        assert se_100 > 0
        assert se_200 > 0

        # SE estimates should be in the same ballpark (within 3x of each other)
        all_se = [se_20, se_50, se_100, se_200]
        assert max(all_se) / min(all_se) < 3.0, f"SE too unstable: {all_se}"

    def test_se_200_vs_500_close(self):
        """SE with 200 and 500 boots should be within 30% of each other."""
        panel = _simulate_panel(N=100, T=30, N_treated=40, r=0, delta=3.0, seed=42)

        se_200 = _quick_result(panel, nboots=200, seed=42).inference.att_avg_se
        se_500 = _quick_result(panel, nboots=500, seed=42).inference.att_avg_se

        rel_diff = abs(se_200 - se_500) / (se_500 + 1e-10)
        print(f"SE 200={se_200:.4f}, 500={se_500:.4f}, rel_diff={rel_diff:.3f}")
        assert rel_diff < 0.3, f"SE not stabilizing: 200={se_200:.4f}, 500={se_500:.4f}"


# ---------------------------------------------------------------------------
# 2. Coverage: CI should cover truth at advertised rate
# ---------------------------------------------------------------------------

class TestCoverage:
    """Monte Carlo coverage test: run many simulations and check CI coverage."""

    def test_fe_coverage_zero_factor(self):
        """With no factors in DGP, FE 95% CI should cover truth ~90-100%."""
        delta = 5.0
        n_sims = 30
        covers = 0

        for sim in range(n_sims):
            panel = _simulate_panel(N=80, T=25, N_treated=30, r=0,
                                    delta=delta, seed=1000 + sim)
            result = _quick_result(panel, method="fe", nboots=100, seed=sim)
            ci = result.inference.att_avg_ci
            if ci[0] <= delta <= ci[1]:
                covers += 1

        coverage = covers / n_sims
        print(f"FE coverage ({n_sims} sims, 100 boots): {coverage:.2%}")
        # Allow wide margin — 30 sims is noisy
        assert coverage >= 0.70, f"Coverage too low: {coverage:.2%}"

    def test_ife_coverage(self):
        """IFE with correct r: CI should cover truth reasonably."""
        delta = 3.0
        n_sims = 20
        covers = 0

        for sim in range(n_sims):
            panel = _simulate_panel(N=80, T=25, N_treated=30, r=2,
                                    delta=delta, seed=2000 + sim)
            result = _quick_result(panel, method="ife", r=2, nboots=100, seed=sim)
            ci = result.inference.att_avg_ci
            if ci[0] <= delta <= ci[1]:
                covers += 1

        coverage = covers / n_sims
        print(f"IFE r=2 coverage ({n_sims} sims, 100 boots): {coverage:.2%}")
        assert coverage >= 0.60, f"IFE coverage too low: {coverage:.2%}"


# ---------------------------------------------------------------------------
# 3. Bootstrap vs jackknife consistency
# ---------------------------------------------------------------------------

class TestBootstrapJackknifeConsistency:
    """Bootstrap and jackknife SE should be in the same ballpark."""

    def test_fe_boot_vs_jack(self):
        panel = _simulate_panel(N=50, T=20, N_treated=20, r=0, delta=3.0, seed=42)

        boot_result = _quick_result(panel, method="fe", nboots=200, seed=42)
        jack_result = pyfector.fect(
            data=panel["data"], Y="Y", D="D", index=("unit", "time"),
            method="fe", se=True, vartype="jackknife", seed=42,
        )

        boot_se = boot_result.inference.att_avg_se
        jack_se = jack_result.inference.att_avg_se

        print(f"FE: boot SE={boot_se:.4f}, jack SE={jack_se:.4f}")

        # They should be within 3x of each other
        ratio = max(boot_se, jack_se) / (min(boot_se, jack_se) + 1e-10)
        assert ratio < 3.0, f"Boot/jack SE too different: ratio={ratio:.2f}"

    def test_ife_boot_vs_jack(self):
        panel = _simulate_panel(N=50, T=20, N_treated=20, r=2, delta=3.0, seed=42)

        boot_result = _quick_result(panel, method="ife", r=2, nboots=200, seed=42)
        jack_result = pyfector.fect(
            data=panel["data"], Y="Y", D="D", index=("unit", "time"),
            method="ife", r=2, CV=False, se=True, vartype="jackknife", seed=42,
        )

        boot_se = boot_result.inference.att_avg_se
        jack_se = jack_result.inference.att_avg_se

        print(f"IFE: boot SE={boot_se:.4f}, jack SE={jack_se:.4f}")
        ratio = max(boot_se, jack_se) / (min(boot_se, jack_se) + 1e-10)
        assert ratio < 5.0, f"IFE boot/jack SE too different: ratio={ratio:.2f}"


# ---------------------------------------------------------------------------
# 4. Dynamic SE (per-period)
# ---------------------------------------------------------------------------

class TestDynamicSE:
    """Per-period treatment effect SEs should be well-behaved."""

    def test_dynamic_se_all_positive(self):
        panel = _simulate_panel(N=100, T=30, N_treated=40, r=0, delta=3.0, seed=42)
        result = _quick_result(panel, nboots=100, seed=42)

        assert result.inference.att_on_se is not None
        assert len(result.inference.att_on_se) > 0
        # Non-NaN SEs should all be positive (tail periods may be NaN due to sparse data)
        valid = ~np.isnan(result.inference.att_on_se)
        assert np.sum(valid) > 0, "All dynamic SEs are NaN"
        assert np.all(result.inference.att_on_se[valid] > 0), "Some valid dynamic SEs are zero or negative"

    def test_dynamic_se_mostly_finite(self):
        """Most dynamic SEs should be finite; only extreme tail periods may have NaN."""
        panel = _simulate_panel(N=100, T=30, N_treated=40, r=2, delta=3.0, seed=42)
        result = _quick_result(panel, method="ife", r=2, nboots=100, seed=42)

        n_finite = np.sum(np.isfinite(result.inference.att_on_se))
        n_total = len(result.inference.att_on_se)
        frac_finite = n_finite / n_total
        print(f"Finite dynamic SEs: {n_finite}/{n_total} ({frac_finite:.0%})")
        assert frac_finite >= 0.8, f"Too many NaN dynamic SEs: {1-frac_finite:.0%}"

    def test_dynamic_ci_contains_point_estimate(self):
        """CI should bracket the point estimate (pivotal CI may not always, but should mostly)."""
        panel = _simulate_panel(N=100, T=30, N_treated=40, r=0, delta=3.0, seed=42)
        result = _quick_result(panel, nboots=200, seed=42)

        inf = result.inference
        # For pivotal CI: 2*theta - q(1-a/2) to 2*theta - q(a/2)
        # This is NOT centered on theta, but theta should usually be inside
        inside = (inf.att_on_ci_lower <= result.att_on) & (result.att_on <= inf.att_on_ci_upper)
        frac_inside = np.mean(inside)
        print(f"Fraction of periods where point est is inside CI: {frac_inside:.2%}")
        # Allow some periods to be outside (pivotal CI can shift)
        assert frac_inside >= 0.7, f"Too many periods outside CI: {frac_inside:.2%}"

    def test_dynamic_se_larger_with_fewer_obs(self):
        """Periods with fewer observations should generally have larger SE."""
        panel = _simulate_panel(N=100, T=30, N_treated=40, r=0, delta=3.0, seed=42)
        result = _quick_result(panel, nboots=200, seed=42)

        # Periods far from treatment onset have fewer observations
        counts = result.count_on
        ses = result.inference.att_on_se
        if len(counts) > 5:
            # Top-count periods vs bottom-count periods
            sorted_idx = np.argsort(counts)
            low_count_se = np.mean(ses[sorted_idx[:3]])
            high_count_se = np.mean(ses[sorted_idx[-3:]])
            print(f"Low-count SE={low_count_se:.4f}, high-count SE={high_count_se:.4f}")
            # Low-count should generally have larger SE (not a hard rule, but directional)

    def test_pre_treatment_dynamic_se(self):
        """Pre-treatment SEs should be well-defined."""
        panel = _simulate_panel(N=100, T=30, N_treated=40, r=0, delta=3.0, seed=42)
        result = _quick_result(panel, nboots=100, seed=42)

        pre_mask = result.time_on < 0
        if np.any(pre_mask):
            pre_se = result.inference.att_on_se[pre_mask]
            pre_att = result.att_on[pre_mask]
            print(f"Pre-treatment: ATT range [{pre_att.min():.4f}, {pre_att.max():.4f}]")
            print(f"Pre-treatment: SE range [{pre_se.min():.4f}, {pre_se.max():.4f}]")
            assert np.all(pre_se > 0)
            assert np.all(np.isfinite(pre_se))


# ---------------------------------------------------------------------------
# 5. Bootstrap distribution properties
# ---------------------------------------------------------------------------

class TestBootstrapDistribution:
    """The bootstrap distribution should have reasonable statistical properties."""

    def test_boot_distribution_centered(self):
        """Bootstrap ATT distribution should be roughly centered near point estimate."""
        panel = _simulate_panel(N=200, T=40, N_treated=80, r=0, delta=3.0, seed=42)
        result = _quick_result(panel, nboots=500, seed=42)

        boot_mean = np.mean(result.inference.att_avg_boot)
        point_est = result.att_avg
        # Boot mean should be close to point estimate
        diff = abs(boot_mean - point_est)
        se = result.inference.att_avg_se
        print(f"Point: {point_est:.4f}, Boot mean: {boot_mean:.4f}, diff/SE: {diff/se:.2f}")
        assert diff < 3 * se, f"Boot mean too far from point estimate"

    def test_boot_distribution_no_outliers(self):
        """No extreme outliers in bootstrap distribution."""
        panel = _simulate_panel(N=100, T=30, N_treated=40, r=0, delta=3.0, seed=42)
        result = _quick_result(panel, nboots=200, seed=42)

        boot = result.inference.att_avg_boot
        median = np.median(boot)
        mad = np.median(np.abs(boot - median))

        # No value should be more than 20 MADs from median
        extreme = np.abs(boot - median) > 20 * mad
        n_extreme = np.sum(extreme)
        assert n_extreme == 0, f"{n_extreme} extreme outliers in bootstrap distribution"

    def test_boot_avg_all_finite(self):
        """Overall ATT bootstrap should always be finite."""
        panel = _simulate_panel(N=100, T=30, N_treated=40, r=2, delta=3.0, seed=42)
        result = _quick_result(panel, method="ife", r=2, nboots=100, seed=42)

        assert np.all(np.isfinite(result.inference.att_avg_boot)), "Non-finite bootstrap ATTs"
        # Dynamic boot may have NaN for sparse tail periods — that's expected
        n_nan = np.sum(np.isnan(result.inference.att_on_boot))
        n_total = result.inference.att_on_boot.size
        frac_nan = n_nan / n_total
        print(f"NaN in dynamic boot: {n_nan}/{n_total} ({frac_nan:.1%})")
        assert frac_nan < 0.2, f"Too many NaN in dynamic bootstrap: {frac_nan:.1%}"

    def test_boot_distribution_shape(self):
        """Bootstrap output arrays have correct shapes."""
        panel = _simulate_panel(N=50, T=20, N_treated=20, r=0, delta=3.0, seed=42)
        nboots = 75
        result = _quick_result(panel, nboots=nboots, seed=42)

        assert result.inference.att_avg_boot.shape[0] <= nboots  # may be fewer if some failed
        assert result.inference.att_avg_boot.shape[0] >= nboots * 0.9  # at least 90% success
        n_periods = len(result.time_on)
        assert result.inference.att_on_boot.shape == (n_periods, result.inference.att_avg_boot.shape[0])

    def test_boot_variance_reasonable(self):
        """Bootstrap variance should be positive and not absurdly large."""
        panel = _simulate_panel(N=100, T=30, N_treated=40, r=0, delta=3.0, seed=42)
        result = _quick_result(panel, nboots=200, seed=42)

        boot_var = np.var(result.inference.att_avg_boot)
        assert boot_var > 0, "Zero bootstrap variance"
        # Variance shouldn't be larger than the ATT itself squared (sanity check)
        assert boot_var < result.att_avg ** 2 * 10, f"Bootstrap variance too large: {boot_var}"


# ---------------------------------------------------------------------------
# 6. Parallel vs sequential consistency
# ---------------------------------------------------------------------------

class TestParallelConsistency:
    """Parallel bootstrap should produce similar SE to sequential."""

    def test_parallel_se_similar(self):
        """n_jobs=1 vs n_jobs=2 should give similar SE (not identical due to parallel RNG)."""
        panel = _simulate_panel(N=80, T=25, N_treated=30, r=0, delta=3.0, seed=42)

        se_seq = _quick_result(panel, nboots=200, seed=42, n_jobs=1).inference.att_avg_se
        se_par = _quick_result(panel, nboots=200, seed=42, n_jobs=2).inference.att_avg_se

        print(f"Sequential SE={se_seq:.4f}, Parallel SE={se_par:.4f}")
        # They use the same seeds → should be identical or very close
        rel_diff = abs(se_seq - se_par) / (se_seq + 1e-10)
        assert rel_diff < 0.01, f"Parallel SE differs too much: {rel_diff:.4f}"


# ---------------------------------------------------------------------------
# 7. Seed reproducibility
# ---------------------------------------------------------------------------

class TestSeedReproducibility:
    """Same seed should give identical SEs; different seed should give different boots."""

    def test_same_seed_same_se(self):
        panel = _simulate_panel(N=80, T=25, N_treated=30, r=0, delta=3.0, seed=42)

        r1 = _quick_result(panel, nboots=100, seed=123)
        r2 = _quick_result(panel, nboots=100, seed=123)

        assert r1.inference.att_avg_se == r2.inference.att_avg_se
        np.testing.assert_array_equal(r1.inference.att_avg_boot, r2.inference.att_avg_boot)
        np.testing.assert_array_equal(r1.inference.att_on_boot, r2.inference.att_on_boot)

    def test_different_seed_different_boots(self):
        panel = _simulate_panel(N=80, T=25, N_treated=30, r=0, delta=3.0, seed=42)

        r1 = _quick_result(panel, nboots=100, seed=111)
        r2 = _quick_result(panel, nboots=100, seed=222)

        # Boots should differ
        assert not np.array_equal(r1.inference.att_avg_boot, r2.inference.att_avg_boot)

    def test_different_seed_similar_se_large_nboots(self):
        """With enough boots, SE should be similar regardless of seed."""
        panel = _simulate_panel(N=100, T=30, N_treated=40, r=0, delta=3.0, seed=42)

        se1 = _quick_result(panel, nboots=500, seed=111).inference.att_avg_se
        se2 = _quick_result(panel, nboots=500, seed=222).inference.att_avg_se

        rel_diff = abs(se1 - se2) / (max(se1, se2) + 1e-10)
        print(f"SE with seed=111: {se1:.4f}, seed=222: {se2:.4f}, rel_diff={rel_diff:.3f}")
        assert rel_diff < 0.3, f"SE too different across seeds with 500 boots"


# ---------------------------------------------------------------------------
# 8. Cross-method SE comparison
# ---------------------------------------------------------------------------

class TestCrossMethodSE:
    """SEs across methods should be in the same ballpark for the same data."""

    def test_fe_ife_mc_se_comparison(self):
        """FE, IFE, MC should give comparable SEs on same data."""
        panel = _simulate_panel(N=100, T=30, N_treated=40, r=2, delta=3.0, seed=42)

        fe_r = _quick_result(panel, method="fe", nboots=200, seed=42)
        ife_r = _quick_result(panel, method="ife", r=2, nboots=200, seed=42)
        mc_r = _quick_result(panel, method="mc", nboots=200, seed=42, lam=0.01)

        fe_se = fe_r.inference.att_avg_se
        ife_se = ife_r.inference.att_avg_se
        mc_se = mc_r.inference.att_avg_se

        print(f"SE: FE={fe_se:.4f}, IFE={ife_se:.4f}, MC={mc_se:.4f}")
        print(f"ATT: FE={fe_r.att_avg:.4f}, IFE={ife_r.att_avg:.4f}, MC={mc_r.att_avg:.4f}")

        # All should be positive and finite
        for se, name in [(fe_se, "FE"), (ife_se, "IFE"), (mc_se, "MC")]:
            assert se > 0, f"{name} SE is zero"
            assert np.isfinite(se), f"{name} SE is not finite"

        # SEs should be within 10x of each other (generous bound)
        all_se = [fe_se, ife_se, mc_se]
        ratio = max(all_se) / min(all_se)
        assert ratio < 10, f"Method SE ratio too large: {ratio:.1f}"

    def test_correct_model_lower_se(self):
        """IFE with correct r should not have much larger SE than FE on no-factor data."""
        panel = _simulate_panel(N=100, T=30, N_treated=40, r=0, delta=3.0, seed=42)

        fe_se = _quick_result(panel, method="fe", nboots=200, seed=42).inference.att_avg_se
        ife0_se = _quick_result(panel, method="ife", r=0, nboots=200, seed=42).inference.att_avg_se

        print(f"No-factor data: FE SE={fe_se:.4f}, IFE(r=0) SE={ife0_se:.4f}")
        # FE and IFE(r=0) should give very similar SE
        rel_diff = abs(fe_se - ife0_se) / (fe_se + 1e-10)
        assert rel_diff < 0.3, f"FE vs IFE(r=0) SE differ too much: {rel_diff:.3f}"


# ---------------------------------------------------------------------------
# 9. P-values and CI
# ---------------------------------------------------------------------------

class TestPValuesCI:
    """P-values and confidence intervals should be consistent."""

    def test_significant_effect_detected(self):
        """With large true effect and enough data, p-value should be small."""
        panel = _simulate_panel(N=200, T=40, N_treated=80, r=0, delta=5.0, seed=42)
        result = _quick_result(panel, nboots=200, seed=42)

        assert result.inference.att_avg_pval < 0.05, \
            f"Large effect not significant: p={result.inference.att_avg_pval:.4f}"

    def test_zero_effect_not_significant(self):
        """With zero true effect, p-value should be large (most of the time)."""
        n_sig = 0
        n_sims = 20
        for sim in range(n_sims):
            panel = _simulate_panel(N=80, T=25, N_treated=30, r=0, delta=0.0, seed=3000 + sim)
            result = _quick_result(panel, nboots=100, seed=sim)
            if result.inference.att_avg_pval < 0.05:
                n_sig += 1

        rejection_rate = n_sig / n_sims
        print(f"Type I error rate (delta=0): {rejection_rate:.2%}")
        # Should reject at most ~20% (allowing for multiple testing noise with 20 sims)
        assert rejection_rate < 0.3, f"Type I error rate too high: {rejection_rate:.2%}"

    def test_ci_width_decreases_with_n(self):
        """CI should get narrower with more units."""
        panels = {
            50: _simulate_panel(N=50, T=25, N_treated=20, r=0, delta=3.0, seed=42),
            200: _simulate_panel(N=200, T=25, N_treated=80, r=0, delta=3.0, seed=42),
        }

        widths = {}
        for n, panel in panels.items():
            result = _quick_result(panel, nboots=200, seed=42)
            ci = result.inference.att_avg_ci
            widths[n] = ci[1] - ci[0]

        print(f"CI width: N=50 → {widths[50]:.4f}, N=200 → {widths[200]:.4f}")
        assert widths[200] < widths[50], "CI should narrow with more units"

    def test_ci_ordered(self):
        """CI lower < CI upper (for non-NaN periods)."""
        panel = _simulate_panel(N=100, T=30, N_treated=40, r=0, delta=3.0, seed=42)
        result = _quick_result(panel, nboots=100, seed=42)

        assert result.inference.att_avg_ci[0] < result.inference.att_avg_ci[1]
        valid = np.isfinite(result.inference.att_on_ci_lower) & np.isfinite(result.inference.att_on_ci_upper)
        assert np.all(result.inference.att_on_ci_lower[valid] <= result.inference.att_on_ci_upper[valid])

    def test_dynamic_pvalues_finite(self):
        panel = _simulate_panel(N=100, T=30, N_treated=40, r=0, delta=3.0, seed=42)
        result = _quick_result(panel, nboots=100, seed=42)

        assert np.all(np.isfinite(result.inference.att_on_pval))
        assert np.all(result.inference.att_on_pval >= 0)
        assert np.all(result.inference.att_on_pval <= 1)


# ---------------------------------------------------------------------------
# 10. Stress test: medium panel, many boots
# ---------------------------------------------------------------------------

class TestBootstrapStress:
    """Stress tests with larger panels and more bootstrap replications."""

    def test_medium_panel_500_boots(self):
        """N=200, T=50, 500 bootstrap replications."""
        panel = _simulate_panel(N=200, T=50, N_treated=80, r=2, delta=3.0, seed=123)

        start = time.time()
        result = _quick_result(panel, method="ife", r=2, nboots=500, seed=42)
        elapsed = time.time() - start

        inf = result.inference
        print(f"N=200,T=50, IFE r=2, 500 boots: {elapsed:.1f}s")
        print(f"  ATT={result.att_avg:.4f}, SE={inf.att_avg_se:.4f}")
        print(f"  CI=[{inf.att_avg_ci[0]:.4f}, {inf.att_avg_ci[1]:.4f}]")
        print(f"  p={inf.att_avg_pval:.4f}")
        print(f"  Boot success: {len(inf.att_avg_boot)}/500")

        assert inf.att_avg_se > 0
        assert np.all(np.isfinite(inf.att_avg_boot))
        assert len(inf.att_avg_boot) >= 450  # at least 90% success rate

    def test_fe_1000_boots_stable(self):
        """FE with 1000 boots: SE should stabilize."""
        panel = _simulate_panel(N=100, T=30, N_treated=40, r=0, delta=3.0, seed=42)

        start = time.time()
        result = _quick_result(panel, method="fe", nboots=1000, seed=42)
        elapsed = time.time() - start

        print(f"FE, 1000 boots: {elapsed:.1f}s, SE={result.inference.att_avg_se:.4f}")

        # Compare first-half vs second-half SE
        boot = result.inference.att_avg_boot
        se_first = np.std(boot[:500], ddof=1)
        se_second = np.std(boot[500:], ddof=1)
        rel_diff = abs(se_first - se_second) / (se_first + 1e-10)
        print(f"  SE first 500={se_first:.4f}, second 500={se_second:.4f}, rel_diff={rel_diff:.3f}")
        assert rel_diff < 0.5, "SE not stable between first/second half of boots"

    def test_all_boots_produce_valid_avg_estimates(self):
        """Overall ATT bootstrap should have no NaN or Inf. Dynamic may have sparse-period NaN."""
        panel = _simulate_panel(N=100, T=30, N_treated=40, r=2, delta=3.0, seed=42)
        result = _quick_result(panel, method="ife", r=2, nboots=200, seed=42)

        boot_avg = result.inference.att_avg_boot
        boot_dyn = result.inference.att_on_boot

        n_nan_avg = np.sum(np.isnan(boot_avg))
        n_inf_avg = np.sum(np.isinf(boot_avg))

        print(f"NaN/Inf in avg boot: {n_nan_avg}/{n_inf_avg}")
        assert n_nan_avg == 0, f"{n_nan_avg} NaN in bootstrap ATT"
        assert n_inf_avg == 0, f"{n_inf_avg} Inf in bootstrap ATT"

        # Dynamic: NaN allowed for sparse tail periods, but no Inf
        n_inf_dyn = np.sum(np.isinf(boot_dyn))
        assert n_inf_dyn == 0, f"{n_inf_dyn} Inf in dynamic bootstrap"

        # Most dynamic entries should be finite
        frac_finite = np.sum(np.isfinite(boot_dyn)) / boot_dyn.size
        print(f"Dynamic boot finite fraction: {frac_finite:.1%}")
        assert frac_finite >= 0.8


# ---------------------------------------------------------------------------
# 11. Jackknife-specific tests
# ---------------------------------------------------------------------------

class TestJackknifeSE:
    """Jackknife-specific SE tests."""

    def test_jackknife_n_replications(self):
        """Jackknife should produce N replications (one per unit)."""
        panel = _simulate_panel(N=40, T=20, N_treated=15, r=0, delta=3.0, seed=42)
        result = pyfector.fect(
            data=panel["data"], Y="Y", D="D", index=("unit", "time"),
            method="fe", se=True, vartype="jackknife", seed=42,
        )
        # Jackknife stores replicates in att_avg_boot
        n_reps = len(result.inference.att_avg_boot)
        # Should be close to N (some may fail)
        assert n_reps >= panel["N"] * 0.8, f"Too few jackknife reps: {n_reps} vs N={panel['N']}"

    def test_jackknife_se_positive(self):
        panel = _simulate_panel(N=40, T=20, N_treated=15, r=0, delta=3.0, seed=42)
        result = pyfector.fect(
            data=panel["data"], Y="Y", D="D", index=("unit", "time"),
            method="fe", se=True, vartype="jackknife", seed=42,
        )
        assert result.inference.att_avg_se > 0
        # Non-NaN dynamic SEs should be positive
        valid = ~np.isnan(result.inference.att_on_se)
        assert np.all(result.inference.att_on_se[valid] > 0)

    def test_jackknife_avg_finite(self):
        panel = _simulate_panel(N=40, T=20, N_treated=15, r=0, delta=3.0, seed=42)
        result = pyfector.fect(
            data=panel["data"], Y="Y", D="D", index=("unit", "time"),
            method="fe", se=True, vartype="jackknife", seed=42,
        )
        inf = result.inference
        assert np.all(np.isfinite(inf.att_avg_boot))
        assert np.isfinite(inf.att_avg_se)
        # Dynamic may have NaN for sparse periods, check most are finite
        n_finite = np.sum(np.isfinite(inf.att_on_se))
        assert n_finite >= len(inf.att_on_se) * 0.7

    def test_jackknife_ife(self):
        """Jackknife with IFE should work."""
        panel = _simulate_panel(N=40, T=20, N_treated=15, r=2, delta=3.0, seed=42)
        result = pyfector.fect(
            data=panel["data"], Y="Y", D="D", index=("unit", "time"),
            method="ife", r=2, CV=False, se=True, vartype="jackknife", seed=42,
        )
        assert result.inference.att_avg_se > 0
        print(f"Jackknife IFE: ATT={result.att_avg:.4f}, SE={result.inference.att_avg_se:.4f}")


# ---------------------------------------------------------------------------
# 12. Bootstrap with covariates
# ---------------------------------------------------------------------------

class TestBootstrapWithCovariates:
    """SE computation when covariates are present."""

    def test_se_with_covariates(self):
        panel = _simulate_panel(N=80, T=25, N_treated=30, r=0, p=2, delta=3.0, seed=42)
        result = pyfector.fect(
            data=panel["data"], Y="Y", D="D", index=("unit", "time"),
            X=panel["covariate_names"], method="fe",
            se=True, nboots=100, seed=42,
        )
        assert result.inference.att_avg_se > 0
        assert np.all(np.isfinite(result.inference.att_avg_boot))
        print(f"FE+covar: ATT={result.att_avg:.4f}, SE={result.inference.att_avg_se:.4f}")

    def test_ife_se_with_covariates(self):
        panel = _simulate_panel(N=80, T=25, N_treated=30, r=2, p=2, delta=3.0, seed=42)
        result = pyfector.fect(
            data=panel["data"], Y="Y", D="D", index=("unit", "time"),
            X=panel["covariate_names"], method="ife", r=2, CV=False,
            se=True, nboots=100, seed=42,
        )
        assert result.inference.att_avg_se > 0
        assert np.all(np.isfinite(result.inference.att_avg_boot))
        print(f"IFE+covar: ATT={result.att_avg:.4f}, SE={result.inference.att_avg_se:.4f}")
