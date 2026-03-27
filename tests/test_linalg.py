"""
Unit tests for core linear algebra operations.
"""

import numpy as np
import pytest

from pyfect.linalg import (
    truncated_svd,
    panel_factor,
    panel_FE,
    demean,
    fe_add,
    panel_beta,
    compute_xxinv,
    covar_fit,
)


class TestTruncatedSVD:
    """Test randomized truncated SVD."""

    def test_rank1_reconstruction(self):
        """Rank-1 matrix should be perfectly recovered."""
        rng = np.random.default_rng(42)
        u = rng.normal(0, 1, (50, 1))
        v = rng.normal(0, 1, (1, 30))
        M = u @ v
        result = truncated_svd(M, r=1)
        recon = result.U * result.s[None, :] @ result.Vt
        np.testing.assert_allclose(M, recon, atol=1e-10)

    def test_low_rank_approximation(self):
        """Top-r SVD should capture most variance of a rank-r matrix + noise."""
        rng = np.random.default_rng(42)
        T, N, r = 100, 50, 3
        F = rng.normal(0, 1, (T, r))
        L = rng.normal(0, 1, (N, r))
        noise = rng.normal(0, 0.01, (T, N))
        M = F @ L.T + noise

        result = truncated_svd(M, r=3)
        recon = result.U * result.s[None, :] @ result.Vt
        rel_error = np.linalg.norm(M - recon, "fro") / np.linalg.norm(M, "fro")
        assert rel_error < 0.05  # should capture >95% of variance

    def test_shapes(self):
        rng = np.random.default_rng(42)
        M = rng.normal(0, 1, (80, 60))
        result = truncated_svd(M, r=5)
        assert result.U.shape == (80, 5)
        assert result.s.shape == (5,)
        assert result.Vt.shape == (5, 60)

    def test_singular_values_ordered(self):
        rng = np.random.default_rng(42)
        M = rng.normal(0, 1, (50, 40))
        result = truncated_svd(M, r=10)
        # Singular values should be in descending order
        assert np.all(np.diff(result.s) <= 1e-10)

    def test_fallback_to_exact_for_high_r(self):
        """When r is close to min(T,N), should still work."""
        rng = np.random.default_rng(42)
        M = rng.normal(0, 1, (20, 15))
        result = truncated_svd(M, r=14)
        assert result.U.shape[1] == 14


class TestPanelFactor:
    """Test factor extraction."""

    def test_known_factors(self):
        """Should recover known factor structure."""
        rng = np.random.default_rng(42)
        T, N, r = 50, 30, 2
        F_true = rng.normal(0, 1, (T, r))
        L_true = rng.normal(0, 1, (N, r))
        E = F_true @ L_true.T

        result = panel_factor(E, r=2)
        assert result.factors.shape == (T, 2)
        assert result.loadings.shape == (N, 2)

        # FE reconstruction should be close to E
        np.testing.assert_allclose(result.FE, E, atol=1e-6)

    def test_r0_returns_zeros(self):
        rng = np.random.default_rng(42)
        E = rng.normal(0, 1, (20, 10))
        result = panel_factor(E, r=0)
        assert result.FE.shape == (20, 10)
        np.testing.assert_array_equal(result.FE, 0)


class TestPanelFE:
    """Test nuclear norm thresholding."""

    def test_soft_threshold_reduces_rank(self):
        """Soft thresholding with large lambda should produce low-rank result."""
        rng = np.random.default_rng(42)
        T, N = 30, 20
        E = rng.normal(0, 1, (T, N))
        FE = panel_FE(E, lam=0.1, hard=False)
        # Should have reduced nuclear norm
        _, s_orig, _ = np.linalg.svd(E, full_matrices=False)
        _, s_thresh, _ = np.linalg.svd(FE, full_matrices=False)
        assert np.sum(s_thresh) < np.sum(s_orig)

    def test_hard_threshold(self):
        rng = np.random.default_rng(42)
        T, N = 30, 20
        E = rng.normal(0, 1, (T, N))
        FE = panel_FE(E, lam=0.05, hard=True)
        assert FE.shape == (T, N)

    def test_zero_lambda_identity(self):
        """Lambda=0 should return the original matrix."""
        rng = np.random.default_rng(42)
        E = rng.normal(0, 1, (20, 15))
        FE = panel_FE(E, lam=0.0, hard=False)
        np.testing.assert_allclose(FE, E, atol=1e-6)


class TestDemean:
    """Test demeaning operations."""

    def test_unit_demean(self):
        rng = np.random.default_rng(42)
        Y = rng.normal(0, 1, (20, 10))
        result = demean(Y, force=1)
        # Column means should be zero
        np.testing.assert_allclose(result.YY.mean(axis=0), 0, atol=1e-12)
        assert result.alpha is not None
        assert result.xi is None

    def test_time_demean(self):
        rng = np.random.default_rng(42)
        Y = rng.normal(0, 1, (20, 10))
        result = demean(Y, force=2)
        # Row means should be zero
        np.testing.assert_allclose(result.YY.mean(axis=1), 0, atol=1e-12)
        assert result.xi is not None
        assert result.alpha is None

    def test_twoway_demean(self):
        rng = np.random.default_rng(42)
        Y = rng.normal(0, 1, (20, 10))
        result = demean(Y, force=3)
        # Both row and column means should be close to grand mean
        assert result.alpha is not None
        assert result.xi is not None

    def test_weighted_demean(self):
        rng = np.random.default_rng(42)
        Y = rng.normal(0, 1, (20, 10))
        W = rng.uniform(0.5, 1.5, (20, 10))
        result = demean(Y, force=1, W=W)
        assert result.alpha is not None


class TestFeAdd:
    """Test FE reconstruction."""

    def test_roundtrip_unit(self):
        rng = np.random.default_rng(42)
        Y = rng.normal(0, 1, (20, 10))
        dm = demean(Y, force=1)
        FE = fe_add(dm.mu, dm.alpha, dm.xi, 20, 10, force=1)
        # Y = demean(Y) + FE should hold
        np.testing.assert_allclose(Y, dm.YY + FE, atol=1e-10)


class TestPanelBeta:
    """Test OLS beta estimation."""

    def test_known_beta(self):
        """Should recover known beta."""
        rng = np.random.default_rng(42)
        T, N, p = 50, 30, 2
        X = rng.normal(0, 1, (T, N, p))
        beta_true = np.array([2.0, -1.5])
        FE = rng.normal(0, 0.1, (T, N))
        Y = FE + covar_fit(X, beta_true) + rng.normal(0, 0.01, (T, N))

        xxinv = compute_xxinv(X)
        beta_hat = panel_beta(X, Y, FE, xxinv)
        np.testing.assert_allclose(beta_hat, beta_true, atol=0.1)
