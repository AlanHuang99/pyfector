"""Regression coverage for empty fits, placebo eligibility and clustered inference."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

import pyfector
from pyfector.diagnostics import _run_placebo_test
from pyfector.inference import bootstrap, jackknife


def _panel(early=2):
    rng = np.random.default_rng(20261012)
    T, N = 8, 24
    starts = np.repeat([early, 5, T], [6, 6, 12])
    D = (np.arange(T)[:, None] >= starts).astype(float)
    Y = 4 + rng.normal(0, 0.5, N) + 0.2 * np.arange(T)[:, None]
    Y += rng.normal(0, 0.05, (T, N))
    return pl.DataFrame({
        "unit": np.tile(np.arange(N), T),
        "time": np.repeat(np.arange(T), N),
        "Y": Y.ravel(), "D": D.ravel(),
        "office": np.tile(np.arange(N) % 6, T),
    })


def _fit(data, **kwargs):
    options = dict(method="fe", CV=False, nboots=20, n_jobs=1, seed=17)
    options.update(kwargs)
    return pyfector.fect(data, Y="Y", D="D", index=("unit", "time"), **options)


@pytest.mark.parametrize("method,extra", [("fe", {}), ("ife", {"r": 1}), ("mc", {"lam": 1.0})])
def test_no_treated_units_after_filter_raises(method, extra):
    with pytest.raises(ValueError, match="treated"):
        _fit(_panel(), method=method, min_T0=8, **extra)


def test_controls_only_fit_raises():
    with pytest.raises(ValueError, match="treated"):
        _fit(_panel().with_columns(pl.lit(0.0).alias("D")))


def test_explicit_unavailable_diagnostic_raises():
    result = SimpleNamespace(inference=SimpleNamespace(), time_on=np.array([0.0]), att_on=np.array([1.0]))
    with pytest.raises(ValueError, match="pretrend_f"):
        pyfector.run_diagnostics(result, _requested=["pretrend_f"])


def test_best_effort_unavailable_diagnostic_warns():
    result = SimpleNamespace(inference=None)
    with pytest.warns(RuntimeWarning, match="unavailable"):
        diag = pyfector.run_diagnostics(result)
    assert not diag.tests


def test_full_diagnostics_warns_when_automatic_tost_bound_is_unavailable():
    data = _panel().with_columns((2.0 * pl.col("D")).alias("Y"))
    with pytest.warns(RuntimeWarning, match="tost.*sigma2_fect"):
        fit = _fit(data, force="none", se=True, diagnostics="full")
    assert fit.att_avg == pytest.approx(2.0)
    assert fit.diagnostics.pretrend_f is not None
    assert fit.diagnostics.tost is None


def test_strict_tost_rejects_unusable_standard_errors():
    result = SimpleNamespace(
        inference=SimpleNamespace(att_on_se=np.array([np.nan, 0.1]), att_on_boot=None),
        time_on=np.array([-1.0, 0.0]), att_on=np.array([0.0, 1.0]), sigma2_fect=1.0,
    )
    with pytest.raises(ValueError, match="tost"):
        pyfector.run_diagnostics(result, _requested=["tost"])


def test_f_test_calibration_uses_pre_period_unit_count_not_bootstrap_count():
    from scipy import stats
    rng = np.random.default_rng(27)
    base = rng.normal(size=(2, 40)) * 0.1
    covariance = np.cov(base)
    beta = np.array([0.15, 0.1])
    N, k, threshold = 60, 2, 0.5
    f_stat = float(beta @ np.linalg.inv(covariance) @ beta) * (N - k) / ((N - 1) * k)
    expected_equiv = stats.ncf.cdf(f_stat, k, N - k, N * threshold)
    expected_pretrend = stats.f.sf(f_stat, k, N - k)
    pvals = []
    for copies in (1, 5, 20):
        nboot = base.shape[1] * copies
        pre_boot = np.tile(base, (1, copies)) * np.sqrt((nboot - 1) / (copies * (base.shape[1] - 1)))
        result = SimpleNamespace(
            inference=SimpleNamespace(att_on_boot=np.vstack([pre_boot, np.zeros(nboot)])),
            time_on=np.array([-2.0, -1.0, 0.0]), att_on=np.r_[beta, 1.0],
            count_on=np.array([45, N, 100]),
        )
        diag = pyfector.run_diagnostics(result, _requested=["pretrend_f", "equiv_f"])
        assert diag.pretrend_f.f_stat == pytest.approx(f_stat)
        assert diag.pretrend_f.df2 == N - k
        assert diag.pretrend_f.p_value == pytest.approx(expected_pretrend)
        assert diag.equiv_f.p_value == pytest.approx(expected_equiv)
        assert diag.equiv_f.n_treated == N
        assert diag.equiv_f.ncp == N * threshold
        pvals.append(diag.equiv_f.p_value)
    assert pvals == pytest.approx([pvals[0]] * 3)


def test_f_test_rejects_insufficient_treated_units():
    result = SimpleNamespace(
        inference=SimpleNamespace(att_on_boot=np.array([[0, 1, 0, -1], [1, 0, -1, 0]], dtype=float)),
        time_on=np.array([-2.0, -1.0]), att_on=np.array([0.1, 0.1]), count_on=np.array([2, 2]),
    )
    with pytest.raises(ValueError, match="treated units"):
        pyfector.run_diagnostics(result, _requested=["pretrend_f", "equiv_f"])


def test_placebo_excludes_unidentified_units_and_is_level_invariant():
    data = _panel()
    shifted = data.with_columns((pl.col("Y") + pl.when(pl.col("unit") < 6).then(10).otherwise(0)).alias("Y"))
    results = []
    for sample in [data, shifted]:
        fit = _fit(sample)
        placebo = _run_placebo_test(fit, placebo_period=(-2, -1), threshold_abs=0.36)
        assert placebo.n_dropped_units == 6
        assert placebo.n_obs == 12
        assert fit.panel.N == 24
        results.append(placebo)
    assert results[0].estimate == pytest.approx(results[1].estimate, abs=1e-10)
    assert results[0].se == pytest.approx(results[1].se, abs=1e-10)
    assert abs(results[0].estimate) < 0.1


def test_placebo_reapplies_configured_minimum():
    fit = _fit(_panel(early=3), min_T0=3)
    placebo = _run_placebo_test(fit, placebo_period=(-2, -1), threshold_abs=0.36)
    assert placebo.n_dropped_units == 6
    assert placebo.n_obs == 12


def test_placebo_all_ineligible_raises():
    fit = _fit(_panel().filter((pl.col("unit") < 6) | (pl.col("unit") >= 12)))
    with pytest.raises(ValueError, match="treated.*placebo|placebo.*treated"):
        _run_placebo_test(fit, placebo_period=(-2, -1), threshold_abs=0.36)


def test_full_diagnostics_warns_when_placebo_is_unavailable():
    data = _panel().filter((pl.col("unit") < 6) | (pl.col("unit") >= 12))
    with pytest.warns(RuntimeWarning, match="placebo.*eligible"):
        fit = _fit(data, se=True, diagnostics="full", diagnostics_options={"placebo_period": (-2, -1)})
    assert fit.diagnostics.placebo is None
    assert fit.diagnostics.tost is not None


def test_strict_diagnostics_raises_when_placebo_is_unavailable():
    data = _panel().filter((pl.col("unit") < 6) | (pl.col("unit") >= 12))
    with pytest.raises(ValueError, match="eligible"):
        _fit(data, se=True, diagnostics=["placebo"], diagnostics_options={"placebo_period": (-2, -1)})


def test_placebo_parallel_backend_and_reproducibility(monkeypatch):
    import joblib
    from joblib._parallel_backends import LokyBackend

    class CheckedParallel(joblib.Parallel):
        def __call__(self, iterable):
            assert isinstance(self._backend, LokyBackend)
            return super().__call__(iterable)

    fit = _fit(_panel())
    serial = _run_placebo_test(fit, placebo_period=(-2, -1), threshold_abs=0.36)
    monkeypatch.setattr(joblib, "Parallel", CheckedParallel)
    parallel = _run_placebo_test(replace(fit, fit_options={**fit.fit_options, "n_jobs": 2}), placebo_period=(-2, -1), threshold_abs=0.36)
    assert parallel.estimate == pytest.approx(serial.estimate)
    assert parallel.se == pytest.approx(serial.se)


def test_cluster_column_is_aligned_after_filtering():
    fit = _fit(_panel(), min_T0=3, cluster="office", se=True)
    np.testing.assert_array_equal(fit.panel.cluster, fit.panel.unit_ids % 6)
    assert fit.inference.n_boot == 20
    assert fit.fit_options["cluster"] == "office"
    placebo = fit.diagnose(placebo_period=(-2, -1), _requested=["placebo"]).placebo
    assert placebo.n_boot == 20


@pytest.mark.parametrize("change,match", [
    (pl.lit(None).alias("office"), "missing"),
    (pl.when(pl.col("time") == 0).then(99).otherwise(pl.col("office")).alias("office"), "constant"),
    (pl.lit(1).alias("office"), "two clusters"),
])
def test_invalid_cluster_column_raises(change, match):
    with pytest.raises(ValueError, match=match):
        _fit(_panel().with_columns(change), cluster="office", se=True)


def test_cluster_with_jackknife_raises():
    with pytest.raises(ValueError, match="bootstrap"):
        _fit(_panel(), cluster="office", se=True, vartype="jackknife")


def test_cluster_reaches_main_and_eligible_placebo_sampler(monkeypatch):
    import pyfector.inference as inference
    original = inference._make_bootstrap_sampler
    seen = []

    def record(unit_type, cluster=None):
        seen.append(np.array(cluster))
        return original(unit_type, cluster)

    monkeypatch.setattr(inference, "_make_bootstrap_sampler", record)
    fit = _fit(_panel(), cluster="office", se=True)
    fit.diagnose(placebo_period=(-2, -1), _requested=["placebo"])
    np.testing.assert_array_equal(seen[0], np.arange(24) % 6)
    np.testing.assert_array_equal(seen[1], np.arange(6, 24) % 6)


def test_unit_column_can_define_clusters():
    fit = _fit(_panel(), cluster="unit", se=True)
    np.testing.assert_array_equal(fit.panel.cluster, fit.panel.unit_ids)


def test_existing_positional_inference_arguments_are_preserved():
    fit = pyfector.fect(
        _panel(), "Y", "D", ("unit", "time"),
        None, None, None, "fe", "two-way", 0, None, 10, None,
        False, 10, 0.1, 3, True, 0, "mspe", "min", True, "bootstrap", 8,
        n_jobs=1, seed=17,
    )
    assert fit.inference.n_boot == 8


def _toy_inference(estimate, **kwargs):
    D = np.array([[0, 0, 0, 0], [1, 1, 0, 0]], dtype=float)
    I = np.ones_like(D)
    eff = np.array([[0, 0, 0, 0], [2, 4, 0, 0]], dtype=float)
    T_on = np.array([[-1, -1, np.nan, np.nan], [0, 0, np.nan, np.nan]])
    return bootstrap(estimate(eff, D, T_on, I), eff, D, I, T_on, np.array([2, 2, 1, 1]), nboots=20, seed=17, point_estimate=(eff, D, T_on, I), **kwargs)


def test_cluster_resampling_keeps_whole_clusters():
    seen = []

    def factory(eff, D, T_on, I):
        def estimate(idx):
            seen.append(idx)
            return eff[:, idx], D[:, idx], T_on[:, idx], I[:, idx]
        return estimate

    result = _toy_inference(factory, cluster=np.array([0, 1, 0, 1]))
    for idx in seen:
        counts = np.bincount(idx, minlength=4)
        assert counts[0] == counts[2]
        assert counts[1] == counts[3]
    assert result.n_boot == 20


@pytest.mark.parametrize("nonfinite", [False, True])
def test_failed_bootstrap_replicates_warn_and_are_counted(nonfinite):
    def factory(eff, D, T_on, I):
        def estimate(idx):
            if idx[0] == 0:
                if not nonfinite:
                    raise ValueError("injected failure")
                return eff[:, idx] * np.nan, D[:, idx], T_on[:, idx], I[:, idx]
            return eff[:, idx], D[:, idx], T_on[:, idx], I[:, idx]
        return estimate

    with pytest.warns(RuntimeWarning, match="bootstrap replications failed"):
        result = _toy_inference(factory)
    assert 0 < result.n_boot_failed < 20
    assert result.n_boot + result.n_boot_failed == 20
    assert np.isfinite(result.att_avg_se)


def test_single_valid_bootstrap_replicate_raises():
    def factory(eff, D, T_on, I):
        calls = 0
        def estimate(idx):
            nonlocal calls
            calls += 1
            if calls > 1:
                raise ValueError("injected failure")
            return eff[:, idx], D[:, idx], T_on[:, idx], I[:, idx]
        return estimate

    with pytest.raises(RuntimeError, match="Too few"):
        _toy_inference(factory)


def test_numerically_invalid_pre_period_draw_is_counted_as_failed():
    def factory(eff, D, T_on, I):
        def estimate(idx):
            sampled = eff[:, idx].copy()
            if idx[0] == 0:
                sampled[0, 0] = np.nan
            return sampled, D[:, idx], T_on[:, idx], I[:, idx]
        return estimate

    with pytest.warns(RuntimeWarning, match="bootstrap replications failed"):
        result = _toy_inference(factory)
    assert 0 < result.n_boot_failed < 20
    assert result.n_boot + result.n_boot_failed == 20
    assert np.isfinite(result.att_on_boot).all()


def test_jackknife_rejects_incomplete_deletions_with_one_treated_unit():
    D = np.array([[0, 0, 0, 0], [1, 0, 0, 0]], dtype=float)
    I = np.ones_like(D)
    eff = D * 2.0
    T_on = np.array([[-1, np.nan, np.nan, np.nan], [0, np.nan, np.nan, np.nan]])

    def estimate(idx):
        return eff[:, idx], D[:, idx], T_on[:, idx], I[:, idx]

    with pytest.raises(RuntimeError, match="jackknife.*failed|failed.*jackknife"):
        jackknife(estimate, eff, D, I, T_on, np.array([2, 1, 1, 1]))


def test_failed_placebo_replicates_warn_and_are_counted(monkeypatch):
    import pyfector.diagnostics as diagnostics
    fit = _fit(_panel())
    original = diagnostics._estimate_with_selected_config
    calls = 0

    def sometimes_fail(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls > 1 and calls % 2 == 0:
            raise ValueError("injected failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(diagnostics, "_estimate_with_selected_config", sometimes_fail)
    with pytest.warns(RuntimeWarning, match="placebo bootstrap replications failed"):
        placebo = _run_placebo_test(fit, placebo_period=(-2, -1), threshold_abs=0.36)
    assert placebo.n_boot == placebo.n_boot_failed == 10
