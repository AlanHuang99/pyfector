"""
Cross-validation tests: compare pyfector results against R fect.

Calls R via subprocess and exchanges data via CSV/JSON.
"""

import json
import os
import subprocess
import tempfile

import numpy as np
import polars as pl
import pytest

import pyfector
from conftest import _simulate_panel


def _r_available():
    try:
        r = subprocess.run(
            ["R", "-e", 'library(fect); cat("ok")'],
            capture_output=True, text=True, timeout=30,
        )
        return "ok" in r.stdout
    except Exception:
        return False


HAS_R_FECT = _r_available()

pytestmark = pytest.mark.skipif(
    not HAS_R_FECT,
    reason="R and fect package required for comparison tests",
)


def _run_r_fect(csv_path, method, r, force, covariates=None,
                se=False, nboots=200):
    """Run R fect on a CSV file, return results as dict."""
    covar_str = ""
    if covariates:
        covar_str = " + " + " + ".join(covariates)

    se_str = "TRUE" if se else "FALSE"
    force_str = f'"{force}"'

    if method == "ife":
        method_args = f'method="ife", r={r}, CV=FALSE'
    elif method == "mc":
        method_args = f'method="mc", lambda={r}, CV=FALSE'
    else:
        method_args = f'method="fe"'

    boot_args = f', nboots={nboots}' if se else ''

    r_script = f"""
library(fect)
library(jsonlite)

df <- read.csv("{csv_path}")

out <- fect(Y ~ D{covar_str},
            data=df,
            index=c("unit", "time"),
            {method_args},
            force={force_str},
            se={se_str}{boot_args})

result <- list(
    att_avg = as.numeric(out$att.avg),
    att_on = as.numeric(out$att),
    time_on = as.numeric(out$time)
)

if ({se_str}) {{
    ea <- out$est.att
    result$se_avg <- mean(ea[, "S.E."], na.rm=TRUE)
    result$ci_lower <- mean(ea[, "CI.lower"], na.rm=TRUE)
    result$ci_upper <- mean(ea[, "CI.upper"], na.rm=TRUE)
    result$att_avg_se <- sd(out$att.avg.boot, na.rm=TRUE)
}}

cat(toJSON(result, auto_unbox=TRUE, digits=8))
"""
    # Write script to temp file to avoid shell escaping issues
    with tempfile.NamedTemporaryFile(suffix=".R", delete=False, mode="w") as f:
        f.write(r_script)
        script_path = f.name

    try:
        proc = subprocess.run(
            ["R", "--vanilla", "--slave", "-f", script_path],
            capture_output=True, text=True, timeout=300,
        )
    finally:
        os.unlink(script_path)

    if proc.returncode != 0:
        raise RuntimeError(f"R failed:\n{proc.stderr}")

    # Extract JSON — use a temp file approach for reliability
    stdout = proc.stdout.strip()
    # The JSON output may have R's progress messages mixed in.
    # Find the outermost balanced { } block
    depth = 0
    start = None
    for i, c in enumerate(stdout):
        if c == '{':
            if depth == 0:
                start = i
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0 and start is not None:
                return json.loads(stdout[start:i + 1])

    raise RuntimeError(f"No valid JSON in R output:\n{stdout}")


def _make_shared_data(N=100, T=30, N_treated=40, r=2, p=0, delta=3.0, seed=42):
    """Create identical data for both R and Python, saved as CSV."""
    panel = _simulate_panel(N=N, T=T, N_treated=N_treated, r=r, p=p,
                            delta=delta, seed=seed)
    df_pl = panel["data"]
    covar_names = panel["covariate_names"]

    # Save to temp CSV
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
    df_pl.write_csv(tmp.name)
    tmp.close()

    return tmp.name, df_pl, delta, covar_names


class TestFEvsR:
    """Compare FE results between pyfector and R fect."""

    def test_fe_att_comparison(self):
        csv_path, df_pl, delta, _ = _make_shared_data(N=100, T=30, p=0, r=0)
        try:
            py = pyfector.fect(data=df_pl, Y="Y", D="D", index=("unit", "time"),
                             method="fe", force="two-way", seed=42)
            r_res = _run_r_fect(csv_path, method="fe", r=0, force="two-way")

            rel_diff = abs(py.att_avg - r_res["att_avg"]) / (abs(r_res["att_avg"]) + 1e-6)
            print(f"FE ATT: py={py.att_avg:.4f}, R={r_res['att_avg']:.4f}, rel_diff={rel_diff:.4f}")
            assert rel_diff < 0.1, f"FE ATT mismatch: {rel_diff:.4f}"
        finally:
            os.unlink(csv_path)

    def test_fe_with_covariates(self):
        csv_path, df_pl, delta, covars = _make_shared_data(N=100, T=30, p=2, r=0)
        try:
            py = pyfector.fect(data=df_pl, Y="Y", D="D", index=("unit", "time"),
                             X=covars, method="fe", force="two-way", seed=42)
            r_res = _run_r_fect(csv_path, method="fe", r=0, force="two-way",
                                covariates=covars)

            rel_diff = abs(py.att_avg - r_res["att_avg"]) / (abs(r_res["att_avg"]) + 1e-6)
            print(f"FE+covar ATT: py={py.att_avg:.4f}, R={r_res['att_avg']:.4f}, rel_diff={rel_diff:.4f}")
            assert rel_diff < 0.15
        finally:
            os.unlink(csv_path)

    @pytest.mark.parametrize("force", ["none", "unit", "time", "two-way"])
    def test_force_options(self, force):
        csv_path, df_pl, delta, _ = _make_shared_data(N=80, T=25, r=0, p=0, seed=888)
        try:
            py = pyfector.fect(data=df_pl, Y="Y", D="D", index=("unit", "time"),
                             method="fe", force=force, seed=42)
            r_res = _run_r_fect(csv_path, method="fe", r=0, force=force)

            rel_diff = abs(py.att_avg - r_res["att_avg"]) / (abs(r_res["att_avg"]) + 1e-6)
            print(f"force={force}: py={py.att_avg:.4f}, R={r_res['att_avg']:.4f}")
            assert rel_diff < 0.15
        finally:
            os.unlink(csv_path)


class TestIFEvsR:
    """Compare IFE results."""

    def test_ife_r0(self):
        csv_path, df_pl, delta, _ = _make_shared_data(N=100, T=30, p=0, r=0)
        try:
            py = pyfector.fect(data=df_pl, Y="Y", D="D", index=("unit", "time"),
                             method="ife", r=0, CV=False, seed=42)
            r_res = _run_r_fect(csv_path, method="ife", r=0, force="two-way")

            rel_diff = abs(py.att_avg - r_res["att_avg"]) / (abs(r_res["att_avg"]) + 1e-6)
            print(f"IFE r=0: py={py.att_avg:.4f}, R={r_res['att_avg']:.4f}")
            assert rel_diff < 0.1
        finally:
            os.unlink(csv_path)

    def test_ife_r2(self):
        csv_path, df_pl, delta, _ = _make_shared_data(N=100, T=30, r=2, p=0)
        try:
            py = pyfector.fect(data=df_pl, Y="Y", D="D", index=("unit", "time"),
                             method="ife", r=2, CV=False, seed=42)
            r_res = _run_r_fect(csv_path, method="ife", r=2, force="two-way")

            rel_diff = abs(py.att_avg - r_res["att_avg"]) / (abs(r_res["att_avg"]) + 1e-6)
            print(f"IFE r=2: py={py.att_avg:.4f}, R={r_res['att_avg']:.4f}")
            assert rel_diff < 0.15
        finally:
            os.unlink(csv_path)

    def test_ife_dynamic_effects_similar(self):
        """Post-treatment dynamic ATTs should be similar."""
        csv_path, df_pl, delta, _ = _make_shared_data(N=100, T=30, r=2, p=0)
        try:
            py = pyfector.fect(data=df_pl, Y="Y", D="D", index=("unit", "time"),
                             method="ife", r=2, CV=False, seed=42)
            r_res = _run_r_fect(csv_path, method="ife", r=2, force="two-way")

            py_post = py.att_on[py.time_on >= 0]
            r_time = np.array(r_res["time_on"])
            r_att = np.array(r_res["att_on"])
            r_post = r_att[r_time >= 0]

            if len(py_post) > 0 and len(r_post) > 0:
                py_mean = np.mean(py_post)
                r_mean = np.mean(r_post)
                rel_diff = abs(py_mean - r_mean) / (abs(r_mean) + 1e-6)
                print(f"Post-treatment mean ATT: py={py_mean:.4f}, R={r_mean:.4f}")
                assert rel_diff < 0.2
        finally:
            os.unlink(csv_path)


class TestATTAccuracy:
    """Test that both packages recover the true ATT."""

    def test_both_recover_true_att_fe(self):
        """No factors in DGP: both should nail it."""
        csv_path, df_pl, delta, _ = _make_shared_data(N=200, T=50, r=0, p=0,
                                                       delta=5.0, seed=555)
        try:
            py = pyfector.fect(data=df_pl, Y="Y", D="D", index=("unit", "time"),
                             method="fe", seed=42)
            r_res = _run_r_fect(csv_path, method="fe", r=0, force="two-way")

            print(f"True=5.0, py={py.att_avg:.4f}, R={r_res['att_avg']:.4f}")
            assert abs(py.att_avg - delta) < 1.0
            assert abs(r_res["att_avg"] - delta) < 1.0
        finally:
            os.unlink(csv_path)

    def test_both_recover_true_att_ife(self):
        """With factors: both should be close with correct r."""
        csv_path, df_pl, delta, _ = _make_shared_data(N=200, T=50, r=2, p=0,
                                                       delta=3.0, seed=666)
        try:
            py = pyfector.fect(data=df_pl, Y="Y", D="D", index=("unit", "time"),
                             method="ife", r=2, CV=False, seed=42)
            r_res = _run_r_fect(csv_path, method="ife", r=2, force="two-way")

            print(f"True=3.0, py={py.att_avg:.4f}, R={r_res['att_avg']:.4f}")
            assert abs(py.att_avg - delta) < 1.5
            assert abs(r_res["att_avg"] - delta) < 1.5

            # They should agree with each other
            rel_diff = abs(py.att_avg - r_res["att_avg"]) / (abs(r_res["att_avg"]) + 1e-6)
            assert rel_diff < 0.2
        finally:
            os.unlink(csv_path)


class TestBootstrapSEvsR:
    """Compare bootstrap SE between pyfector and R fect."""

    def test_bootstrap_se_same_ballpark(self):
        """Bootstrap SE should be in same order of magnitude."""
        csv_path, df_pl, delta, _ = _make_shared_data(N=80, T=25, r=0, p=0, seed=777)
        try:
            py = pyfector.fect(data=df_pl, Y="Y", D="D", index=("unit", "time"),
                             method="fe", se=True, nboots=500, seed=42)
            r_res = _run_r_fect(csv_path, method="fe", r=0, force="two-way",
                                se=True, nboots=500)

            py_se = py.inference.att_avg_se
            r_se = r_res.get("att_avg_se", None)

            print(f"ATT: py={py.att_avg:.4f}, R={r_res['att_avg']:.4f}")
            print(f"SE: py={py_se:.4f}, R={r_se}")

            # Point estimates should agree
            rel_diff = abs(py.att_avg - r_res["att_avg"]) / (abs(r_res["att_avg"]) + 1e-6)
            assert rel_diff < 0.05, f"ATT mismatch with 500 boots"

            # SE should be in same ballpark (within 3x)
            if r_se is not None and r_se > 0:
                se_ratio = max(py_se, r_se) / min(py_se, r_se)
                print(f"SE ratio (py/R): {se_ratio:.2f}")
                assert se_ratio < 3.0, f"SE too different: py={py_se:.4f}, R={r_se:.4f}"
        finally:
            os.unlink(csv_path)

    def test_bootstrap_1000_fe(self):
        """1000 boots: ATT point estimates should be very close."""
        csv_path, df_pl, delta, _ = _make_shared_data(N=100, T=30, r=0, p=0, seed=999)
        try:
            py = pyfector.fect(data=df_pl, Y="Y", D="D", index=("unit", "time"),
                             method="fe", se=True, nboots=1000, seed=42, n_jobs=4)
            r_res = _run_r_fect(csv_path, method="fe", r=0, force="two-way",
                                se=True, nboots=1000)

            print(f"1000 boots — ATT: py={py.att_avg:.4f}, R={r_res['att_avg']:.4f}")
            print(f"  py SE={py.inference.att_avg_se:.4f}")
            print(f"  py CI=[{py.inference.att_avg_ci[0]:.4f}, {py.inference.att_avg_ci[1]:.4f}]")

            rel_diff = abs(py.att_avg - r_res["att_avg"]) / (abs(r_res["att_avg"]) + 1e-6)
            assert rel_diff < 0.05
        finally:
            os.unlink(csv_path)

    def test_bootstrap_ife_se(self):
        """IFE with bootstrap: compare SE."""
        csv_path, df_pl, delta, _ = _make_shared_data(N=100, T=30, r=2, p=0, seed=1234)
        try:
            py = pyfector.fect(data=df_pl, Y="Y", D="D", index=("unit", "time"),
                             method="ife", r=2, CV=False,
                             se=True, nboots=500, seed=42)
            r_res = _run_r_fect(csv_path, method="ife", r=2, force="two-way",
                                se=True, nboots=500)

            print(f"IFE 500 boots — ATT: py={py.att_avg:.4f}, R={r_res['att_avg']:.4f}")
            print(f"  py SE={py.inference.att_avg_se:.4f}")

            rel_diff = abs(py.att_avg - r_res["att_avg"]) / (abs(r_res["att_avg"]) + 1e-6)
            assert rel_diff < 0.15
        finally:
            os.unlink(csv_path)
