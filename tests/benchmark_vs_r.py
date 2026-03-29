"""
Detailed head-to-head comparison: pyfector vs R fect.

Runs identical data through both packages and prints a comparison table
with exact numbers for ATT, SE, CI, dynamic effects, and timing.

Usage:
    python tests/benchmark_vs_r.py
"""

import json
import os
import subprocess
import tempfile
import time

import numpy as np
import polars as pl

import pyfector
from tests.conftest import _simulate_panel


def run_r_fect(csv_path, method, r, force="two-way", covariates=None,
               se=False, nboots=200):
    covar_str = ""
    if covariates:
        covar_str = " + " + " + ".join(covariates)
    se_str = "TRUE" if se else "FALSE"

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
t0 <- proc.time()

out <- fect(Y ~ D{covar_str},
            data=df,
            index=c("unit", "time"),
            {method_args},
            force="{force}",
            se={se_str}{boot_args})

elapsed <- (proc.time() - t0)["elapsed"]

result <- list(
    att_avg = as.numeric(out$att.avg),
    att_on = as.numeric(out$att),
    time_on = as.numeric(out$time),
    count_on = as.numeric(out$count),
    niter = as.numeric(out$niter),
    elapsed = as.numeric(elapsed)
)

if (!is.null(out$beta)) {{
    result$beta <- as.numeric(out$beta)
}}

if ({se_str}) {{
    ea <- out$est.att
    result$se_per_period <- as.numeric(ea[, "S.E."])
    result$ci_lower_per_period <- as.numeric(ea[, "CI.lower"])
    result$ci_upper_per_period <- as.numeric(ea[, "CI.upper"])
    result$pval_per_period <- as.numeric(ea[, "p.value"])

    ea_avg <- out$est.avg
    result$att_avg_se <- as.numeric(ea_avg[, "S.E."])
    result$att_avg_ci_lower <- as.numeric(ea_avg[, "CI.lower"])
    result$att_avg_ci_upper <- as.numeric(ea_avg[, "CI.upper"])
    result$att_avg_pval <- as.numeric(ea_avg[, "p.value"])
}}

cat(toJSON(result, auto_unbox=TRUE, digits=10))
"""
    with tempfile.NamedTemporaryFile(suffix=".R", delete=False, mode="w") as f:
        f.write(r_script)
        script_path = f.name

    try:
        proc = subprocess.run(
            ["R", "--vanilla", "--slave", "-f", script_path],
            capture_output=True, text=True, timeout=600,
        )
    finally:
        os.unlink(script_path)

    if proc.returncode != 0:
        raise RuntimeError(f"R failed:\n{proc.stderr}")

    stdout = proc.stdout.strip()
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


def compare(title, dgp, method, r=0, lam=None, covars=None, se=False, nboots=200):
    """Run one comparison and print results."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")

    # Save data
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
    dgp["data"].write_csv(tmp.name)
    tmp.close()

    try:
        # --- Python ---
        py_kw = dict(
            data=dgp["data"], Y="Y", D="D", index=("unit", "time"),
            method=method, force="two-way", seed=42,
        )
        if method == "ife":
            py_kw["r"] = r
            py_kw["CV"] = False
        elif method == "mc":
            py_kw["lam"] = lam
            py_kw["CV"] = False
        if covars:
            py_kw["X"] = covars
        if se:
            py_kw["se"] = True
            py_kw["nboots"] = nboots
            py_kw["n_jobs"] = 4

        t0 = time.time()
        py_res = pyfector.fect(**py_kw)
        py_time = time.time() - t0

        # --- R ---
        r_r = r if method != "mc" else lam
        r_res = run_r_fect(
            tmp.name, method=method, r=r_r, force="two-way",
            covariates=covars, se=se, nboots=nboots,
        )
        r_time = r_res["elapsed"]

        # --- Print comparison ---
        true_att = dgp["true_att"]
        print(f"\n  DGP: N={dgp['N']}, T={dgp['T']}, r={dgp['r']}, "
              f"delta={dgp['delta']}, p={dgp['p']}")
        print(f"  Method: {method}" + (f", r={r}" if method == "ife" else "") +
              (f", lambda={lam}" if method == "mc" else ""))

        print(f"\n  {'':30s} {'pyfector':>14s} {'R fect':>14s} {'Difference':>14s}")
        print(f"  {'-'*72}")

        # ATT
        diff = py_res.att_avg - r_res["att_avg"]
        print(f"  {'ATT (average)':30s} {py_res.att_avg:>14.6f} {r_res['att_avg']:>14.6f} {diff:>+14.6f}")
        print(f"  {'True ATT':30s} {true_att:>14.6f}")
        print(f"  {'|py - true|':30s} {abs(py_res.att_avg - true_att):>14.6f}")
        print(f"  {'|R  - true|':30s} {abs(r_res['att_avg'] - true_att):>14.6f}")

        # Beta
        if covars and py_res.beta is not None and "beta" in r_res:
            print(f"\n  Coefficients:")
            for i, name in enumerate(covars):
                py_b = py_res.beta[i] if i < len(py_res.beta) else float('nan')
                r_b = r_res["beta"][i] if i < len(r_res["beta"]) else float('nan')
                print(f"    {name:28s} {py_b:>14.6f} {r_b:>14.6f} {py_b - r_b:>+14.6f}")

        # Dynamic ATT (first 5 post-treatment periods)
        r_time_on = np.array(r_res["time_on"])
        r_att_on = np.array(r_res["att_on"])
        print(f"\n  Dynamic ATT (post-treatment):")
        print(f"  {'Period':>8s} {'pyfector':>12s} {'R fect':>12s} {'Diff':>12s} {'Count':>8s}")
        for t_val in range(min(8, len(py_res.time_on))):
            py_mask = py_res.time_on == t_val
            r_mask = r_time_on == t_val
            if np.any(py_mask) and np.any(r_mask):
                py_att = py_res.att_on[py_mask][0]
                r_att = r_att_on[r_mask][0]
                cnt = py_res.count_on[py_mask][0] if py_res.count_on is not None else ""
                print(f"  {t_val:>8d} {py_att:>12.6f} {r_att:>12.6f} {py_att - r_att:>+12.6f} {cnt:>8}")

        # SE comparison
        if se:
            print(f"\n  Inference ({nboots} bootstrap replications):")
            print(f"  {'':30s} {'pyfector':>14s} {'R fect':>14s} {'Difference':>14s}")
            print(f"  {'-'*72}")

            r_se = r_res.get("att_avg_se", float('nan'))
            r_ci_lo = r_res.get("att_avg_ci_lower", float('nan'))
            r_ci_hi = r_res.get("att_avg_ci_upper", float('nan'))
            r_pval = r_res.get("att_avg_pval", float('nan'))

            py_se = py_res.inference.att_avg_se
            py_ci = py_res.inference.att_avg_ci
            py_pval = py_res.inference.att_avg_pval

            print(f"  {'SE':30s} {py_se:>14.6f} {r_se:>14.6f} {py_se - r_se:>+14.6f}")
            print(f"  {'CI lower':30s} {py_ci[0]:>14.6f} {r_ci_lo:>14.6f} {py_ci[0] - r_ci_lo:>+14.6f}")
            print(f"  {'CI upper':30s} {py_ci[1]:>14.6f} {r_ci_hi:>14.6f} {py_ci[1] - r_ci_hi:>+14.6f}")
            print(f"  {'p-value':30s} {py_pval:>14.6f} {r_pval:>14.6f}")

            # Per-period SE comparison
            r_se_pp = r_res.get("se_per_period", [])
            if len(r_se_pp) > 0 and py_res.inference.att_on_se is not None:
                print(f"\n  Per-period SE (post-treatment):")
                print(f"  {'Period':>8s} {'py SE':>12s} {'R SE':>12s} {'Ratio':>10s}")
                r_time_arr = np.array(r_res["time_on"])
                for t_val in range(min(8, len(py_res.time_on))):
                    py_mask = py_res.time_on == t_val
                    r_mask = r_time_arr == t_val
                    if np.any(py_mask) and np.any(r_mask):
                        py_se_t = py_res.inference.att_on_se[py_mask][0]
                        r_idx = np.where(r_mask)[0][0]
                        r_se_t = r_se_pp[r_idx] if r_idx < len(r_se_pp) else float('nan')
                        if not np.isnan(py_se_t) and not np.isnan(r_se_t) and r_se_t > 0:
                            ratio = py_se_t / r_se_t
                            print(f"  {t_val:>8d} {py_se_t:>12.6f} {r_se_t:>12.6f} {ratio:>10.3f}")

        # Timing
        print(f"\n  {'Timing (seconds)':30s} {py_time:>14.2f} {r_time:>14.2f} "
              f"{'':>4s}({py_time/r_time:.1f}x)" if r_time > 0 else "")
        print(f"  {'Iterations':30s} {py_res.niter:>14d} {int(r_res.get('niter', 0)):>14d}")

    finally:
        os.unlink(tmp.name)

    return py_res, r_res


if __name__ == "__main__":
    print("=" * 80)
    print("  pyfector vs R fect — Head-to-Head Comparison")
    print("=" * 80)

    # 1. FE, no factors, no covariates
    dgp1 = _simulate_panel(N=200, T=50, N_treated=80, r=0, p=0, delta=5.0, seed=42)
    compare("FE on clean TWFE data (N=200, T=50, delta=5.0)", dgp1, method="fe")

    # 2. FE with covariates
    dgp2 = _simulate_panel(N=200, T=50, N_treated=80, r=0, p=3, delta=5.0, seed=42)
    compare("FE with 3 covariates", dgp2, method="fe",
            covars=dgp2["covariate_names"])

    # 3. IFE r=2 (true model)
    dgp3 = _simulate_panel(N=200, T=50, N_treated=80, r=2, p=0, delta=3.0, seed=123)
    compare("IFE r=2 on factor DGP (N=200, T=50, delta=3.0)", dgp3,
            method="ife", r=2)

    # 4. IFE r=2 with covariates
    dgp4 = _simulate_panel(N=200, T=50, N_treated=80, r=2, p=2, delta=3.0, seed=123)
    compare("IFE r=2 with 2 covariates", dgp4, method="ife", r=2,
            covars=dgp4["covariate_names"])

    # 5. MC
    dgp5 = _simulate_panel(N=200, T=50, N_treated=80, r=2, p=0, delta=3.0, seed=456)
    compare("MC lambda=0.01", dgp5, method="mc", lam=0.01)

    # 6. FE with bootstrap SE (500 boots)
    dgp6 = _simulate_panel(N=200, T=50, N_treated=80, r=0, p=0, delta=5.0, seed=789)
    compare("FE with 500 bootstrap SE", dgp6, method="fe", se=True, nboots=500)

    # 7. IFE with bootstrap SE (500 boots)
    dgp7 = _simulate_panel(N=200, T=50, N_treated=80, r=2, p=0, delta=3.0, seed=321)
    compare("IFE r=2 with 500 bootstrap SE", dgp7, method="ife", r=2,
            se=True, nboots=500)

    # 8. Large panel (N=1000)
    dgp8 = _simulate_panel(N=1000, T=50, N_treated=400, r=2, p=0, delta=3.0, seed=999)
    compare("IFE r=2 large panel (N=1000, T=50)", dgp8, method="ife", r=2)

    print("\n" + "=" * 80)
    print("  Done.")
    print("=" * 80)
