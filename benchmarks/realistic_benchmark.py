"""
Realistic benchmark: point estimates + bootstrap SE at your typical scale.

Scenarios:
  - N=10k, T=24, 500 boots
  - N=10k, T=36, 500 boots
  - N=20k, T=24, 500 boots
  - N=20k, T=36, 1000 boots
"""
import sys, os, time
sys.path.insert(0, ".")

import numpy as np
import polars as pl
import pyfector
from pyfector.backend import set_device

def simulate(N, T, N_treated, r=2, delta=3.0, seed=42):
    rng = np.random.default_rng(seed)
    mu = 1.0
    alpha = rng.normal(0, 1, N)
    xi = rng.normal(0, 0.5, T)
    factors = rng.normal(0, 1, (T, r))
    loadings = rng.normal(0, 1, (N, r))
    eps = rng.normal(0, 0.5, (T, N))

    D = np.zeros((T, N))
    for j in range(N_treated):
        t0 = rng.integers(T // 3, 2 * T // 3 + 1)
        D[t0:, j] = 1.0

    Y = mu + alpha[None, :] + xi[:, None] + factors @ loadings.T + D * delta + eps

    rows = []
    for j in range(N):
        for t in range(T):
            rows.append({"unit": j, "time": t, "Y": float(Y[t, j]), "D": int(D[t, j])})
    return pl.DataFrame(rows), delta

HAS_GPU = False
try:
    import cupy
    cupy.zeros(1)
    HAS_GPU = True
    gpu_name = cupy.cuda.runtime.getDeviceProperties(0)["name"].decode()
except Exception:
    pass

scenarios = [
    (10000,  24, 4000,  500, "N=10k, T=24, 500 boots"),
    (10000,  36, 4000,  500, "N=10k, T=36, 500 boots"),
    (20000,  24, 8000,  500, "N=20k, T=24, 500 boots"),
    (20000,  36, 8000, 1000, "N=20k, T=36, 1000 boots"),
]

devices = ["cpu"]
if HAS_GPU:
    devices.append("gpu")
    print(f"GPU: {gpu_name}")
else:
    print("No GPU — CPU only")

print()
print(f"{'Scenario':35s}", end="")
for d in devices:
    print(f"  {'Point('+d+')':>12s}  {'Boot('+d+')':>12s}  {'Total('+d+')':>12s}", end="")
if len(devices) > 1:
    print(f"  {'Speedup':>8s}", end="")
print()
print("=" * (35 + len(devices) * 42 + (10 if len(devices) > 1 else 0)))

for N, T, N_tr, nboots, label in scenarios:
    print(f"\nGenerating {label}...", flush=True)
    df, delta = simulate(N, T, N_tr, r=2, delta=3.0, seed=42)

    results = {}
    for device in devices:
        print(f"  Running {device}...", end="", flush=True)

        # Point estimate
        t0 = time.perf_counter()
        res_point = pyfector.fect(
            data=df, Y="Y", D="D", index=("unit", "time"),
            method="ife", r=2, CV=False, se=False,
            device=device, seed=42,
        )
        t_point = time.perf_counter() - t0

        # Bootstrap SE
        t0 = time.perf_counter()
        res_boot = pyfector.fect(
            data=df, Y="Y", D="D", index=("unit", "time"),
            method="ife", r=2, CV=False,
            se=True, nboots=nboots, n_jobs=1,
            device=device, seed=42,
        )
        t_boot = time.perf_counter() - t0

        t_total = t_point + t_boot
        results[device] = (t_point, t_boot, t_total, res_boot)
        print(f" done ({t_total:.1f}s)", flush=True)

    # Print row
    print(f"\n  {label:33s}", end="")
    for d in devices:
        tp, tb, tt, _ = results[d]
        print(f"  {tp:>11.1f}s  {tb:>11.1f}s  {tt:>11.1f}s", end="")
    if len(devices) > 1:
        speedup = results["cpu"][2] / results["gpu"][2]
        print(f"  {speedup:>7.1f}x", end="")
    print()

    # Print estimation results (should match across devices)
    for d in devices:
        _, _, _, res = results[d]
        inf = res.inference
        print(f"    [{d}] ATT={res.att_avg:.4f}  SE={inf.att_avg_se:.4f}  "
              f"CI=[{inf.att_avg_ci[0]:.4f}, {inf.att_avg_ci[1]:.4f}]  "
              f"p={inf.att_avg_pval:.4f}  iter={res.niter}")

print("\nDone.")
