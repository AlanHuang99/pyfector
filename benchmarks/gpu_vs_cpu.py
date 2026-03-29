"""
GPU vs CPU benchmark for pyfector.

Tests different panel sizes and measures wall-clock time for:
1. CPU (NumPy)
2. GPU (CuPy) — if available

Usage:
    python benchmarks/gpu_vs_cpu.py

Run on a machine with NVIDIA GPU and CuPy installed for GPU numbers.
Without GPU, only CPU benchmarks are reported.
"""

from __future__ import annotations

import sys
import time

import numpy as np

# Add project root to path
sys.path.insert(0, ".")


def generate_panel(N, T, N_treated, r=2, p=0, delta=3.0, seed=42):
    """Generate panel data directly as matrices (skip Polars for speed)."""
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

    I = np.ones((T, N))
    II = I * (1 - D)

    X = rng.normal(0, 1, (T, N, p)) if p > 0 else None

    return Y, D, I, II, X


def benchmark_single(Y, II, X, r, force, device, tol=1e-5, max_iter=500):
    """Time a single estimation."""
    from pyfector.backend import set_device, to_device
    from pyfector.panel import initial_fit
    from pyfector.estimators import estimate_ife

    set_device(device)
    Y_d = to_device(Y)
    II_d = to_device(II)
    X_d = to_device(X) if X is not None else None

    # Initial fit
    Y0, beta0 = initial_fit(Y_d, X_d, II_d, force)

    # Warm-up (for GPU — first call compiles kernels)
    if device == "gpu":
        import cupy
        estimate_ife(Y_d, Y0, X_d, II_d, None, beta0, r=r, force=force,
                     tol=0.1, max_iter=3)
        cupy.cuda.Device().synchronize()

    # Timed run
    if device == "gpu":
        import cupy
        cupy.cuda.Device().synchronize()

    t0 = time.perf_counter()
    result = estimate_ife(Y_d, Y0, X_d, II_d, None, beta0, r=r, force=force,
                          tol=tol, max_iter=max_iter)

    if device == "gpu":
        import cupy
        cupy.cuda.Device().synchronize()

    elapsed = time.perf_counter() - t0
    return elapsed, result.niter, result.converged


def benchmark_bootstrap(Y, D, I, II, T_on, X, unit_type, r, force, device,
                        nboots=50):
    """Time bootstrap inference."""
    from pyfector.backend import set_device, to_device, to_numpy
    from pyfector.panel import initial_fit
    from pyfector.estimators import estimate_ife
    from pyfector.inference import bootstrap

    set_device(device)
    Y_d = to_device(Y)
    II_d = to_device(II)
    X_d = to_device(X) if X is not None else None
    Y0, beta0 = initial_fit(Y_d, X_d, II_d, force)

    def _estimate_fn(unit_idx):
        Y_sub = Y_d[:, unit_idx]
        II_sub = II_d[:, unit_idx]
        X_sub = X_d[:, unit_idx, :] if X_d is not None else None
        T_on_sub = T_on[:, unit_idx]
        Y0_sub, beta0_sub = initial_fit(Y_sub, X_sub, II_sub, force)
        est = estimate_ife(Y_sub, Y0_sub, X_sub, II_sub, None, beta0_sub,
                           r=r, force=force, tol=1e-4, max_iter=300)
        eff = to_numpy(Y_sub - est.fit)
        return eff, to_numpy(to_device(D)[:, unit_idx]), T_on_sub

    t0 = time.perf_counter()
    result = bootstrap(
        _estimate_fn, Y, D, I, T_on, unit_type,
        nboots=nboots, seed=42,
    )
    elapsed = time.perf_counter() - t0
    return elapsed, result.att_avg_se


def run_benchmarks():
    """Run all benchmarks."""
    # Check GPU
    has_gpu = False
    try:
        import cupy
        cupy.zeros(1)
        has_gpu = True
        gpu_name = cupy.cuda.runtime.getDeviceProperties(0)["name"].decode()
        print(f"GPU detected: {gpu_name}")
    except Exception:
        print("No GPU detected — running CPU-only benchmarks")

    devices = ["cpu"]
    if has_gpu:
        devices.append("gpu")

    # Panel configurations
    configs = [
        # (N, T, N_treated, r, label)
        (100,    30,   40,  2, "Small (100x30)"),
        (1000,   50,  400,  2, "Medium (1kx50)"),
        (5000,   50, 2000,  2, "Large (5kx50)"),
        (10000,  20, 4000,  2, "Wide (10kx20)"),
        (50000,  10, 20000, 2, "Very wide (50kx10)"),
        (100000, 10, 40000, 2, "Huge (100kx10)"),
        (100000, 20, 40000, 2, "Huge (100kx20)"),
    ]

    print("\n" + "=" * 90)
    print(f"  {'Config':25s}", end="")
    for d in devices:
        print(f"  {'Time ('+d+')':>12s}  {'Iter':>5s}", end="")
    if len(devices) > 1:
        print(f"  {'Speedup':>8s}", end="")
    print()
    print("=" * 90)

    for N, T, N_tr, r, label in configs:
        try:
            Y, D, I, II, X = generate_panel(N, T, N_tr, r=r, seed=42)
        except MemoryError:
            print(f"  {label:25s}  [OOM generating data]")
            continue

        times = {}
        for device in devices:
            try:
                elapsed, niter, conv = benchmark_single(
                    Y, II, X, r=r, force=3, device=device,
                    tol=1e-5, max_iter=500,
                )
                times[device] = (elapsed, niter)
            except Exception as e:
                times[device] = (None, None)
                print(f"  {label:25s}  {device}: FAILED ({e})")

        # Print row
        print(f"  {label:25s}", end="")
        for d in devices:
            t, n = times.get(d, (None, None))
            if t is not None:
                print(f"  {t:>10.3f}s  {n:>5d}", end="")
            else:
                print(f"  {'FAIL':>12s}  {'':>5s}", end="")
        if len(devices) > 1 and times.get("cpu", (None,))[0] and times.get("gpu", (None,))[0]:
            speedup = times["cpu"][0] / times["gpu"][0]
            print(f"  {speedup:>7.1f}x", end="")
        print()

    # Bootstrap benchmark (smaller configs)
    print("\n" + "=" * 90)
    print("  Bootstrap SE benchmark (50 replications)")
    print("=" * 90)

    boot_configs = [
        (200,   50,   80, 2, "N=200, T=50, 50 boots"),
        (1000,  30,  400, 2, "N=1k, T=30, 50 boots"),
        (5000,  20, 2000, 2, "N=5k, T=20, 50 boots"),
        (10000, 10, 4000, 2, "N=10k, T=10, 50 boots"),
    ]

    for N, T, N_tr, r, label in boot_configs:
        Y, D, I, II, X = generate_panel(N, T, N_tr, r=r, seed=42)

        # Build T_on and unit_type
        unit_type = np.ones(N, dtype=np.int32)
        unit_type[:N_tr] = 2
        T_on = np.full((T, N), np.nan)
        for j in range(N):
            first_treat = np.where(D[:, j] > 0)[0]
            if len(first_treat) > 0:
                t0 = first_treat[0]
                for t in range(T):
                    T_on[t, j] = t - t0

        times = {}
        for device in devices:
            try:
                elapsed, se = benchmark_bootstrap(
                    Y, D, I, II, T_on, X, unit_type, r=r, force=3,
                    device=device, nboots=50,
                )
                times[device] = elapsed
                se_val = se
            except Exception as e:
                times[device] = None

        print(f"  {label:30s}", end="")
        for d in devices:
            t = times.get(d)
            if t is not None:
                print(f"  {d}: {t:>8.2f}s", end="")
            else:
                print(f"  {d}: FAIL", end="")
        if len(devices) > 1 and times.get("cpu") and times.get("gpu"):
            speedup = times["cpu"] / times["gpu"]
            print(f"  speedup: {speedup:.1f}x", end="")
        print()


if __name__ == "__main__":
    run_benchmarks()
