"""Large-scale GPU vs CPU benchmark."""
import sys, time
sys.path.insert(0, ".")
from benchmarks.gpu_vs_cpu import generate_panel, benchmark_single

configs = [
    (100000, 10, 40000, 2),
    (100000, 20, 40000, 2),
    (100000, 50, 40000, 2),
    (200000, 10, 80000, 2),
    (200000, 20, 80000, 2),
]

for N, T, Nt, r in configs:
    Y, D, I, II, X = generate_panel(N, T, Nt, r=r, seed=42)
    t_cpu, niter_cpu, _ = benchmark_single(Y, II, X, r=r, force=3, device="cpu")
    t_gpu, niter_gpu, _ = benchmark_single(Y, II, X, r=r, force=3, device="gpu")
    print(f"N={N} T={T}: cpu={t_cpu:.2f}s gpu={t_gpu:.2f}s speedup={t_cpu/t_gpu:.1f}x iter={niter_cpu}")
