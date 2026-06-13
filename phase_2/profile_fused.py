import sys
import os
import time
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                                # phase_2/
sys.path.insert(0, os.path.join(_HERE, "..", "phase_1"))  # phase_1/

from fused import (
    fused_knn_pass,
    build_csr,
    sample_neighbor_stats,
    compute_stats,
    detect_flags,
    analyze,
)
from primitives import cell_membership, covering_radius, cell_occupancy

N = 100_000
M = 1024
WARMUP = 10
RUNS = 20


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_fn(fn, *args):
    for _ in range(WARMUP):
        fn(*args)
    sync()

    times = []
    for _ in range(RUNS):
        sync()
        t0 = time.perf_counter()
        fn(*args)
        sync()
        times.append((time.perf_counter() - t0) * 1_000)

    mean = sum(times) / len(times)
    var = sum((t - mean) ** 2 for t in times) / len(times)
    return mean, var ** 0.5


def warmup_gpu(fn, *args, seconds: float = 2.0):
    """Run `fn` repeatedly for `seconds` so GPU clocks reach steady-state
    Boost before any timed measurement starts. No-op on CPU."""
    if not torch.cuda.is_available():
        return
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        fn(*args)
    torch.cuda.synchronize()


def unfused_pass(P, S):
    """Phase 1's three separate sweeps — what fused_knn_pass replaces."""
    cell_ids, distances = cell_membership(P, S)
    covering_radius(cell_ids, distances, S.shape[0])
    cell_occupancy(cell_ids, S.shape[0])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"N={N:,}  M={M:,}  warmup={WARMUP}  runs={RUNS}\n")

    torch.manual_seed(42)
    P = torch.randn(N, 3, device=device)
    # Random subsample — quality doesn't affect pass timing
    idx = torch.randperm(N, device=device)[:M]
    S = P[idx]

    warmup_gpu(fused_knn_pass, P, S)

    # Pre-compute stats for downstream timings (detect_flags needs them)
    stats = compute_stats(P, S)

    results: dict[str, float] = {}
    rows: list[tuple[str, float, float]] = []

    def record(name, fn, *args):
        mean, std = time_fn(fn, *args)
        results[name] = mean
        rows.append((name, mean, std))

    record("fused_knn_pass",            fused_knn_pass,        P, S)
    record("build_csr",                 build_csr,             stats.nearest_id, stats.occupancy)
    record("sample_neighbor_stats",     sample_neighbor_stats, S)
    record("compute_stats (full)",      compute_stats,         P, S)
    record("detect_flags",              detect_flags,          P, stats)
    record("analyze (end-to-end)",      analyze,               P, S)
    record("phase1 unfused (3 sweeps)", unfused_pass,          P, S)

    col = 30
    print(f"{'Operation':<{col}} {'Mean (ms)':>12} {'Std (ms)':>10}")
    print("─" * (col + 24))
    for name, mean, std in rows:
        print(f"{name:<{col}} {mean:>12.3f} {std:>10.3f}")

    bottleneck = max(results, key=results.__getitem__)
    print(f"\nBottleneck → {bottleneck}  ({results[bottleneck]:.3f} ms mean)")

    fusion_speedup = results["phase1 unfused (3 sweeps)"] / results["fused_knn_pass"]
    print(f"Fusion speedup (unfused / fused_knn_pass): {fusion_speedup:.2f}x")


if __name__ == "__main__":
    main()
