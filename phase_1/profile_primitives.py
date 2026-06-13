import sys
import os
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from primitives import (
    cell_membership,
    cell_occupancy,
    covering_radius,
    delaunay_neighbors,
    min_pairwise_distance,
)

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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"N={N:,}  M={M:,}  warmup={WARMUP}  runs={RUNS}\n")

    torch.manual_seed(42)
    P = torch.randn(N, 3, device=device)
    # Random subsample — quality doesn't affect primitive timing
    idx = torch.randperm(N, device=device)[:M]
    S = P[idx]

    warmup_gpu(cell_membership, P, S)

    # Pre-compute cell_membership output for downstream primitives
    cell_ids, distances = cell_membership(P, S)
    S_cpu = S.cpu()

    results: dict[str, float] = {}
    rows: list[tuple[str, float, float]] = []

    def record(name, fn, *args):
        mean, std = time_fn(fn, *args)
        results[name] = mean
        rows.append((name, mean, std))

    record("cell_membership",     cell_membership,    P, S)
    record("covering_radius",     covering_radius,    cell_ids, distances, M)
    record("min_pairwise_dist",   min_pairwise_distance, S)
    record("cell_occupancy",      cell_occupancy,     cell_ids, M)
    record("delaunay_neighbors",  delaunay_neighbors, S_cpu)

    col = 26
    print(f"{'Primitive':<{col}} {'Mean (ms)':>12} {'Std (ms)':>10}")
    print("─" * (col + 24))
    for name, mean, std in rows:
        print(f"{name:<{col}} {mean:>12.3f} {std:>10.3f}")

    bottleneck = max(results, key=results.__getitem__)
    print(f"\nBottleneck → {bottleneck}  ({results[bottleneck]:.3f} ms mean)")

if __name__ == "__main__":
    main()
