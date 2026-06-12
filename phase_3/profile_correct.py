import sys
import os
import time
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                                # phase_3/
sys.path.insert(0, os.path.join(_HERE, "..", "phase_2"))  # phase_2/
sys.path.insert(0, os.path.join(_HERE, "..", "phase_1"))  # phase_1/

from correct import CorrectionState, correct
from fused import compute_stats

N = 100_000
M = 256
WARMUP = 10
RUNS = 20
# Caps Delaunay calls per correct() frame: a random S can flag dozens of
# coverage gaps, and each insertion does one Qhull triangulation.
MAX_EDITS = 8


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


def time_fn_with_setup(setup, work):
    """Like time_fn, but `setup()` (excluded from timing) builds fresh args for
    each call. Needed for insert/evict, which mutate the CorrectionState."""
    for _ in range(WARMUP):
        args = setup()
        sync()
        work(*args)
    sync()

    times = []
    for _ in range(RUNS):
        args = setup()
        sync()
        t0 = time.perf_counter()
        work(*args)
        sync()
        times.append((time.perf_counter() - t0) * 1_000)

    mean = sum(times) / len(times)
    var = sum((t - mean) ** 2 for t in times) / len(times)
    return mean, var ** 0.5


def clone_state(state: CorrectionState) -> CorrectionState:
    return CorrectionState(
        S=state.S.clone(),
        nearest_id=state.nearest_id.clone(),
        nearest_dist=state.nearest_dist.clone(),
        occupancy=state.occupancy.clone(),
        covering_radius=state.covering_radius.clone(),
    )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"N={N:,}  M={M:,}  warmup={WARMUP}  runs={RUNS}\n")

    torch.manual_seed(42)
    P = torch.randn(N, 3, device=device)
    # Random subsample — a deliberately imperfect sampling so correct() has
    # something to fix.
    idx = torch.randperm(N, device=device)[:M]
    S = P[idx].clone()

    base_state = CorrectionState.from_cloud(P, S)
    insertion_point = P[torch.randint(0, N, (1,), device=device)].squeeze(0)

    results: dict[str, float] = {}
    rows: list[tuple[str, float, float]] = []

    def record(name, fn, *args):
        mean, std = time_fn(fn, *args)
        results[name] = mean
        rows.append((name, mean, std))

    def record_setup(name, setup, work):
        mean, std = time_fn_with_setup(setup, work)
        results[name] = mean
        rows.append((name, mean, std))

    # Full O(N*M) passes — the cost a one-hop update is meant to avoid.
    record("from_cloud (full build)",       CorrectionState.from_cloud, P, S)
    record("compute_stats (full recompute)", compute_stats,             P, S)

    # One-hop edits, each on a fresh clone of base_state so insert/evict
    # don't compound across iterations (insert grows S, evict shrinks it).
    record_setup(
        "state.insert (one-hop)",
        lambda: (clone_state(base_state), insertion_point),
        lambda state, p: state.insert(P, p),
    )
    record_setup(
        "state.evict (one-hop)",
        lambda: (clone_state(base_state), torch.tensor([0], device=device)),
        lambda state, ids: state.evict(P, ids),
    )

    # End-to-end correction frame (diagnose with Phase 2, patch under budget).
    record(
        "correct (within budget)",
        lambda: correct(
            P, S, budget=10_000,
            max_insertions=MAX_EDITS, max_evictions=MAX_EDITS,
        ),
    )

    col = 30
    print(f"{'Operation':<{col}} {'Mean (ms)':>12} {'Std (ms)':>10}")
    print("─" * (col + 24))
    for name, mean, std in rows:
        print(f"{name:<{col}} {mean:>12.3f} {std:>10.3f}")

    insert_speedup = results["compute_stats (full recompute)"] / results["state.insert (one-hop)"]
    evict_speedup = results["compute_stats (full recompute)"] / results["state.evict (one-hop)"]
    print(f"\nOne-hop insert vs full recompute: {insert_speedup:.1f}x cheaper")
    print(f"One-hop evict  vs full recompute: {evict_speedup:.1f}x cheaper")
    print(
        "\nNote: state.insert/evict call scipy's Delaunay (CPU-only, via Qhull) "
        "to find\nDelaunay neighbours, so their wall-clock time is largely "
        "CPU-bound regardless\nof device — GPU speedup shows up mainly in the "
        f"full-pass rows above.\n'correct' caps insertions/evictions at "
        f"{MAX_EDITS} each, so its Delaunay cost\nstays bounded regardless of "
        "how many cells the random sample flags."
    )


if __name__ == "__main__":
    main()
