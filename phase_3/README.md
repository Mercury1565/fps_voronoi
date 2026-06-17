# Phase 3 — Correction unit

Phase 2 *diagnoses* a sampling — it flags coverage gaps, separation violators
and vanishing cells. Phase 3 *acts* on those flags: it edits the sample set and
keeps the Voronoi bookkeeping (who-belongs-to-whom, cell sizes, occupancy) up to
date **without re-running the expensive full nearest-neighbour pass** every time.

This is the piece that closes the loop — it turns the diagnostics into an actual
resampler.

## What it does

| Component | Role |
|-----------|------|
| **Insertion** | Add a candidate sample `p` and work out which existing samples become Delaunay neighbours of its new cell. |
| **Eviction** | Remove samples in a priority order: vanished first, then underpopulated, then smallest covering radius. |
| **One-hop update** | After an edit, recompute only the affected neighbour cells — never the whole cloud. |
| **Budget tracker** | Count edits per frame; if a frame needs more edits than allowed, fall back to a fresh full FPS resample. |

## Files

```
phase_3/
├── correct.py          # the whole correction unit (reuses phases 1 & 2)
├── README.md
└── tests/              # golden "incremental == full recompute" checks + more
```

---

## 1. Insertion — `insertion_neighbors(S, p)`

When you drop a new sample `p` into the set, its Voronoi cell carves territory
out of the cells around it. The cells it borders are exactly `p`'s new **Delaunay
neighbours** — and they're the only cells that can lose points to `p`.

We find them by triangulating `S ∪ {p}` (Phase 1's `delaunay_neighbors`) and
reading off `p`'s neighbours. The sample whose cell `p` lands in is always one of
them. Degenerate inputs (too few or collinear samples, where Qhull fails) fall
back to "every sample is a neighbour" — a safe over-approximation that only makes
the next step look at more points, never fewer.

## 2. Eviction — `eviction_order(...)` and `separation_losers(...)`

`eviction_order` ranks removal candidates by a three-tier priority, exactly as
the phase calls for:

1. **Vanished** — `occupancy <= vanish_max` (default: empty cells).
2. **Underpopulated** — `vanish_max < occupancy <= underpop_max`.
3. **Smallest covering radius** — any extra candidates (e.g. the
   separation losers), most-redundant cell first.

Ties inside a tier break on ascending covering radius, then id.

`separation_losers` decides, for each too-close pair flagged by Phase 2, *which*
of the two to drop: the weaker one (lower occupancy, then smaller radius). Those
losers feed into tier 3 of the eviction order.

## 3. One-hop incremental update — `CorrectionState`

`CorrectionState` holds the minimal bookkeeping the updates need: the samples
`S`, each point's owner (`nearest_id`) and distance (`nearest_dist`), and the
per-cell `occupancy` and `covering_radius`.

- **`insert(P, p)`** — only the members of `p`'s neighbour cells are re-examined;
  each either stays put or moves to `p`. The affected cells (plus the new one)
  are then rebuilt from just those members.
- **`evict(P, ids)`** — only the orphaned points (members of the removed cells)
  are reassigned, and only among the removed cells' surviving one-hop
  neighbours. Survivors' counts are kept; orphans are *added* to their new
  owners. `S` is rebuilt without the evicted samples and an old→new id map is
  returned.

### Why this is exact

Adding one site can only steal points from its Delaunay neighbours; removing one
site hands its cell back to *its* Delaunay neighbours. So the points that can
change owner are precisely the members of those one-hop neighbourhoods. Because
we get the neighbours exactly (via Qhull), the incremental result **matches a
full recompute** — which is exactly what the test suite asserts. The only O(N)
work left per edit is a cheap `isin` mask to gather the affected members; the
costly O(N·M) distance comparison is avoided.

> **One-hop assumption.** Reassignment looks one Delaunay hop out from an evicted
> cell. That's exact for a single eviction and for well-separated batches. If you
> evict a large *contiguous* block of cells at once, an orphan's true nearest
> survivor could be two hops away; the code widens to the union of the block's
> neighbours and falls back to all survivors if that union is empty, but the
> clean guarantee is for spread-out evictions. The budget + FPS fallback exists
> precisely for the messy, many-edit cases.

### A note on numerical accuracy

Distances are computed with a direct norm rather than `torch.cdist`. For large
inputs `cdist` uses a matmul-based formula that loses ~`1e-3` of accuracy near
zero distance — and inserted samples often sit exactly on a cloud point (true
distance 0). Computing directly keeps the maintained state exact and independent
of input size.

## 4. Budget tracker + `correct(...)`

`BudgetTracker` counts edits (one insertion or one eviction = one edit) within a
frame and reports `exceeded` once the count tops `max_edits`.

`correct(P, S, budget=...)` is the end-to-end frame:

1. Diagnose with Phase 2 (`analyze`).
2. Gather insertions (coverage-gap insertion points) and evictions (the priority
   order above).
3. If the requested edit count exceeds `budget` → **fall back**: recompute the
   whole sampling from scratch with FPS (`farthest_point_sampling`). Incremental
   patching only pays off when few edits are needed.
4. Otherwise patch incrementally (insertions first — they only append ids, so the
   eviction ids stay valid — then evictions) and return the corrected `S` with
   fresh stats.

It returns a `CorrectionResult`: the new `S`, its `FusedStats`, whether it fell
back, and the insertion / eviction / requested-edit counts.

## Usage

```python
from correct import correct

res = correct(
    P, S,
    budget=40,             # max edits before full-FPS fallback
    coverage_factor=1.6,   # Phase 2 detector thresholds pass straight through
    separation_factor=0.5,
    min_occupancy=1,
    underpop_max=3,        # cells with 1..3 points count as underpopulated
    fps_seed=0,            # determinism for the fallback path
)

res.S            # corrected samples
res.stats        # FusedStats for the corrected sampling (ready to re-diagnose)
res.fallback     # True if a full FPS resample was used
res.n_inserted, res.n_evicted, res.n_requested
```

Iterating `correct` drives a bad sampling toward a healthy one and converges to a
stable fixed point (no edits requested). On a deliberately bad random 64-sample
set over the mimic LiDAR scene:

```
start   : maxR=23.66  empty=0  minPair=0.300  M=64
iter 0  : maxR=15.03  empty=0  minPair=0.775  M=65   [patch +10/-9]
iter 1  : maxR=15.03  empty=0  minPair=2.052  M=59   [patch  +0/-6]
iter 2  : maxR=15.03  empty=0  minPair=2.570  M=52   [patch  +0/-7]
iter 3  : maxR=15.03  empty=0  minPair=3.718  M=49   [patch  +0/-3]
iter 4  : maxR=15.03  empty=0  minPair=3.718  M=49   [patch  +0/-0]  ← converged
```

The worst cell shrinks and the closest-pair spacing grows by 12× as redundant
samples are evicted.

## Lower-level API

The pieces are exported individually for finer control:

- `insertion_neighbors(S, p)` → Delaunay neighbours of a would-be new cell
- `eviction_order(occupancy, covering_radius, *, vanish_max, underpop_max, extra)`
- `separation_losers(occupancy, covering_radius, pairs)`
- `CorrectionState.from_cloud(P, S)` then `.insert(...)` / `.evict(...)` /
  `.to_stats(P)`
- `BudgetTracker(max_edits)`

## Tests

```bash
../venv/bin/python -m pytest tests/ -q
```

The correctness contract is the **golden check**: after any sequence of
insertions and evictions, the incrementally maintained `nearest_id`,
`occupancy`, `nearest_dist` and `covering_radius` must equal a full recompute on
the edited sample set. The suite also covers neighbour identification vs. full
Delaunay, the eviction priority tiers, the budget tracker, and the `correct`
frame including the FPS fallback.

## Profiling

```bash
../venv/bin/python profile_correct.py
```

Benchmarks the O(N·M) cost that one-hop updates exist to avoid against the
updates themselves, at N=100 k, M=256:

| Row | What it times |
|---|---|
| `from_cloud (full build)` | Cold-start full pass — `CorrectionState.from_cloud(P, S)`. |
| `compute_stats (full recompute)` | The O(N·M) pass `insert`/`evict` are designed to skip. |
| `state.insert (one-hop)` | One insertion plus its one-hop neighbour rebuild. |
| `state.evict (one-hop)` | One eviction plus its one-hop reassignment. |
| `correct (within budget)` | A full diagnose-and-patch frame, capped at `MAX_EDITS` (8) insertions/evictions each. |

`insert`/`evict` resolve Delaunay neighbours via scipy/Qhull — a CPU-bound cost
that's largely independent of `N`. That makes the one-hop-vs-full-recompute
ratio **scale- and device-dependent**: on CPU, where `compute_stats` itself is
expensive, one-hop updates win outright; on GPU at this `N`/`M`, `compute_stats`
gets cheap enough that the fixed Qhull cost can dominate instead. The crossover
shifts with `N` — one-hop's advantage returns once the full O(N·M) pass is
costly enough to outweigh Qhull's fixed overhead.
