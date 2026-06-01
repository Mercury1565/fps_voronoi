# Phase 2 — Fused Voronoi diagnostics

Phase 1 built five standalone geometric primitives, each making its own pass
over the data. Phase 2 **fuses** the per-point primitives into a single sweep
over the point cloud, runs the cheap sample-set reductions, then applies
threshold logic to flag three classes of bad Voronoi cells and propose where to
fix them.

The pipeline answers three questions about a sampling `S` of a cloud `P`:

| Detector | Question | Signal | Fix it suggests |
|----------|----------|--------|-----------------|
| **Coverage gap** | Is any cell covering too large a region? | per-cell covering radius too big | insert a sample at the cell's farthest point |
| **Separation violator** | Are two samples redundantly close? | sample-to-nearest-sample distance too small | merge / drop one of the pair |
| **Vanishing cell** | Does a sample represent (almost) nothing? | per-cell occupancy too low | remove / relocate the sample |

## Files

```
phase_2/
├── fused.py            # the whole pipeline (no extra deps beyond torch)
├── demo.py             # 2-D visualisation of the three detectors
├── README.md
└── tests/              # pytest + hypothesis, incl. golden checks vs phase_1
```

## The pipeline

### 1. Fused single-pass KNN + reductions — `fused_knn_pass(P, S)`

One chunked loop over `P` computes the `cdist(P, S)` **exactly once** and, in
that same loop, accumulates every per-cell reduction that depends on the
point→sample assignment:

- `nearest_id`, `nearest_dist` — 1-NN of each point over `S` (lowest-index
  tie-break, matching Phase 1's `cell_membership`)
- `covering_radius[c]` — max member distance per cell (`scatter_reduce` amax)
- `occupancy[c]` — point count per cell (`bincount`)
- `farthest_idx[c]` — index into `P` of the farthest member of cell `c`
  (`-1` for empty cells), derived with light 1-D ops so it needs **no second
  pass** over `P`. This is the natural candidate insertion point.

`build_csr(nearest_id, occupancy)` then produces compressed per-cell point
lists: cell `c` owns `csr_order[csr_offsets[c] : csr_offsets[c+1]]`.

> This is operation-level fusion in PyTorch — one sweep, device-agnostic
> (CPU/GPU). The interface is stable so a hand-written CUDA/Triton kernel can
> drop in behind `fused_knn_pass` later, the same way Phase 1 plans to replace
> the scipy Delaunay step.

### 2. Sample-set reductions — `sample_neighbor_stats(S)`

For every sample, the distance and id of its nearest *other* sample, plus the
global minimum pairwise distance and its canonical `(i < j)` pair. The
per-sample nearest-neighbour distance is what lets the separation detector flag
*all* offenders, not just the single closest pair. (`M < 2` returns `+inf` /
`-1` sentinels.)

### 3. Threshold logic & flags — `detect_flags(...)`

- **Coverage gap**: cells with `covering_radius > threshold` → flagged cell ids
  **plus** `coverage_insertion_points` (the farthest member of each gap cell).
- **Separation violator**: samples whose nearest neighbour is closer than
  `threshold` → flagged sample ids **plus** de-duplicated canonical
  `separation_pairs`.
- **Vanishing cell**: cells with `occupancy < min_occupancy` (default `1`,
  i.e. empty).

Thresholds are either **absolute** or a multiple of a robust **median**
reference, so behaviour is predictable and testable:

| Param | Absolute override | Default (median-relative) |
|-------|-------------------|---------------------------|
| `coverage_radius_max` | exact radius cutoff | `coverage_factor * median(non-zero radii)`, `factor=2.0` |
| `separation_min` | exact distance cutoff | `separation_factor * median(sample NN dist)`, `factor=0.5` |
| `min_occupancy` | — | `1` |

## Usage

```python
from fused import analyze

stats, flags = analyze(P, S)            # one call: fused stats + flags

flags.coverage_gap_cells                # (K1,) cell ids
flags.coverage_insertion_points         # (K1, D) where to add samples
flags.separation_cells                  # (K2,) sample ids too close
flags.separation_pairs                  # (K2, 2) offending (i<j) pairs
flags.vanishing_cells                   # (K3,) cell ids to remove/relocate

# tune the detectors
stats, flags = analyze(
    P, S,
    coverage_factor=1.6,      # or coverage_radius_max=<abs>
    separation_factor=0.45,   # or separation_min=<abs>
    min_occupancy=1,
)
```

`analyze` returns a `FusedStats` (all per-point / per-cell tensors, the CSR
lists, sample-NN stats, global min pairwise) and a `Flags` dataclass. The
individual stages (`fused_knn_pass`, `build_csr`, `sample_neighbor_stats`,
`compute_stats`, `detect_flags`) are also exported for finer control.

## Demo

```bash
python demo.py                 # synthetic mimic LiDAR scene → phase2_demo.png
python demo.py frame.bin       # KITTI .bin (X,Y used)
python demo.py --samples 80 --coverage-factor 1.6 --separation-factor 0.45
```

The demo builds a deliberately *imperfect* sample set (a random subsample plus a
few stray samples in empty space) so all three detectors fire, and renders a
2×2 panel: cell overview, coverage gaps with insertion points, separation
violators, and vanishing cells.

![Phase 2 detectors](phase2_demo.png)

## Tests

```bash
../venv/bin/python -m pytest tests/ -q
```

The suite's correctness contract is a set of **golden cross-checks**: the fused
outputs must match Phase 1's standalone `cell_membership`, `covering_radius`,
`cell_occupancy`, and `min_pairwise_distance` exactly. It also covers chunk-size
invariance, CSR partition correctness, `farthest_idx` semantics, and each
detector's threshold logic on controlled geometry plus hypothesis property
tests.
