# Phase 1 — Voronoi Primitives

Five standalone functions that answer geometric questions about the implicit Voronoi tessellation defined by a sample set **S** over a point cloud **P** — without ever materialising the tessellation.

---

## Conventions

| Convention | Detail |
|---|---|
| **Tensor layout** | `P: (N, 3) float32`, `S: (M, 3) float32` |
| **Cell ID** | An integer index into `S`. Point `p` belongs to cell `k` iff `S[k]` is its nearest sample. No Voronoi object exists. |
| **Device** | All outputs live on `P.device` (or `S.device`). No silent CPU copies. |
| **Tie-breaking** | When two samples are equidistant, the lower index wins (`argmin` semantics). |
| **Dtype** | `cell_ids` → `int64`; distances, radii, scalar distances → `float32`. |

---

## Primitives

### `cell_membership(P, S) → (cell_ids, distances)`
```
cell_ids  : (N,) int64   — nearest sample index for each point
distances : (N,) float32 — Euclidean distance to that sample
```
Raises `ValueError` if `M == 0`.

### `covering_radius(cell_ids, distances, num_samples) → radii`
```
radii : (M,) float32 — max distance within each cell; 0.0 for empty cells
```

### `min_pairwise_distance(S) → (distance, pair_indices)`
```
distance     : scalar float32
pair_indices : (2,) int64, pair[0] < pair[1]
```
Raises `ValueError` if `M < 2`.

### `cell_occupancy(cell_ids, num_samples) → counts`
```
counts : (M,) int64 — number of cloud points in each cell
```

### `delaunay_neighbors(S) → list[list[int]]`
```
neighbors[i] : sorted list of sample indices Delaunay-adjacent to i
```
Raises `ValueError` for geometrically degenerate S (Qhull failure).

---

## Setup

All three phases share one virtual environment at the repo root. From the repo
root:

```bash
python -m venv venv                      # if not already present
venv/bin/pip install -r requirements.txt
```

Python ≥ 3.9 and PyTorch ≥ 2.0 are required.

---

## Running tests

```bash
cd phase_1
../venv/bin/python -m pytest tests/ -v
```

Each primitive has hand-built unit tests, Hypothesis property tests, and
(where applicable) a CPU-vs-CUDA cross-backend agreement test.

---

## Demo

```bash
# Synthetic data (fallback)
../venv/bin/python demo.py

# Real KITTI velodyne frame
../venv/bin/python demo.py /path/to/000000.bin
```

Opens an Open3D window with the point cloud coloured by cell ID and
samples drawn in red.

---

## Profiling

```bash
../venv/bin/python profile_primitives.py
```

Reports mean ± std runtime (ms) for each primitive at N=100 k, M=1024,
and identifies the bottleneck (input to Phase 2 fusion decisions).
