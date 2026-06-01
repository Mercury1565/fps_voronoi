"""
Phase 2 — fused Voronoi diagnostics.

Fuses Phase 1's per-point primitives (membership, covering radius, occupancy)
into a single sweep over P, then flags three classes of bad cells: coverage gaps
(under-sampled), separation violators (redundant samples), vanishing cells
(wasted samples). The cdist over P × S runs once; conventions match Phase 1
(int64 ids, float32 dists, lowest-index tie-break, outputs on the input device).
"""

from dataclasses import dataclass

import torch

_CHUNK = 4096


# ─────────────────────────────────────────────────────────────────────────────
# Result containers
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class FusedStats:
    """Output of the fused pass + reductions.

    Per-point (N): nearest_id (owning cell), nearest_dist.
    Per-cell (M): covering_radius (max member dist, 0 if empty), occupancy,
        farthest_idx (P index of farthest member, -1 if empty), sample_nn_dist
        and sample_nn_idx (nearest *other* sample, -1 if M < 2).
    CSR point lists: cell c owns csr_order[csr_offsets[c]:csr_offsets[c+1]].
    Scalars: min_pairwise + min_pair (closest two samples, i < j; inf/-1 if M<2).
    """

    nearest_id: torch.Tensor
    nearest_dist: torch.Tensor
    covering_radius: torch.Tensor
    occupancy: torch.Tensor
    farthest_idx: torch.Tensor
    csr_order: torch.Tensor
    csr_offsets: torch.Tensor
    sample_nn_dist: torch.Tensor
    sample_nn_idx: torch.Tensor
    min_pairwise: torch.Tensor
    min_pair: torch.Tensor

    def cell_points(self, c: int) -> torch.Tensor:
        """Point indices belonging to cell ``c`` (from the CSR)."""
        lo = int(self.csr_offsets[c])
        hi = int(self.csr_offsets[c + 1])
        return self.csr_order[lo:hi]


@dataclass
class Flags:
    """Detector output.

    coverage_gap_cells / coverage_insertion_points: over-large cells and the
        farthest member of each (where to insert a sample).
    separation_cells / separation_pairs: too-close samples and their (i<j) pairs.
    vanishing_cells: cells with too few points (removal candidates).
    """

    coverage_gap_cells: torch.Tensor
    coverage_insertion_points: torch.Tensor
    separation_cells: torch.Tensor
    separation_pairs: torch.Tensor
    vanishing_cells: torch.Tensor


# ─────────────────────────────────────────────────────────────────────────────
# 1. Fused single-pass KNN(P, S) + per-cell reductions
# ─────────────────────────────────────────────────────────────────────────────
def fused_knn_pass(P: torch.Tensor, S: torch.Tensor, chunk: int = _CHUNK):
    """One sweep over P producing the assignment and all per-cell reductions.

    Returns (nearest_id, nearest_dist, covering_radius, occupancy, farthest_idx)
    — membership, covering radius and occupancy fused into one cdist loop, plus
    each cell's farthest member (for insertion points).
    """
    M = S.shape[0]
    if M == 0:
        raise ValueError("S must contain at least one sample (M > 0).")

    N = P.shape[0]
    device = P.device

    nearest_id = torch.empty(N, dtype=torch.int64, device=device)
    nearest_dist = torch.empty(N, dtype=torch.float32, device=device)
    covering_radius = torch.zeros(M, dtype=torch.float32, device=device)
    occupancy = torch.zeros(M, dtype=torch.int64, device=device)

    for lo in range(0, N, chunk):
        hi = min(lo + chunk, N)
        d = torch.cdist(P[lo:hi], S)                      # (chunk, M)
        idx = d.argmin(dim=1)                             # 1-NN, lowest index wins
        dist = d.gather(1, idx.unsqueeze(1)).squeeze(1).float()

        nearest_id[lo:hi] = idx
        nearest_dist[lo:hi] = dist

        # Per-chunk reductions, accumulated in place.
        covering_radius.scatter_reduce_(
            0, idx, dist, reduce="amax", include_self=True
        )
        occupancy += torch.bincount(idx, minlength=M)

    # Farthest member per cell: a point is extremal when its distance equals the
    # cell's max; keep the lowest P index among them (deterministic).
    farthest_idx = torch.full((M,), -1, dtype=torch.int64, device=device)
    if N > 0:
        is_max = nearest_dist >= covering_radius[nearest_id] - 1e-12
        cand = torch.nonzero(is_max, as_tuple=False).squeeze(1)
        big = torch.full((M,), N, dtype=torch.int64, device=device)
        big.scatter_reduce_(
            0, nearest_id[cand], cand, reduce="amin", include_self=True
        )
        nonempty = big < N
        farthest_idx[nonempty] = big[nonempty]

    return nearest_id, nearest_dist, covering_radius, occupancy, farthest_idx


# ─────────────────────────────────────────────────────────────────────────────
# CSR per-cell point lists
# ─────────────────────────────────────────────────────────────────────────────
def build_csr(nearest_id: torch.Tensor, occupancy: torch.Tensor):
    """CSR per-cell point lists: cell c owns csr_order[csr_offsets[c]:[c+1]].

    A stable sort by cell id keeps points in ascending original index per cell.
    """
    M = occupancy.shape[0]
    device = nearest_id.device

    csr_offsets = torch.zeros(M + 1, dtype=torch.int64, device=device)
    torch.cumsum(occupancy, dim=0, out=csr_offsets[1:])

    if nearest_id.numel() == 0:
        csr_order = torch.empty(0, dtype=torch.int64, device=device)
    else:
        csr_order = torch.argsort(nearest_id, stable=True)

    return csr_order, csr_offsets


# ─────────────────────────────────────────────────────────────────────────────
# 2. Sample-set reductions: per-sample NN, global min pairwise distance
# ─────────────────────────────────────────────────────────────────────────────
def sample_neighbor_stats(S: torch.Tensor, chunk: int = _CHUNK):
    """Per-sample nearest *other* sample, plus the global min pairwise distance.

    Returns (nn_dist, nn_idx, min_pairwise, min_pair). For M < 2 these are
    +inf / -1 / +inf / (-1, -1).
    """
    M = S.shape[0]
    device = S.device

    nn_dist = torch.full((M,), float("inf"), dtype=torch.float32, device=device)
    nn_idx = torch.full((M,), -1, dtype=torch.int64, device=device)

    if M < 2:
        return (
            nn_dist,
            nn_idx,
            torch.tensor(float("inf"), dtype=torch.float32, device=device),
            torch.tensor([-1, -1], dtype=torch.int64, device=device),
        )

    for lo in range(0, M, chunk):
        hi = min(lo + chunk, M)
        d = torch.cdist(S[lo:hi], S)                      # (rows, M)
        rows = torch.arange(lo, hi, device=device)
        d[torch.arange(hi - lo, device=device), rows] = float("inf")  # mask self
        idx = d.argmin(dim=1)
        nn_idx[lo:hi] = idx
        nn_dist[lo:hi] = d.gather(1, idx.unsqueeze(1)).squeeze(1).float()

    # Global min = smallest per-sample NN distance; canonical (i < j) pair.
    i = int(nn_dist.argmin())
    j = int(nn_idx[i])
    if i > j:
        i, j = j, i
    min_pairwise = nn_dist.min().float()
    min_pair = torch.tensor([i, j], dtype=torch.int64, device=device)

    return nn_dist, nn_idx, min_pairwise, min_pair


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator: gather every statistic in one place
# ─────────────────────────────────────────────────────────────────────────────
def compute_stats(P: torch.Tensor, S: torch.Tensor, chunk: int = _CHUNK) -> FusedStats:
    """Fused pass + CSR lists + sample reductions, bundled into a FusedStats."""
    nearest_id, nearest_dist, covering_radius, occupancy, farthest_idx = (
        fused_knn_pass(P, S, chunk=chunk)
    )
    csr_order, csr_offsets = build_csr(nearest_id, occupancy)
    nn_dist, nn_idx, min_pairwise, min_pair = sample_neighbor_stats(S, chunk=chunk)

    return FusedStats(
        nearest_id=nearest_id,
        nearest_dist=nearest_dist,
        covering_radius=covering_radius,
        occupancy=occupancy,
        farthest_idx=farthest_idx,
        csr_order=csr_order,
        csr_offsets=csr_offsets,
        sample_nn_dist=nn_dist,
        sample_nn_idx=nn_idx,
        min_pairwise=min_pairwise,
        min_pair=min_pair,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Threshold logic & flag generation
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_threshold(absolute, factor, reference):
    """Absolute threshold if given, else factor * reference."""
    if absolute is not None:
        return float(absolute)
    return float(factor) * float(reference)


def detect_flags(
    P: torch.Tensor,
    stats: FusedStats,
    *,
    coverage_radius_max: float | None = None,
    coverage_factor: float = 2.0,
    separation_min: float | None = None,
    separation_factor: float = 0.5,
    min_occupancy: int = 1,
) -> Flags:
    """Apply the three detectors to a FusedStats.

    Coverage gap: covering_radius > coverage_radius_max (default
        coverage_factor * median non-zero radius); each contributes its farthest
        member as an insertion point.
    Separation violator: sample NN distance < separation_min (default
        separation_factor * median NN distance); returns ids and (i<j) pairs.
    Vanishing cell: occupancy < min_occupancy (default 1 = empty).

    Thresholds are absolute or a multiple of a robust median reference.
    """
    device = stats.nearest_id.device
    M = stats.occupancy.shape[0]

    # ── Coverage gap ─────────────────────────────────────────────────────────
    nz = stats.covering_radius[stats.covering_radius > 0]
    ref_cov = nz.median() if nz.numel() > 0 else torch.tensor(0.0, device=device)
    cov_thr = _resolve_threshold(coverage_radius_max, coverage_factor, ref_cov)

    coverage_gap_cells = torch.nonzero(
        stats.covering_radius > cov_thr, as_tuple=False
    ).squeeze(1)
    # Only non-empty cells can yield an insertion point.
    has_pt = stats.farthest_idx[coverage_gap_cells] >= 0
    coverage_gap_cells = coverage_gap_cells[has_pt]
    pt_idx = stats.farthest_idx[coverage_gap_cells]
    coverage_insertion_points = (
        P[pt_idx] if pt_idx.numel() > 0
        else torch.empty(0, P.shape[1], dtype=P.dtype, device=device)
    )

    # ── Separation violator ──────────────────────────────────────────────────
    if M >= 2:
        finite = stats.sample_nn_dist[torch.isfinite(stats.sample_nn_dist)]
        ref_sep = (
            finite.median() if finite.numel() > 0
            else torch.tensor(0.0, device=device)
        )
        sep_thr = _resolve_threshold(separation_min, separation_factor, ref_sep)
        sep_mask = stats.sample_nn_dist < sep_thr
        separation_cells = torch.nonzero(sep_mask, as_tuple=False).squeeze(1)
        # Canonical (i < j) pairs, de-duplicated.
        i = separation_cells
        j = stats.sample_nn_idx[separation_cells]
        lo = torch.minimum(i, j)
        hi = torch.maximum(i, j)
        pairs = torch.stack([lo, hi], dim=1)
        separation_pairs = torch.unique(pairs, dim=0) if pairs.numel() > 0 else pairs
    else:
        separation_cells = torch.empty(0, dtype=torch.int64, device=device)
        separation_pairs = torch.empty(0, 2, dtype=torch.int64, device=device)

    # ── Vanishing cell ───────────────────────────────────────────────────────
    vanishing_cells = torch.nonzero(
        stats.occupancy < min_occupancy, as_tuple=False
    ).squeeze(1)

    return Flags(
        coverage_gap_cells=coverage_gap_cells,
        coverage_insertion_points=coverage_insertion_points,
        separation_cells=separation_cells,
        separation_pairs=separation_pairs,
        vanishing_cells=vanishing_cells,
    )


def analyze(
    P: torch.Tensor,
    S: torch.Tensor,
    *,
    chunk: int = _CHUNK,
    coverage_radius_max: float | None = None,
    coverage_factor: float = 2.0,
    separation_min: float | None = None,
    separation_factor: float = 0.5,
    min_occupancy: int = 1,
):
    """End-to-end Phase 2 entry point. Returns (stats, flags)."""
    stats = compute_stats(P, S, chunk=chunk)
    flags = detect_flags(
        P,
        stats,
        coverage_radius_max=coverage_radius_max,
        coverage_factor=coverage_factor,
        separation_min=separation_min,
        separation_factor=separation_factor,
        min_occupancy=min_occupancy,
    )
    return stats, flags
