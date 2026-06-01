"""
Phase 3 — correction unit.

Phase 2 *diagnoses* a sampling: it flags coverage gaps (under-sampled cells),
separation violators (redundant, too-close samples) and vanishing cells (samples
that represent almost nothing).  Phase 3 *acts* on those flags — it edits the
sample set and keeps the Voronoi bookkeeping up to date without re-running the
expensive full nearest-neighbour pass.

It does four things:

    1. Insertion       — add a candidate sample p and work out which existing
                         samples become Delaunay neighbours of its new cell.
    2. Eviction        — remove samples in a priority order: vanished first,
                         then underpopulated, then smallest covering radius.
    3. One-hop update  — after an edit, recompute only the affected neighbour
                         cells (membership / occupancy / covering radius), never
                         the whole cloud.
    4. Budget tracking — count edits per frame; if a frame needs more edits than
                         the budget allows, give up on patching and fall back to
                         a fresh full FPS resample.

Why one-hop updates are correct
--------------------------------
Adding a single site p can only steal cloud points from the cells that border
p's new cell — i.e. p's Delaunay neighbours (its nearest existing sample is
always one of them).  No point in a non-adjacent cell can suddenly be closer to
p.  So we only re-examine the members of those cells, and a re-examined point
either stays put or moves to p.

Removing a site e hands e's cell back to e's former Delaunay neighbours — every
orphaned point's new owner is one of them.  So we only reassign e's points, and
only among e's one-hop neighbourhood.

These properties make the incremental update *exact* (it matches a full
recompute) as long as the Delaunay neighbours are exact, which they are because
we lean on Phase 1's ``delaunay_neighbors`` (scipy/Qhull) — the same
"correct-now, GPU-later" stance as the earlier phases.  The only cheap O(N)
operation left per edit is an ``isin`` mask to gather the affected members; the
costly O(N·M) distance work is avoided.

Conventions match Phases 1–2: ``int64`` ids, ``float32`` distances,
lowest-index tie-breaks, outputs on the input device.
"""

import os
import sys
from dataclasses import dataclass

import torch

# Reuse the earlier phases (same path-insertion style as the demos).
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "phase_1"))
sys.path.insert(0, os.path.join(_HERE, "..", "phase_2"))

from primitives import delaunay_neighbors                       # noqa: E402
from fps import farthest_point_sampling                          # noqa: E402
from fused import (                                              # noqa: E402
    analyze,
    compute_stats,
    build_csr,
    sample_neighbor_stats,
    FusedStats,
)


def _accurate_cell_stats(P: torch.Tensor, S: torch.Tensor,
                         nearest_id: torch.Tensor):
    """Per-point distance + per-cell occupancy/radius for a *fixed* assignment.

    Distances use a direct norm rather than ``cdist``: ``cdist`` switches to a
    matmul-based formula for large inputs that loses ~1e-3 of accuracy near zero
    distance, which matters here because inserted samples often sit exactly on a
    cloud point (true distance 0).  Computing them directly keeps the maintained
    state exact and reproducible, independent of input size.
    """
    M = S.shape[0]
    device = S.device
    nearest_dist = (P - S[nearest_id]).norm(dim=1).float()
    occupancy = torch.bincount(nearest_id, minlength=M).to(torch.int64)
    covering_radius = torch.zeros(M, dtype=torch.float32, device=device)
    if nearest_id.numel() > 0:
        covering_radius.scatter_reduce_(
            0, nearest_id, nearest_dist, reduce="amax", include_self=True
        )
    return nearest_dist, occupancy, covering_radius


# ─────────────────────────────────────────────────────────────────────────────
# 1. Insertion — Delaunay neighbours of a new cell
# ─────────────────────────────────────────────────────────────────────────────
def insertion_neighbors(S: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Existing samples that become Delaunay neighbours of a new site ``p``.

    These are exactly the cells whose territory the new cell of ``p`` borders —
    the cells that will lose points to ``p``.  Computed as ``p``'s neighbours in
    the Delaunay triangulation of ``S ∪ {p}`` (Phase 1's ``delaunay_neighbors``).

    Degenerate / tiny ``S`` (too few points or collinear/coplanar, so Qhull
    fails) falls back to "every existing sample is a neighbour", which is a safe
    over-approximation: it only makes the one-hop update examine more points, not
    fewer, so the result stays exact.
    """
    M = S.shape[0]
    if M == 0:
        return torch.empty(0, dtype=torch.int64, device=S.device)

    aug = torch.cat([S, p.reshape(1, -1).to(S)], dim=0)
    try:
        nbrs = delaunay_neighbors(aug)               # list of length M + 1
    except ValueError:
        return torch.arange(M, dtype=torch.int64, device=S.device)

    new = [j for j in nbrs[M] if j < M]              # drop self, keep existing ids
    return torch.tensor(sorted(new), dtype=torch.int64, device=S.device)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Eviction priority order
# ─────────────────────────────────────────────────────────────────────────────
def separation_losers(
    occupancy: torch.Tensor,
    covering_radius: torch.Tensor,
    pairs: torch.Tensor,
) -> torch.Tensor:
    """For each too-close ``(i, j)`` pair pick the sample to drop.

    The "loser" is the weaker member: lower occupancy, then smaller covering
    radius, then higher id (deterministic).  Returns the unique loser ids.
    """
    if pairs.numel() == 0:
        return torch.empty(0, dtype=torch.int64, device=occupancy.device)

    i, j = pairs[:, 0], pairs[:, 1]
    oi, oj = occupancy[i], occupancy[j]
    ri, rj = covering_radius[i], covering_radius[j]
    # Prefer to keep the stronger sample; drop `j` when `i` is the keeper.
    keep_i = (oi > oj) | ((oi == oj) & (ri >= rj))
    losers = torch.where(keep_i, j, i)
    return torch.unique(losers)


def eviction_order(
    occupancy: torch.Tensor,
    covering_radius: torch.Tensor,
    *,
    vanish_max: int = 0,
    underpop_max: int = 0,
    extra: torch.Tensor | None = None,
) -> torch.Tensor:
    """Rank eviction *candidates* by priority.

    Tier 0 — **vanished**: ``occupancy <= vanish_max`` (default 0 → empty cells).
    Tier 1 — **underpopulated**: ``vanish_max < occupancy <= underpop_max``.
    Tier 2 — **smallest covering radius**: any ``extra`` candidates (e.g. the
             separation losers) not already covered, ordered by ascending
             covering radius so the most redundant cells come first.

    Within every tier ties break on ascending covering radius then ascending id.
    The result is the order in which a caller should evict, highest priority
    first; the caller decides how many to actually remove (budget permitting).
    """
    device = occupancy.device
    M = occupancy.shape[0]

    def _sorted(ids: torch.Tensor) -> torch.Tensor:
        if ids.numel() == 0:
            return ids
        # ascending covering radius, id as a stable tie-break
        key = covering_radius[ids]
        order = torch.argsort(key, stable=True)
        return ids[order]

    vanished = torch.nonzero(occupancy <= vanish_max, as_tuple=False).squeeze(1)
    underpop = torch.nonzero(
        (occupancy > vanish_max) & (occupancy <= underpop_max), as_tuple=False
    ).squeeze(1)

    tier0 = _sorted(vanished)
    tier1 = _sorted(underpop)

    if extra is not None and extra.numel() > 0:
        already = torch.zeros(M, dtype=torch.bool, device=device)
        already[tier0] = True
        already[tier1] = True
        extra = extra[~already[extra]]
        tier2 = _sorted(torch.unique(extra))
    else:
        tier2 = torch.empty(0, dtype=torch.int64, device=device)

    return torch.cat([tier0, tier1, tier2])


# ─────────────────────────────────────────────────────────────────────────────
# 3. One-hop incremental state
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class CorrectionState:
    """Mutable Voronoi bookkeeping that survives incremental edits.

    Holds only what the one-hop updates need to maintain cheaply:

        S                (M, D) samples
        nearest_id       (N,)   owning sample per cloud point
        nearest_dist     (N,)   distance to that sample
        occupancy        (M,)   points per cell
        covering_radius  (M,)   max member distance per cell
    """

    S: torch.Tensor
    nearest_id: torch.Tensor
    nearest_dist: torch.Tensor
    occupancy: torch.Tensor
    covering_radius: torch.Tensor

    # ── construction ──────────────────────────────────────────────────────────
    @classmethod
    def from_cloud(cls, P: torch.Tensor, S: torch.Tensor) -> "CorrectionState":
        """Build initial state with one full Phase 2 pass for the assignment,
        then accurate per-cell distances/reductions on top of it."""
        nearest_id = compute_stats(P, S).nearest_id.clone()
        nearest_dist, occupancy, covering_radius = _accurate_cell_stats(P, S, nearest_id)
        return cls(
            S=S.clone(),
            nearest_id=nearest_id,
            nearest_dist=nearest_dist,
            occupancy=occupancy,
            covering_radius=covering_radius,
        )

    @property
    def num_samples(self) -> int:
        return self.S.shape[0]

    # ── insertion ─────────────────────────────────────────────────────────────
    def insert(self, P: torch.Tensor, p: torch.Tensor,
               neighbors: torch.Tensor | None = None) -> int:
        """Add sample ``p``; reassign only the members of its neighbour cells.

        Returns the new sample's id (appended at the end of ``S``).
        """
        S = self.S
        D = S.shape[1]
        p = p.reshape(1, D).to(S)
        new_id = S.shape[0]
        device = S.device

        # Cell p lands in (always a Delaunay neighbour) + the rest of its ring.
        c0 = torch.cdist(p, S).argmin(dim=1)
        if neighbors is None:
            neighbors = insertion_neighbors(S, p.squeeze(0))
        affected = torch.unique(torch.cat([neighbors.to(device), c0]))

        # Grow S and the per-cell arrays by one slot for the new cell.
        self.S = torch.cat([S, p], dim=0)
        self.occupancy = torch.cat(
            [self.occupancy, torch.zeros(1, dtype=self.occupancy.dtype, device=device)]
        )
        self.covering_radius = torch.cat(
            [self.covering_radius,
             torch.zeros(1, dtype=self.covering_radius.dtype, device=device)]
        )

        # Candidate points = current members of the affected cells.
        cand = torch.nonzero(
            torch.isin(self.nearest_id, affected), as_tuple=False
        ).squeeze(1)

        if cand.numel() > 0:
            d_new = (P[cand] - p).norm(dim=1).float()       # accurate direct norm
            move = d_new < self.nearest_dist[cand]          # strictly closer → moves
            moved = cand[move]
            self.nearest_id[moved] = new_id
            self.nearest_dist[moved] = d_new[move]

            # Refresh affected ∪ {new}: every member of these cells is in `cand`,
            # so reset and rebuild from `cand` exactly.
            refresh = torch.cat([affected, torch.tensor([new_id], device=device)])
            self.occupancy[refresh] = 0
            self.covering_radius[refresh] = 0.0
            self.occupancy.scatter_add_(
                0, self.nearest_id[cand],
                torch.ones(cand.shape[0], dtype=self.occupancy.dtype, device=device),
            )
            self.covering_radius.scatter_reduce_(
                0, self.nearest_id[cand], self.nearest_dist[cand],
                reduce="amax", include_self=True,
            )

        return new_id

    # ── eviction ──────────────────────────────────────────────────────────────
    def evict(self, P: torch.Tensor, evict_ids: torch.Tensor,
              delaunay: list | None = None) -> torch.Tensor:
        """Remove ``evict_ids``; reassign their points to one-hop neighbours.

        ``S`` is rebuilt without the evicted samples, so every id shifts down.
        Returns the old→new id map (evicted entries map to ``-1``).
        """
        device = self.S.device
        M = self.S.shape[0]
        evict_ids = torch.unique(evict_ids.to(device))
        if evict_ids.numel() == 0:
            return torch.arange(M, dtype=torch.int64, device=device)

        keep_mask = torch.ones(M, dtype=torch.bool, device=device)
        keep_mask[evict_ids] = False
        if keep_mask.sum() == 0:
            raise ValueError("Cannot evict every sample.")

        if delaunay is None:
            delaunay = delaunay_neighbors(self.S)

        # Candidate new owners: one-hop neighbours of the evicted cells that
        # survive the eviction. A point handed back from cell e goes to one of
        # e's former Delaunay neighbours.
        evict_set = set(evict_ids.tolist())
        cand_set = set()
        for e in evict_ids.tolist():
            cand_set.update(delaunay[e])
        cand_set -= evict_set
        if not cand_set:
            # Whole neighbourhood evicted — fall back to all survivors.
            cand_samples = torch.nonzero(keep_mask, as_tuple=False).squeeze(1)
        else:
            cand_samples = torch.tensor(sorted(cand_set), dtype=torch.int64,
                                        device=device)

        # Orphaned points (members of the evicted cells).
        orphan = torch.nonzero(
            torch.isin(self.nearest_id, evict_ids), as_tuple=False
        ).squeeze(1)
        if orphan.numel() > 0:
            d = torch.cdist(P[orphan], self.S[cand_samples])
            j = d.argmin(dim=1)
            new_owner_old = cand_samples[j]               # still in OLD index space
            self.nearest_id[orphan] = new_owner_old
            # accurate distance to the chosen owner (cdist only picks the owner)
            self.nearest_dist[orphan] = (
                P[orphan] - self.S[new_owner_old]
            ).norm(dim=1).float()

        # Reindex old → new for the survivors.
        new_index = torch.cumsum(keep_mask.long(), dim=0) - 1  # (M,)

        # New per-cell arrays. Survivors keep their counts (which never included
        # orphans); orphans are *added* to their new owners (occupancy +=,
        # radius = max). Gaining cells only grow, so the max is exact.
        occ_keep = self.occupancy[keep_mask].clone()
        rad_keep = self.covering_radius[keep_mask].clone()
        if orphan.numel() > 0:
            owner_new = new_index[self.nearest_id[orphan]]
            occ_keep.scatter_add_(
                0, owner_new,
                torch.ones(orphan.shape[0], dtype=occ_keep.dtype, device=device),
            )
            rad_keep.scatter_reduce_(
                0, owner_new, self.nearest_dist[orphan],
                reduce="amax", include_self=True,
            )

        self.nearest_id = new_index[self.nearest_id]
        self.S = self.S[keep_mask]
        self.occupancy = occ_keep
        self.covering_radius = rad_keep

        id_map = new_index.clone()
        id_map[evict_ids] = -1
        return id_map

    # ── export ──────────────────────────────────────────────────────────────────
    def to_stats(self, P: torch.Tensor) -> FusedStats:
        """Assemble a full :class:`FusedStats` from the maintained fields.

        The expensive membership is already known; only the cheap derived
        aggregates are (re)built — CSR lists, the per-cell farthest member, and
        the sample-set nearest-neighbour reductions.
        """
        device = self.S.device
        N = self.nearest_id.shape[0]
        M = self.S.shape[0]

        csr_order, csr_offsets = build_csr(self.nearest_id, self.occupancy)

        farthest_idx = torch.full((M,), -1, dtype=torch.int64, device=device)
        if N > 0:
            is_max = self.nearest_dist >= self.covering_radius[self.nearest_id] - 1e-12
            cand = torch.nonzero(is_max, as_tuple=False).squeeze(1)
            big = torch.full((M,), N, dtype=torch.int64, device=device)
            big.scatter_reduce_(0, self.nearest_id[cand], cand,
                                reduce="amin", include_self=True)
            nonempty = big < N
            farthest_idx[nonempty] = big[nonempty]

        nn_dist, nn_idx, min_pairwise, min_pair = sample_neighbor_stats(self.S)

        return FusedStats(
            nearest_id=self.nearest_id,
            nearest_dist=self.nearest_dist,
            covering_radius=self.covering_radius,
            occupancy=self.occupancy,
            farthest_idx=farthest_idx,
            csr_order=csr_order,
            csr_offsets=csr_offsets,
            sample_nn_dist=nn_dist,
            sample_nn_idx=nn_idx,
            min_pairwise=min_pairwise,
            min_pair=min_pair,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Budget tracker
# ─────────────────────────────────────────────────────────────────────────────
class BudgetTracker:
    """Counts edits within a frame and reports when the budget is blown.

    An "edit" is one insertion or one eviction.  When the number of edits a
    frame wants exceeds ``max_edits``, the caller should stop patching and do a
    full FPS resample instead — incremental fixes are only worthwhile when few
    are needed.
    """

    def __init__(self, max_edits: int):
        self.max_edits = max_edits
        self.edits = 0

    def record(self, n: int = 1) -> "BudgetTracker":
        self.edits += n
        return self

    @property
    def exceeded(self) -> bool:
        return self.edits > self.max_edits

    @property
    def remaining(self) -> int:
        return max(0, self.max_edits - self.edits)

    def reset(self) -> "BudgetTracker":
        self.edits = 0
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class CorrectionResult:
    """Outcome of correcting one frame.

    S            corrected sample set
    stats        FusedStats for the corrected sampling
    fallback     True if the budget was blown and a full FPS resample was used
    n_inserted   insertions applied (0 on fallback)
    n_evicted    evictions applied (0 on fallback)
    n_requested  edits the frame asked for (drives the budget decision)
    """

    S: torch.Tensor
    stats: FusedStats
    fallback: bool
    n_inserted: int
    n_evicted: int
    n_requested: int


def correct(
    P: torch.Tensor,
    S: torch.Tensor,
    *,
    budget: int,
    fps_seed: int | None = None,
    underpop_max: int = 0,
    max_insertions: int | None = None,
    max_evictions: int | None = None,
    coverage_factor: float = 2.0,
    coverage_radius_max: float | None = None,
    separation_factor: float = 0.5,
    separation_min: float | None = None,
    min_occupancy: int = 1,
) -> CorrectionResult:
    """One correction frame: diagnose with Phase 2, then insert/evict under a
    budget, falling back to a full FPS resample when the budget is exceeded.

    Insertions come from the coverage-gap insertion points; evictions come from
    the vanished / underpopulated / separation-loser priority order.  The frame
    edits ``S`` in place via one-hop updates; if the requested edit count tops
    ``budget`` the whole sampling is recomputed from scratch with FPS instead.
    """
    stats, flags = analyze(
        P, S,
        coverage_factor=coverage_factor,
        coverage_radius_max=coverage_radius_max,
        separation_factor=separation_factor,
        separation_min=separation_min,
        min_occupancy=min_occupancy,
    )

    insertions = flags.coverage_insertion_points
    if max_insertions is not None:
        insertions = insertions[:max_insertions]

    losers = separation_losers(stats.occupancy, stats.covering_radius,
                               flags.separation_pairs)
    evict = eviction_order(
        stats.occupancy, stats.covering_radius,
        vanish_max=min_occupancy - 1,
        underpop_max=underpop_max,
        extra=losers,
    )
    if max_evictions is not None:
        evict = evict[:max_evictions]

    n_requested = int(insertions.shape[0] + evict.shape[0])

    tracker = BudgetTracker(budget).record(n_requested)
    if tracker.exceeded:
        idx = farthest_point_sampling(P, S.shape[0], seed=fps_seed)
        S_new = P[idx].clone()
        return CorrectionResult(
            S=S_new, stats=compute_stats(P, S_new), fallback=True,
            n_inserted=0, n_evicted=0, n_requested=n_requested,
        )

    # Within budget — patch incrementally. Reuse Phase 2's assignment but rebuild
    # distances/reductions accurately so the maintained state stays exact.
    nearest_id = stats.nearest_id.clone()
    nearest_dist, occupancy, covering_radius = _accurate_cell_stats(P, S, nearest_id)
    state = CorrectionState(
        S=S.clone(),
        nearest_id=nearest_id,
        nearest_dist=nearest_dist,
        occupancy=occupancy,
        covering_radius=covering_radius,
    )

    # Insertions first: they only append ids, so the eviction ids (computed on
    # the original S) stay valid through every insertion.
    for k in range(insertions.shape[0]):
        state.insert(P, insertions[k])

    state.evict(P, evict)

    return CorrectionResult(
        S=state.S, stats=state.to_stats(P), fallback=False,
        n_inserted=int(insertions.shape[0]), n_evicted=int(evict.shape[0]),
        n_requested=n_requested,
    )
