"""Eviction: priority ordering + exact one-hop reassignment of orphans."""

import pytest
import torch

from fused import compute_stats
from correct import CorrectionState, eviction_order, separation_losers


def _assert_state_matches_full(state, P):
    ref = compute_stats(P, state.S)
    assert (state.nearest_id == ref.nearest_id).all()
    acc_dist = (P - state.S[state.nearest_id]).norm(dim=1)
    torch.testing.assert_close(state.nearest_dist, acc_dist, atol=1e-5, rtol=1e-5)
    assert (state.occupancy == ref.occupancy).all()
    acc_rad = torch.zeros(state.num_samples)
    acc_rad.scatter_reduce_(0, state.nearest_id, acc_dist, reduce="amax", include_self=True)
    torch.testing.assert_close(state.covering_radius, acc_rad, atol=1e-5, rtol=1e-5)


# ── eviction_order priority ──────────────────────────────────────────────────
def test_priority_vanished_before_underpopulated_before_radius():
    #            cell:   0      1      2      3
    occupancy = torch.tensor([0,     2,     50,    50])
    radius =    torch.tensor([0.0,   1.0,   0.3,   9.0])
    extra = torch.tensor([2, 3])  # separation losers, resolved by radius
    order = eviction_order(occupancy, radius,
                           vanish_max=0, underpop_max=5, extra=extra)
    # tier0 vanished = {0}; tier1 underpop = {1}; tier2 extra by radius = [2, 3]
    assert order.tolist() == [0, 1, 2, 3]


def test_order_sorts_within_tier_by_radius():
    occupancy = torch.tensor([0, 0, 0])
    radius = torch.tensor([5.0, 1.0, 3.0])
    order = eviction_order(occupancy, radius, vanish_max=0)
    assert order.tolist() == [1, 2, 0]  # ascending radius


def test_extra_deduped_against_higher_tiers():
    occupancy = torch.tensor([0, 10, 10])
    radius = torch.tensor([0.0, 1.0, 2.0])
    extra = torch.tensor([0, 2])  # 0 already vanished → dropped from tier2
    order = eviction_order(occupancy, radius, vanish_max=0, extra=extra)
    assert order.tolist() == [0, 2]


def test_no_candidates_returns_empty():
    occupancy = torch.tensor([10, 20, 30])
    radius = torch.tensor([1.0, 2.0, 3.0])
    order = eviction_order(occupancy, radius, vanish_max=0, underpop_max=0)
    assert order.numel() == 0


# ── separation_losers ────────────────────────────────────────────────────────
def test_separation_loser_is_weaker_member():
    occupancy = torch.tensor([100, 3, 50])
    radius = torch.tensor([5.0, 1.0, 4.0])
    pairs = torch.tensor([[0, 1]])  # 1 has lower occupancy → loses
    assert separation_losers(occupancy, radius, pairs).tolist() == [1]


def test_separation_loser_tie_breaks_on_radius():
    occupancy = torch.tensor([10, 10])
    radius = torch.tensor([5.0, 2.0])
    pairs = torch.tensor([[0, 1]])  # equal occ → smaller radius (1) loses
    assert separation_losers(occupancy, radius, pairs).tolist() == [1]


def test_separation_losers_empty():
    occ = torch.tensor([1, 2])
    rad = torch.tensor([1.0, 2.0])
    out = separation_losers(occ, rad, torch.empty(0, 2, dtype=torch.int64))
    assert out.numel() == 0


# ── one-hop eviction equals full recompute ───────────────────────────────────
def test_single_eviction_exact():
    torch.manual_seed(0)
    P = torch.randn(2000, 3)
    S = torch.randn(40, 3)
    state = CorrectionState.from_cloud(P, S)
    id_map = state.evict(P, torch.tensor([7]))
    assert state.num_samples == 39
    assert id_map[7].item() == -1
    _assert_state_matches_full(state, P)


def test_spread_out_batch_eviction_exact():
    torch.manual_seed(1)
    P = torch.randn(3000, 3)
    S = torch.randn(50, 3)
    state = CorrectionState.from_cloud(P, S)
    # Evict a few well-separated samples (non-adjacent → one-hop is exact).
    state.evict(P, torch.tensor([2, 20, 40]))
    assert state.num_samples == 47
    _assert_state_matches_full(state, P)


def test_eviction_id_map_remaps_survivors():
    torch.manual_seed(2)
    P = torch.randn(1000, 3)
    S = torch.randn(10, 3)
    state = CorrectionState.from_cloud(P, S)
    id_map = state.evict(P, torch.tensor([3]))
    # ids 0..2 unchanged; ids 4..9 shift down by one
    assert id_map[:3].tolist() == [0, 1, 2]
    assert id_map[3].item() == -1
    assert id_map[4:].tolist() == [3, 4, 5, 6, 7, 8]


def test_eviction_empty_is_noop():
    torch.manual_seed(3)
    P = torch.randn(500, 3)
    S = torch.randn(15, 3)
    state = CorrectionState.from_cloud(P, S)
    before = state.S.clone()
    state.evict(P, torch.empty(0, dtype=torch.int64))
    assert torch.equal(state.S, before)


def test_cannot_evict_everything():
    P = torch.randn(100, 3)
    S = torch.randn(3, 3)
    state = CorrectionState.from_cloud(P, S)
    with pytest.raises(ValueError, match="every sample"):
        state.evict(P, torch.tensor([0, 1, 2]))
