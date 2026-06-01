"""Insertion: neighbour identification + exact one-hop membership update."""

import pytest
import torch

import primitives as p1
from fused import compute_stats
from correct import CorrectionState, insertion_neighbors


def _assert_state_matches_full(state, P):
    """The incrementally maintained fields must equal a full recompute on S.

    Membership is checked against Phase 2's ``compute_stats``; distances/radii
    are checked against accurate direct norms consistent with that membership
    (``compute_stats`` itself carries ~1e-3 ``cdist`` noise near zero distance).
    """
    ref = compute_stats(P, state.S)
    assert (state.nearest_id == ref.nearest_id).all()
    acc_dist = (P - state.S[state.nearest_id]).norm(dim=1)
    torch.testing.assert_close(state.nearest_dist, acc_dist, atol=1e-5, rtol=1e-5)
    assert (state.occupancy == ref.occupancy).all()
    acc_rad = torch.zeros(state.num_samples)
    acc_rad.scatter_reduce_(0, state.nearest_id, acc_dist, reduce="amax", include_self=True)
    torch.testing.assert_close(state.covering_radius, acc_rad, atol=1e-5, rtol=1e-5)


# ── insertion_neighbors ──────────────────────────────────────────────────────
def test_insertion_neighbors_match_full_delaunay():
    torch.manual_seed(0)
    S = torch.randn(40, 3)
    p = torch.randn(3)
    got = insertion_neighbors(S, p)
    # reference: neighbours of the new site in Delaunay(S ∪ {p})
    aug = torch.cat([S, p.view(1, 3)], 0)
    ref = sorted(j for j in p1.delaunay_neighbors(aug)[40] if j < 40)
    assert got.tolist() == ref


def test_insertion_neighbors_include_nearest_sample():
    torch.manual_seed(1)
    S = torch.randn(30, 3)
    p = torch.randn(3)
    nbrs = set(insertion_neighbors(S, p).tolist())
    c0 = torch.cdist(p.view(1, 3), S).argmin().item()
    assert c0 in nbrs  # the cell p lands in always borders the new cell


def test_insertion_neighbors_degenerate_fallback():
    S = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])  # collinear, Qhull fails
    nbrs = insertion_neighbors(S, torch.tensor([0.5, 0.0, 0.0]))
    assert nbrs.tolist() == [0, 1]  # safe over-approximation


# ── one-hop insertion equals full recompute ──────────────────────────────────
def test_single_insertion_exact():
    torch.manual_seed(2)
    P = torch.randn(2000, 3)
    S = torch.randn(40, 3)
    state = CorrectionState.from_cloud(P, S)
    new_id = state.insert(P, torch.randn(3))
    assert new_id == 40
    assert state.num_samples == 41
    _assert_state_matches_full(state, P)


def test_insertion_of_cloud_point_steals_it():
    torch.manual_seed(3)
    P = torch.randn(1500, 3)
    S = torch.randn(20, 3)
    state = CorrectionState.from_cloud(P, S)
    # Insert an actual cloud point — it must end up owning at least itself.
    target = P[123]
    new_id = state.insert(P, target)
    assert state.nearest_id[123].item() == new_id
    assert state.occupancy[new_id].item() >= 1
    _assert_state_matches_full(state, P)


def test_many_sequential_insertions_exact():
    torch.manual_seed(4)
    P = torch.randn(3000, 3)
    S = torch.randn(30, 3)
    state = CorrectionState.from_cloud(P, S)
    pts = P[torch.randperm(3000)[:15]]
    for k in range(pts.shape[0]):
        state.insert(P, pts[k])
    assert state.num_samples == 45
    _assert_state_matches_full(state, P)


def test_insertion_in_2d():
    torch.manual_seed(5)
    P = torch.randn(1000, 2)
    S = torch.randn(25, 2)
    state = CorrectionState.from_cloud(P, S)
    state.insert(P, torch.randn(2))
    _assert_state_matches_full(state, P)
