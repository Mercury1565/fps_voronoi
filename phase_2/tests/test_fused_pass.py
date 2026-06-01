"""The fused pass must reproduce, exactly, what the Phase 1 standalone
primitives compute separately — that's the whole correctness contract."""

import pytest
import torch
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

import primitives as p1
from fused import fused_knn_pass, build_csr, sample_neighbor_stats, compute_stats


# ── Golden cross-checks against Phase 1 ──────────────────────────────────────
def test_membership_matches_phase1():
    torch.manual_seed(0)
    P, S = torch.randn(500, 3), torch.randn(32, 3)
    nid, ndist, *_ = fused_knn_pass(P, S)
    g_id, g_dist = p1.cell_membership(P, S)
    assert (nid == g_id).all()
    torch.testing.assert_close(ndist, g_dist, atol=1e-5, rtol=1e-5)


def test_covering_radius_matches_phase1():
    torch.manual_seed(1)
    P, S = torch.randn(800, 3), torch.randn(40, 3)
    nid, ndist, radius, *_ = fused_knn_pass(P, S)
    g = p1.covering_radius(nid, ndist, 40)
    torch.testing.assert_close(radius, g, atol=1e-5, rtol=1e-5)


def test_occupancy_matches_phase1():
    torch.manual_seed(2)
    P, S = torch.randn(800, 3), torch.randn(40, 3)
    nid, _, _, occ, _ = fused_knn_pass(P, S)
    g = p1.cell_occupancy(nid, 40)
    assert (occ == g).all()


def test_min_pairwise_matches_phase1():
    torch.manual_seed(3)
    S = torch.randn(64, 3)
    _, _, mp, pair = sample_neighbor_stats(S)
    g_dist, g_pair = p1.min_pairwise_distance(S)
    assert mp.item() == pytest.approx(g_dist.item(), abs=1e-5)
    assert (pair == g_pair).all()


@given(n=st.integers(1, 300), m=st.integers(1, 30))
@settings(max_examples=40, suppress_health_check=[HealthCheck.too_slow])
def test_fused_equals_phase1_property(n, m):
    P, S = torch.randn(n, 3), torch.randn(m, 3)
    nid, ndist, radius, occ, _ = fused_knn_pass(P, S)
    torch.testing.assert_close(ndist, p1.cell_membership(P, S)[1], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(radius, p1.covering_radius(nid, ndist, m), atol=1e-5, rtol=1e-5)
    assert (occ == p1.cell_occupancy(nid, m)).all()


# ── Chunking must not change the answer ──────────────────────────────────────
def test_chunk_size_invariant():
    torch.manual_seed(4)
    P, S = torch.randn(1000, 3), torch.randn(16, 3)
    a = fused_knn_pass(P, S, chunk=64)
    b = fused_knn_pass(P, S, chunk=4096)
    for x, y in zip(a, b):
        if x.dtype.is_floating_point:
            torch.testing.assert_close(x, y, atol=1e-5, rtol=1e-5)
        else:
            assert (x == y).all()


# ── farthest_idx semantics ───────────────────────────────────────────────────
def test_farthest_idx_points_to_max_member():
    torch.manual_seed(5)
    P, S = torch.randn(600, 3), torch.randn(20, 3)
    nid, ndist, radius, occ, far = fused_knn_pass(P, S)
    for c in range(20):
        if occ[c] == 0:
            assert far[c].item() == -1
        else:
            fi = far[c].item()
            assert nid[fi].item() == c
            # the chosen member achieves the cell's covering radius
            assert ndist[fi].item() == pytest.approx(radius[c].item(), abs=1e-5)


def test_farthest_idx_empty_cells_are_minus_one():
    S = torch.tensor([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [200.0, 0.0, 0.0]])
    P = torch.tensor([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]])  # all near S[0]
    _, _, _, occ, far = fused_knn_pass(P, S)
    assert occ[1].item() == 0 and far[1].item() == -1
    assert occ[2].item() == 0 and far[2].item() == -1
    assert far[0].item() in (0, 1)


def test_m_zero_raises():
    with pytest.raises(ValueError, match="M > 0"):
        fused_knn_pass(torch.randn(5, 3), torch.zeros(0, 3))


# ── CSR layout ───────────────────────────────────────────────────────────────
def test_csr_partitions_all_points():
    torch.manual_seed(6)
    P, S = torch.randn(700, 3), torch.randn(25, 3)
    nid, _, _, occ, _ = fused_knn_pass(P, S)
    order, offsets = build_csr(nid, occ)
    assert offsets[-1].item() == 700
    assert offsets[0].item() == 0
    # every point appears exactly once
    assert torch.equal(torch.sort(order).values, torch.arange(700))
    # each slice contains exactly the points of that cell
    for c in range(25):
        members = order[offsets[c]:offsets[c + 1]]
        assert (nid[members] == c).all()
        assert members.numel() == occ[c].item()


def test_csr_cell_points_helper():
    torch.manual_seed(7)
    P, S = torch.randn(300, 3), torch.randn(10, 3)
    stats = compute_stats(P, S)
    for c in range(10):
        members = stats.cell_points(c)
        assert (stats.nearest_id[members] == c).all()


def test_csr_empty_cloud():
    P = torch.empty(0, 3)
    S = torch.randn(4, 3)
    nid, _, _, occ, far = fused_knn_pass(P, S)
    order, offsets = build_csr(nid, occ)
    assert order.numel() == 0
    assert offsets[-1].item() == 0
    assert (far == -1).all()


# ── sample_neighbor_stats edge cases ─────────────────────────────────────────
def test_sample_nn_is_symmetric_distance():
    torch.manual_seed(8)
    S = torch.randn(30, 3)
    nn_dist, nn_idx, _, _ = sample_neighbor_stats(S)
    for i in range(30):
        j = nn_idx[i].item()
        assert nn_dist[i].item() == pytest.approx(torch.norm(S[i] - S[j]).item(), abs=1e-5)
        assert i != j


def test_sample_nn_single_sample():
    nn_dist, nn_idx, mp, pair = sample_neighbor_stats(torch.randn(1, 3))
    assert torch.isinf(nn_dist).all()
    assert (nn_idx == -1).all()
    assert torch.isinf(mp)
    assert (pair == torch.tensor([-1, -1])).all()


def test_sample_nn_chunk_invariant():
    torch.manual_seed(9)
    S = torch.randn(200, 3)
    a = sample_neighbor_stats(S, chunk=16)
    b = sample_neighbor_stats(S, chunk=4096)
    torch.testing.assert_close(a[0], b[0], atol=1e-5, rtol=1e-5)
    assert (a[1] == b[1]).all()
