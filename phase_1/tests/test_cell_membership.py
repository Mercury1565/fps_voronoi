import pytest
import torch
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from primitives import cell_membership

def test_m_zero_raises():
    P = torch.randn(5, 3)
    S = torch.zeros(0, 3)
    with pytest.raises(ValueError, match="M > 0"):
        cell_membership(P, S)


def test_m_one_all_go_to_cell_zero():
    P = torch.randn(10, 3)
    S = torch.randn(1, 3)
    ids, dists = cell_membership(P, S)
    assert ids.shape == (10,)
    assert dists.shape == (10,)
    assert (ids == 0).all()


def test_output_dtypes():
    P, S = torch.randn(8, 3), torch.randn(4, 3)
    ids, dists = cell_membership(P, S)
    assert ids.dtype == torch.int64
    assert dists.dtype == torch.float32


def test_output_device_matches_p():
    P, S = torch.randn(8, 3), torch.randn(4, 3)
    ids, dists = cell_membership(P, S)
    assert ids.device == P.device
    assert dists.device == P.device


def test_coincident_point_distance_zero():
    S = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    P = torch.tensor([[1.0, 2.0, 3.0]])   # coincident with S[0]
    ids, dists = cell_membership(P, S)
    assert ids[0].item() == 0
    assert dists[0].item() == pytest.approx(0.0, abs=1e-6)


def test_known_geometry():
    S = torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    P_near_0 = torch.tensor([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0]])
    P_near_1 = torch.tensor([[9.0, 0.0, 0.0], [9.5, 0.5, 0.0]])
    P = torch.cat([P_near_0, P_near_1])
    ids, dists = cell_membership(P, S)
    assert (ids[:2] == 0).all()
    assert (ids[2:] == 1).all()
    assert (dists >= 0).all()


def test_tie_break_lowest_index_wins():
    # Point at origin is equidistant from S[0]=(1,0,0) and S[1]=(-1,0,0)
    S = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    P = torch.tensor([[0.0, 0.0, 0.0]])
    ids, _ = cell_membership(P, S)
    assert ids[0].item() == 0


def test_distances_match_euclidean():
    torch.manual_seed(7)
    P, S = torch.randn(20, 3), torch.randn(5, 3)
    ids, dists = cell_membership(P, S)
    for i in range(len(P)):
        expected = torch.norm(S[ids[i]] - P[i]).item()
        assert dists[i].item() == pytest.approx(expected, abs=1e-5)


# ── Hypothesis property tests ──────────────────────────────────────────────────

@given(n=st.integers(1, 50), m=st.integers(1, 20))
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_cell_ids_in_range(n, m):
    P, S = torch.randn(n, 3), torch.randn(m, 3)
    ids, _ = cell_membership(P, S)
    assert (ids >= 0).all() and (ids < m).all()


@given(n=st.integers(1, 50), m=st.integers(1, 20))
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_distances_nonnegative(n, m):
    P, S = torch.randn(n, 3), torch.randn(m, 3)
    _, dists = cell_membership(P, S)
    assert (dists >= 0).all()


@given(n=st.integers(1, 50), m=st.integers(1, 20))
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_distance_equals_norm_to_assigned_sample(n, m):
    P, S = torch.randn(n, 3), torch.randn(m, 3)
    ids, dists = cell_membership(P, S)
    expected = torch.norm(S[ids] - P, dim=1)
    assert torch.allclose(dists, expected, atol=1e-5)


@given(m=st.integers(1, 20))
@settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
def test_first_sample_coincident_gets_id_zero(m):
    S = torch.randn(m, 3)
    P = S[0:1]                        # coincident with S[0]; lowest index always wins
    ids, dists = cell_membership(P, S)
    assert ids[0].item() == 0
    assert dists[0].item() == pytest.approx(0.0, abs=1e-5)


# ── Cross-backend test ─────────────────────────────────────────────────────────

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cpu_cuda_agreement():
    torch.manual_seed(42)
    P, S = torch.randn(200, 3), torch.randn(32, 3)
    ids_cpu, dists_cpu = cell_membership(P, S)
    ids_gpu, dists_gpu = cell_membership(P.cuda(), S.cuda())
    assert (ids_cpu == ids_gpu.cpu()).all()
    torch.testing.assert_close(dists_cpu, dists_gpu.cpu(), atol=1e-5, rtol=1e-5)
