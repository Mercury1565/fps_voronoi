import pytest
import torch
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from primitives import min_pairwise_distance

def test_m_one_raises():
    with pytest.raises(ValueError, match="M >= 2"):
        min_pairwise_distance(torch.randn(1, 3))


def test_m_two_trivial():
    S = torch.tensor([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
    dist, pair = min_pairwise_distance(S)
    assert dist.item() == pytest.approx(5.0, abs=1e-5)
    assert set(pair.tolist()) == {0, 1}


def test_canonical_pair_order():
    S = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    _, pair = min_pairwise_distance(S)
    assert pair[0].item() < pair[1].item()


def test_returned_pair_achieves_minimum():
    torch.manual_seed(3)
    S = torch.randn(15, 3)
    dist, pair = min_pairwise_distance(S)
    i, j = pair[0].item(), pair[1].item()
    assert dist.item() == pytest.approx(torch.norm(S[i] - S[j]).item(), abs=1e-5)


def test_pair_indices_distinct():
    torch.manual_seed(5)
    S = torch.randn(10, 3)
    _, pair = min_pairwise_distance(S)
    assert pair[0].item() != pair[1].item()


def test_output_dtypes():
    S = torch.randn(5, 3)
    dist, pair = min_pairwise_distance(S)
    assert dist.dtype == torch.float32
    assert pair.dtype == torch.int64


def test_result_is_global_min():
    torch.manual_seed(9)
    S = torch.randn(8, 3)
    dist, _ = min_pairwise_distance(S)
    # Brute-force check
    M = S.shape[0]
    bf = float("inf")
    for i in range(M):
        for j in range(i + 1, M):
            bf = min(bf, torch.norm(S[i] - S[j]).item())
    assert dist.item() == pytest.approx(bf, abs=1e-5)


# ── Hypothesis property tests ──────────────────────────────────────────────────

@given(m=st.integers(2, 25))
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_dist_equals_brute_force(m):
    S = torch.randn(m, 3)
    dist, pair = min_pairwise_distance(S)
    M = S.shape[0]
    bf = float("inf")
    for i in range(M):
        for j in range(i + 1, M):
            bf = min(bf, torch.norm(S[i] - S[j]).item())
    assert dist.item() == pytest.approx(bf, abs=1e-5)


@given(m=st.integers(2, 25))
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_pair_valid_and_canonical(m):
    S = torch.randn(m, 3)
    _, pair = min_pairwise_distance(S)
    i, j = pair[0].item(), pair[1].item()
    assert 0 <= i < m
    assert 0 <= j < m
    assert i < j


@given(m=st.integers(2, 25))
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_distance_nonnegative(m):
    S = torch.randn(m, 3)
    dist, _ = min_pairwise_distance(S)
    assert dist.item() >= 0.0


# ── Cross-backend test ─────────────────────────────────────────────────────────

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cpu_cuda_agreement():
    torch.manual_seed(42)
    S = torch.randn(64, 3)
    d_cpu, p_cpu = min_pairwise_distance(S)
    d_gpu, p_gpu = min_pairwise_distance(S.cuda())
    assert torch.abs(d_cpu - d_gpu.cpu()).item() < 1e-5
    assert (p_cpu == p_gpu.cpu()).all()
