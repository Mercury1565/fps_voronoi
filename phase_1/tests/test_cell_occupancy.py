import pytest
import torch
import numpy as np
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from primitives import cell_occupancy

def test_sum_equals_n():
    cell_ids = torch.randint(0, 5, (100,))
    counts = cell_occupancy(cell_ids, 5)
    assert counts.sum().item() == 100


def test_nonnegative():
    cell_ids = torch.randint(0, 10, (50,))
    counts = cell_occupancy(cell_ids, 10)
    assert (counts >= 0).all()


def test_output_dtype_int64():
    cell_ids = torch.tensor([0, 1, 0], dtype=torch.int64)
    counts = cell_occupancy(cell_ids, 2)
    assert counts.dtype == torch.int64


def test_output_shape():
    cell_ids = torch.tensor([0, 2], dtype=torch.int64)
    counts = cell_occupancy(cell_ids, 5)
    assert counts.shape == (5,)


def test_empty_cells_are_zero():
    cell_ids = torch.tensor([0, 0, 3], dtype=torch.int64)
    counts = cell_occupancy(cell_ids, 5)
    assert counts[0].item() == 2
    assert counts[1].item() == 0
    assert counts[2].item() == 0
    assert counts[3].item() == 1
    assert counts[4].item() == 0


def test_single_point():
    cell_ids = torch.tensor([2], dtype=torch.int64)
    counts = cell_occupancy(cell_ids, 4)
    assert counts[2].item() == 1
    assert counts.sum().item() == 1


def test_matches_numpy_bincount():
    torch.manual_seed(11)
    cell_ids = torch.randint(0, 8, (200,))
    counts = cell_occupancy(cell_ids, 8)
    expected = torch.from_numpy(
        np.bincount(cell_ids.numpy(), minlength=8)
    ).to(torch.int64)
    assert (counts == expected).all()


def test_all_same_cell():
    cell_ids = torch.zeros(10, dtype=torch.int64)
    counts = cell_occupancy(cell_ids, 3)
    assert counts[0].item() == 10
    assert counts[1].item() == 0
    assert counts[2].item() == 0


# ── Hypothesis property tests ──────────────────────────────────────────────────
@given(n=st.integers(1, 200), m=st.integers(1, 30))
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_sum_equals_n_property(n, m):
    cell_ids = torch.randint(0, m, (n,))
    counts = cell_occupancy(cell_ids, m)
    assert counts.sum().item() == n


@given(n=st.integers(1, 200), m=st.integers(1, 30))
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_nonnegative_property(n, m):
    cell_ids = torch.randint(0, m, (n,))
    counts = cell_occupancy(cell_ids, m)
    assert (counts >= 0).all()


@given(n=st.integers(1, 100), m=st.integers(1, 20))
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_matches_bincount_property(n, m):
    cell_ids = torch.randint(0, m, (n,))
    counts = cell_occupancy(cell_ids, m)
    expected = torch.from_numpy(
        np.bincount(cell_ids.numpy(), minlength=m)
    ).to(torch.int64)
    assert (counts == expected).all()


# ── Cross-backend test ─────────────────────────────────────────────────────────
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cpu_cuda_agreement():
    torch.manual_seed(0)
    cell_ids = torch.randint(0, 20, (500,))
    c_cpu = cell_occupancy(cell_ids, 20)
    c_gpu = cell_occupancy(cell_ids.cuda(), 20)
    assert (c_cpu == c_gpu.cpu()).all()
