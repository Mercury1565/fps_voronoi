import pytest
import torch
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from primitives import covering_radius

def test_empty_cell_returns_zero():
    cell_ids = torch.tensor([0, 0, 2], dtype=torch.int64)
    dists = torch.tensor([1.0, 2.0, 3.0])
    radii = covering_radius(cell_ids, dists, num_samples=3)
    assert radii[1].item() == pytest.approx(0.0)


def test_single_point_cell():
    cell_ids = torch.tensor([0], dtype=torch.int64)
    dists = torch.tensor([3.14])
    radii = covering_radius(cell_ids, dists, num_samples=1)
    assert radii[0].item() == pytest.approx(3.14, abs=1e-5)


def test_max_per_cell():
    cell_ids = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    dists = torch.tensor([1.0, 5.0, 2.0, 3.0])
    radii = covering_radius(cell_ids, dists, num_samples=2)
    assert radii[0].item() == pytest.approx(5.0)
    assert radii[1].item() == pytest.approx(3.0)


def test_output_dtype():
    cell_ids = torch.tensor([0, 1], dtype=torch.int64)
    dists = torch.tensor([1.0, 2.0])
    radii = covering_radius(cell_ids, dists, num_samples=2)
    assert radii.dtype == torch.float32


def test_output_shape():
    radii = covering_radius(
        torch.tensor([0, 1, 2], dtype=torch.int64),
        torch.tensor([1.0, 2.0, 3.0]),
        num_samples=5,
    )
    assert radii.shape == (5,)


def test_no_points_all_zeros():
    radii = covering_radius(
        torch.tensor([], dtype=torch.int64),
        torch.tensor([], dtype=torch.float32),
        num_samples=4,
    )
    assert radii.shape == (4,)
    assert (radii == 0).all()


def test_zero_distance_cell():
    cell_ids = torch.tensor([0, 0], dtype=torch.int64)
    dists = torch.tensor([0.0, 0.0])
    radii = covering_radius(cell_ids, dists, num_samples=1)
    assert radii[0].item() == pytest.approx(0.0)


# ── Hypothesis property tests ──────────────────────────────────────────────────

@given(n=st.integers(1, 100), m=st.integers(1, 20))
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_radii_geq_all_member_distances(n, m):
    cell_ids = torch.randint(0, m, (n,))
    dists = torch.rand(n)
    radii = covering_radius(cell_ids, dists, num_samples=m)
    assert (radii >= 0).all()
    for c in range(m):
        mask = cell_ids == c
        if mask.any():
            assert radii[c].item() >= dists[mask].max().item() - 1e-6


@given(n=st.integers(1, 100), m=st.integers(1, 20))
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_empty_cells_stay_zero(n, m):
    cell_ids = torch.randint(0, m, (n,))
    dists = torch.rand(n)
    radii = covering_radius(cell_ids, dists, num_samples=m)
    for c in range(m):
        if not (cell_ids == c).any():
            assert radii[c].item() == 0.0


# ── Cross-backend test ─────────────────────────────────────────────────────────

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cpu_cuda_agreement():
    torch.manual_seed(0)
    cell_ids = torch.randint(0, 32, (1000,))
    dists = torch.rand(1000)
    r_cpu = covering_radius(cell_ids, dists, 32)
    r_gpu = covering_radius(cell_ids.cuda(), dists.cuda(), 32)
    torch.testing.assert_close(r_cpu, r_gpu.cpu(), atol=1e-5, rtol=1e-5)
