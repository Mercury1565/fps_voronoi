import pytest
import torch
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from primitives import delaunay_neighbors

def test_output_length():
    S = torch.randn(20, 3)
    nbrs = delaunay_neighbors(S)
    assert len(nbrs) == 20


def test_symmetry():
    torch.manual_seed(0)
    S = torch.randn(20, 3)
    nbrs = delaunay_neighbors(S)
    for i, ns in enumerate(nbrs):
        for j in ns:
            assert i in nbrs[j], f"asymmetry: {j} lists {i} as neighbor but not vice versa"


def test_no_self_loops():
    torch.manual_seed(1)
    S = torch.randn(20, 3)
    nbrs = delaunay_neighbors(S)
    for i, ns in enumerate(nbrs):
        assert i not in ns, f"self-loop at index {i}"


def test_valid_indices():
    torch.manual_seed(2)
    S = torch.randn(15, 3)
    nbrs = delaunay_neighbors(S)
    M = S.shape[0]
    for ns in nbrs:
        for j in ns:
            assert 0 <= j < M


def test_sorted_neighbor_lists():
    torch.manual_seed(3)
    S = torch.randn(15, 3)
    nbrs = delaunay_neighbors(S)
    for ns in nbrs:
        assert ns == sorted(ns), "neighbor lists must be sorted"


def test_minimum_samples():
    # 4 non-coplanar points form one tetrahedron
    S = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    nbrs = delaunay_neighbors(S)
    assert len(nbrs) == 4
    # In a single tetrahedron every vertex is adjacent to every other
    for i in range(4):
        for j in range(4):
            if i != j:
                assert j in nbrs[i]


def test_larger_cloud_has_reasonable_degrees():
    torch.manual_seed(4)
    S = torch.randn(50, 3)
    nbrs = delaunay_neighbors(S)
    degrees = [len(ns) for ns in nbrs]
    # In a 3D Delaunay triangulation average degree is typically > 4
    assert sum(degrees) / len(degrees) > 4


# ── Hypothesis property tests ──────────────────────────────────────────────────

@given(m=st.integers(5, 30))
@settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
def test_symmetry_property(m):
    S = torch.randn(m, 3)
    try:
        nbrs = delaunay_neighbors(S)
    except ValueError:
        return   # degenerate configuration — skip
    for i, ns in enumerate(nbrs):
        for j in ns:
            assert i in nbrs[j]


@given(m=st.integers(5, 30))
@settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
def test_no_self_loops_property(m):
    S = torch.randn(m, 3)
    try:
        nbrs = delaunay_neighbors(S)
    except ValueError:
        return
    for i, ns in enumerate(nbrs):
        assert i not in ns


@given(m=st.integers(5, 30))
@settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
def test_valid_indices_property(m):
    S = torch.randn(m, 3)
    try:
        nbrs = delaunay_neighbors(S)
    except ValueError:
        return
    for ns in nbrs:
        for j in ns:
            assert 0 <= j < m
