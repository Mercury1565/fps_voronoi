"""Threshold logic: coverage gap, separation violator, vanishing cell."""

import pytest
import torch

from fused import compute_stats, detect_flags, analyze


# ── Coverage gap detector ────────────────────────────────────────────────────
def test_coverage_gap_flags_large_cell_with_absolute_threshold():
    # Two tight clusters around S[0] and S[1]; one far outlier joins S[2],
    # blowing up cell 2's covering radius.
    S = torch.tensor([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    P = torch.tensor([
        [0.0, 0.0], [0.1, 0.0],          # cell 0, tiny radius
        [10.0, 0.0], [10.1, 0.0],        # cell 1, tiny radius
        [20.0, 0.0], [20.0, 8.0],        # cell 2, radius 8 (the gap)
    ])
    stats = compute_stats(P, S)
    flags = detect_flags(P, stats, coverage_radius_max=2.0)
    assert flags.coverage_gap_cells.tolist() == [2]
    # insertion point is the farthest member: (20, 8)
    torch.testing.assert_close(
        flags.coverage_insertion_points[0],
        torch.tensor([20.0, 8.0]),
        atol=1e-5, rtol=1e-5,
    )


def test_coverage_insertion_point_count_matches_flags():
    torch.manual_seed(0)
    P, S = torch.randn(2000, 2), torch.randn(50, 2)
    stats = compute_stats(P, S)
    flags = detect_flags(P, stats, coverage_factor=1.5)
    assert flags.coverage_insertion_points.shape[0] == flags.coverage_gap_cells.shape[0]
    assert flags.coverage_insertion_points.shape[1] == 2


def test_coverage_insertion_points_lie_on_flagged_cells():
    torch.manual_seed(1)
    P, S = torch.randn(1500, 3), torch.randn(40, 3)
    stats = compute_stats(P, S)
    flags = detect_flags(P, stats, coverage_factor=1.2)
    for k, c in enumerate(flags.coverage_gap_cells.tolist()):
        pt = flags.coverage_insertion_points[k]
        # the proposed point must actually belong to the flagged cell
        d = torch.cdist(pt.unsqueeze(0), S).squeeze(0)
        assert d.argmin().item() == c


def test_coverage_no_gaps_when_threshold_high():
    torch.manual_seed(2)
    P, S = torch.randn(1000, 2), torch.randn(30, 2)
    stats = compute_stats(P, S)
    flags = detect_flags(P, stats, coverage_radius_max=1e6)
    assert flags.coverage_gap_cells.numel() == 0
    assert flags.coverage_insertion_points.numel() == 0


# ── Separation violator detector ─────────────────────────────────────────────
def test_separation_flags_close_pair():
    # S[0] and S[1] are 0.1 apart; the rest are far away.
    S = torch.tensor([[0.0, 0.0], [0.1, 0.0], [10.0, 0.0], [20.0, 0.0]])
    P = torch.randn(50, 2)
    stats = compute_stats(P, S)
    flags = detect_flags(P, stats, separation_min=1.0)
    assert set(flags.separation_cells.tolist()) == {0, 1}
    assert flags.separation_pairs.tolist() == [[0, 1]]


def test_separation_pairs_are_canonical_and_unique():
    S = torch.tensor([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]])
    P = torch.randn(20, 2)
    stats = compute_stats(P, S)
    flags = detect_flags(P, stats, separation_min=1.0)
    for pair in flags.separation_pairs.tolist():
        assert pair[0] < pair[1]
    # no duplicate rows
    assert len(flags.separation_pairs.tolist()) == len(
        {tuple(r) for r in flags.separation_pairs.tolist()}
    )


def test_separation_none_when_threshold_tiny():
    torch.manual_seed(3)
    S = torch.randn(40, 2) * 100  # well separated
    P = torch.randn(200, 2) * 100
    stats = compute_stats(P, S)
    flags = detect_flags(P, stats, separation_min=1e-6)
    assert flags.separation_cells.numel() == 0
    assert flags.separation_pairs.numel() == 0


def test_separation_empty_when_single_sample():
    P, S = torch.randn(10, 2), torch.randn(1, 2)
    stats = compute_stats(P, S)
    flags = detect_flags(P, stats)
    assert flags.separation_cells.numel() == 0
    assert flags.separation_pairs.shape == (0, 2)


# ── Vanishing cell detector ──────────────────────────────────────────────────
def test_vanishing_flags_empty_cells():
    S = torch.tensor([[0.0, 0.0], [100.0, 0.0], [200.0, 0.0]])
    P = torch.tensor([[0.0, 0.0], [0.1, 0.0]])  # everything in cell 0
    stats = compute_stats(P, S)
    flags = detect_flags(P, stats, min_occupancy=1)
    assert set(flags.vanishing_cells.tolist()) == {1, 2}


def test_vanishing_threshold():
    S = torch.tensor([[0.0, 0.0], [10.0, 0.0]])
    # cell 0 gets 3 points, cell 1 gets 1
    P = torch.tensor([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [10.0, 0.0]])
    stats = compute_stats(P, S)
    flags = detect_flags(P, stats, min_occupancy=2)
    assert flags.vanishing_cells.tolist() == [1]


def test_vanishing_none_when_all_populated():
    torch.manual_seed(4)
    # few samples, many points -> every cell populated
    P, S = torch.randn(5000, 2), torch.randn(8, 2)
    stats = compute_stats(P, S)
    flags = detect_flags(P, stats, min_occupancy=1)
    assert flags.vanishing_cells.numel() == 0


# ── analyze() orchestrator ───────────────────────────────────────────────────
def test_analyze_returns_consistent_stats_and_flags():
    torch.manual_seed(5)
    P, S = torch.randn(3000, 3), torch.randn(60, 3)
    stats, flags = analyze(P, S, coverage_factor=1.5, separation_factor=0.6)
    # stats internally consistent
    assert stats.occupancy.sum().item() == 3000
    assert stats.nearest_id.shape == (3000,)
    assert stats.covering_radius.shape == (60,)
    # flags reference valid cell ids
    assert (flags.coverage_gap_cells < 60).all()
    assert (flags.vanishing_cells < 60).all()
    if flags.separation_cells.numel():
        assert (flags.separation_cells < 60).all()


def test_analyze_default_thresholds_run():
    torch.manual_seed(6)
    P, S = torch.randn(1000, 2), torch.randn(20, 2)
    # Should not raise with all-default (median-derived) thresholds.
    stats, flags = analyze(P, S)
    assert isinstance(flags.coverage_gap_cells, torch.Tensor)
