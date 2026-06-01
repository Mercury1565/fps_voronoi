"""Budget tracker + end-to-end correct() frame, incl. FPS fallback."""

import pytest
import torch

from fused import compute_stats
from correct import BudgetTracker, correct, CorrectionState


# ── BudgetTracker ────────────────────────────────────────────────────────────
def test_budget_not_exceeded_within_limit():
    t = BudgetTracker(5).record(3)
    assert not t.exceeded
    assert t.remaining == 2


def test_budget_exceeded_over_limit():
    t = BudgetTracker(5).record(6)
    assert t.exceeded
    assert t.remaining == 0


def test_budget_reset():
    t = BudgetTracker(2).record(5)
    assert t.exceeded
    t.reset()
    assert not t.exceeded and t.edits == 0


def test_budget_accumulates():
    t = BudgetTracker(10)
    t.record(3).record(4)
    assert t.edits == 7 and not t.exceeded


# ── correct(): incremental path ──────────────────────────────────────────────
def test_correct_within_budget_is_incremental_and_consistent():
    torch.manual_seed(0)
    P = torch.randn(3000, 3)
    S = torch.randn(50, 3)
    res = correct(P, S, budget=1000, coverage_factor=1.5, min_occupancy=1)
    assert not res.fallback
    # the returned stats must be self-consistent with a full recompute on the
    # corrected S (this validates the whole insert+evict one-hop pipeline)
    ref = compute_stats(P, res.S)
    assert (res.stats.nearest_id == ref.nearest_id).all()
    assert (res.stats.occupancy == ref.occupancy).all()
    acc_dist = (P - res.S[res.stats.nearest_id]).norm(dim=1)
    acc_rad = torch.zeros(res.S.shape[0])
    acc_rad.scatter_reduce_(0, res.stats.nearest_id, acc_dist, reduce="amax", include_self=True)
    torch.testing.assert_close(res.stats.covering_radius, acc_rad, atol=1e-5, rtol=1e-5)


def test_correct_sample_count_changes_by_net_edits():
    torch.manual_seed(1)
    P = torch.randn(2500, 2)
    S = torch.randn(40, 2)
    res = correct(P, S, budget=1000, coverage_factor=1.4)
    expected = 40 + res.n_inserted - res.n_evicted
    assert res.S.shape[0] == expected


# ── correct(): budget fallback ───────────────────────────────────────────────
def test_correct_fallback_when_budget_exceeded():
    torch.manual_seed(2)
    P = torch.randn(2000, 3)
    # random subsample → many flags → many requested edits
    S = P[torch.randperm(2000)[:60]].clone()
    res = correct(P, S, budget=0, fps_seed=7, coverage_factor=1.2,
                  separation_factor=0.8)
    assert res.fallback
    assert res.n_inserted == 0 and res.n_evicted == 0
    assert res.n_requested > 0
    # fallback keeps the sample count and returns a valid full sampling
    assert res.S.shape[0] == 60
    ref = compute_stats(P, res.S)
    assert (res.stats.nearest_id == ref.nearest_id).all()


def test_fallback_is_deterministic_with_seed():
    torch.manual_seed(3)
    P = torch.randn(1500, 3)
    S = P[torch.randperm(1500)[:50]].clone()
    a = correct(P, S, budget=0, fps_seed=42, coverage_factor=1.2)
    b = correct(P, S, budget=0, fps_seed=42, coverage_factor=1.2)
    assert a.fallback and b.fallback
    assert torch.equal(a.S, b.S)


def test_healthy_sampling_needs_no_fallback():
    # An FPS-like, well-spread sampling under default thresholds should fit in a
    # generous budget without falling back.
    torch.manual_seed(4)
    from fps import farthest_point_sampling
    P = torch.randn(4000, 3)
    S = P[farthest_point_sampling(P, 64, seed=0)]
    res = correct(P, S, budget=64)
    assert not res.fallback
