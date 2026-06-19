"""Invariants for the warm-FPS resampler."""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from resample import _cell_stats, fps_continue, valid_mask, warm_resample  # noqa: E402


def _grid_cloud(n=40):
    """A dense square grid in 2-D — uniform coverage, easy to reason about."""
    xs = torch.linspace(0, 1, n)
    g = torch.stack(torch.meshgrid(xs, xs, indexing="ij"), dim=-1).reshape(-1, 2)
    return g.contiguous()


def test_fps_continue_cold_returns_requested_count():
    P = _grid_cloud()
    S = fps_continue(P, None, 16, seed=0)
    assert S.shape == (16, 2)


def test_fps_continue_preserves_seeds_in_order():
    P = _grid_cloud()
    seeds = P[:5].clone()
    S = fps_continue(P, seeds, 16, seed=0)
    assert S.shape == (16, 2)
    # the kept seeds occupy the first rows, unchanged
    assert torch.allclose(S[:5], seeds)


def test_fps_continue_noop_when_already_full():
    P = _grid_cloud()
    seeds = P[:16].clone()
    S = fps_continue(P, seeds, 16)
    assert torch.allclose(S, seeds)


def test_valid_mask_drops_vanished_and_stale():
    P = _grid_cloud()                       # cloud lives in [0,1]^2
    # sample 0 sits on the cloud; sample 1 floats far away (stale + empty cell)
    S = torch.tensor([[0.5, 0.5], [50.0, 50.0]])
    keep = valid_mask(P, S, min_occupancy=1, stale_dist=1.0)
    assert bool(keep[0]) is True
    assert bool(keep[1]) is False           # vanished (occ 0) and stale (faith huge)
    # the underlying signals still explain why sample 1 was dropped
    occ, faith = _cell_stats(P, S)
    assert int(occ[1]) == 0
    assert float(faith[1]) > 1.0


def test_valid_mask_drops_redundant_pair():
    P = _grid_cloud()
    # two samples almost on top of each other -> one is redundant
    S = torch.tensor([[0.5, 0.5], [0.5001, 0.5001], [0.1, 0.9]])
    keep = valid_mask(P, S, min_occupancy=1, separation_min=0.05)
    assert int(keep.sum()) == 2             # exactly one of the close pair dropped


def test_warm_resample_holds_count_and_reports_split():
    P = _grid_cloud()
    S_prev = fps_continue(P, None, 16, seed=1)
    # corrupt two samples into empty space so they get dropped + refilled
    S_prev = S_prev.clone()
    S_prev[0] = torch.tensor([99.0, 99.0])
    S_prev[1] = torch.tensor([-99.0, -99.0])
    res = warm_resample(P, S_prev, 16, min_occupancy=1, stale_dist=1.0)
    assert res.S.shape == (16, 2)           # constant M
    assert res.n_dropped >= 2
    assert res.n_kept + res.n_refilled == 16
