"""
Phase 2 demo — visualise the three cell detectors on a 2-D LiDAR-like scene.

We deliberately build an *imperfect* sample set so every detector has something
to flag:

    * a random subsample of the cloud   → uneven coverage + a few close pairs
    * a handful of stray samples dropped → cells that capture (almost) no points
      in empty space                       (vanishing cells)

Run:
    python demo.py                 # synthetic mimic scene
    python demo.py frame.bin       # KITTI .bin (X,Y used)
    python demo.py --samples 80
"""

import argparse
import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phase_1"))

from demo import generate_mimic_2d, load_kitti_bin, _voronoi_finite_segments  # noqa: E402
from fused import analyze  # noqa: E402

NUM_SAMPLES = 64


def build_imperfect_samples(P: torch.Tensor, m: int, n_stray: int = 4, seed: int = 0):
    """Random subsample + a few stray samples in empty space.

    The random subsample alone is enough to produce coverage gaps and close
    pairs; the stray samples (placed just outside the cloud's bounding box)
    create cells that own essentially no points, exercising the vanishing-cell
    detector.
    """
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(P.shape[0], generator=g)[: m - n_stray]
    S = P[idx]

    lo = P.min(0).values
    hi = P.max(0).values
    center = (lo + hi) / 2
    half = (hi - lo) / 2
    # Drop stray samples well beyond the cloud extent (1.4x the half-extent past
    # each edge). Every real point is then closer to some interior sample, so
    # these cells own essentially nothing — the vanishing-cell case.
    offsets = torch.tensor([[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]])
    stray = center + offsets[:n_stray] * half * 1.4
    return torch.cat([S, stray.float()], dim=0)


def main():
    parser = argparse.ArgumentParser(description="Phase 2 detector visualisation (2-D)")
    parser.add_argument("bin_file", nargs="?", help="KITTI .bin file (X,Y used)")
    parser.add_argument("--samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--coverage-factor", type=float, default=1.6)
    parser.add_argument("--separation-factor", type=float, default=0.45)
    parser.add_argument("--min-occupancy", type=int, default=1)
    args = parser.parse_args()

    if args.bin_file:
        print(f"Loading LiDAR frame from {args.bin_file} …")
        P = load_kitti_bin(args.bin_file)
    else:
        print("No .bin file provided — generating mimic 2-D LiDAR scene.")
        P = generate_mimic_2d(N=10_000)
    print(f"  {P.shape[0]:,} points")

    M = args.samples
    S = build_imperfect_samples(P, M, seed=0)
    print(f"  {S.shape[0]} samples (imperfect, to trigger detectors)")

    stats, flags = analyze(
        P, S,
        coverage_factor=args.coverage_factor,
        separation_factor=args.separation_factor,
        min_occupancy=args.min_occupancy,
    )

    print(f"\n  coverage gaps    : {flags.coverage_gap_cells.numel()} cells")
    print(f"  separation viol. : {flags.separation_cells.numel()} samples "
          f"({flags.separation_pairs.shape[0]} pairs)")
    print(f"  vanishing cells  : {flags.vanishing_cells.numel()} cells")
    print(f"  min pairwise dist: {stats.min_pairwise.item():.4f}")

    # ── numpy views ──────────────────────────────────────────────────────────
    Pnp, Snp = P.numpy(), S.numpy()
    cell_np = stats.nearest_id.numpy()
    radii_np = stats.covering_radius.numpy()

    pad = 2.0
    xmin, xmax = Pnp[:, 0].min() - pad, Pnp[:, 0].max() + pad
    ymin, ymax = Pnp[:, 1].min() - pad, Pnp[:, 1].max() + pad
    clip_box = (xmin, xmax, ymin, ymax)

    from scipy.spatial import Voronoi
    vor = Voronoi(Snp)
    vor_segs = _voronoi_finite_segments(vor, clip_box)

    fig, axes = plt.subplots(2, 2, figsize=(15, 13))
    fig.suptitle(
        f"Phase 2 Detectors  |  N={P.shape[0]:,} pts · M={M} samples",
        fontsize=14, fontweight="bold",
    )

    def _base(ax, title, cells_color=True):
        if cells_color:
            ax.scatter(Pnp[:, 0], Pnp[:, 1], s=0.5, c=cell_np % 20, cmap="tab20",
                       alpha=0.35, linewidths=0, vmin=0, vmax=19)
        else:
            ax.scatter(Pnp[:, 0], Pnp[:, 1], s=0.4, c="lightgray",
                       alpha=0.3, linewidths=0)
        if vor_segs:
            ax.add_collection(LineCollection(vor_segs, colors="0.4",
                                             linewidths=0.5, zorder=3))
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.set_title(title)

    # ── Panel 1: cells + samples overview ─────────────────────────────────────
    ax = axes[0, 0]
    _base(ax, "Voronoi cells + samples")
    ax.scatter(Snp[:, 0], Snp[:, 1], s=35, c="red", zorder=6,
               edgecolors="white", linewidths=0.5, label="samples")
    ax.legend(fontsize=8, loc="upper right")

    # ── Panel 2: coverage gaps + insertion points ─────────────────────────────
    ax = axes[0, 1]
    _base(ax, f"Coverage gaps ({flags.coverage_gap_cells.numel()}) "
              "+ insertion points")
    cmap = plt.get_cmap("tab20")
    for c in flags.coverage_gap_cells.tolist():
        r = float(radii_np[c])
        circ = plt.Circle((Snp[c, 0], Snp[c, 1]), r, fill=True,
                          color="crimson", alpha=0.12, zorder=2)
        ax.add_patch(circ)
        ax.add_patch(plt.Circle((Snp[c, 0], Snp[c, 1]), r, fill=False,
                                edgecolor="crimson", linewidth=1.0,
                                alpha=0.8, zorder=3))
    ax.scatter(Snp[:, 0], Snp[:, 1], s=25, c="0.3", zorder=5)
    ins = flags.coverage_insertion_points.numpy()
    if ins.shape[0]:
        ax.scatter(ins[:, 0], ins[:, 1], s=120, marker="*", c="gold",
                   edgecolors="black", linewidths=0.6, zorder=7,
                   label="insertion points")
        ax.legend(fontsize=8, loc="upper right")

    # ── Panel 3: separation violators ─────────────────────────────────────────
    ax = axes[1, 0]
    _base(ax, f"Separation violators ({flags.separation_pairs.shape[0]} pairs)",
          cells_color=False)
    ax.scatter(Snp[:, 0], Snp[:, 1], s=30, c="0.4", zorder=5)
    viol = flags.separation_cells.numpy()
    pair_segs = [[Snp[i], Snp[j]] for i, j in flags.separation_pairs.tolist()]
    if pair_segs:
        ax.add_collection(LineCollection(pair_segs, colors="magenta",
                                         linewidths=2.0, zorder=6))
    if viol.size:
        ax.scatter(Snp[viol, 0], Snp[viol, 1], s=90, facecolors="none",
                   edgecolors="magenta", linewidths=1.8, zorder=7,
                   label="too close")
        ax.legend(fontsize=8, loc="upper right")

    # ── Panel 4: vanishing cells ──────────────────────────────────────────────
    ax = axes[1, 1]
    _base(ax, f"Vanishing cells ({flags.vanishing_cells.numel()})",
          cells_color=False)
    ax.scatter(Snp[:, 0], Snp[:, 1], s=30, c="0.4", zorder=5)
    van = flags.vanishing_cells.numpy()
    if van.size:
        ax.scatter(Snp[van, 0], Snp[van, 1], s=130, marker="X", c="blue",
                   edgecolors="white", linewidths=0.8, zorder=7,
                   label="empty / starved")
        # Vanishing samples often sit in empty space outside the cloud, so widen
        # the view to keep them visible.
        vx, vy = Snp[van, 0], Snp[van, 1]
        ax.set_xlim(min(xmin, vx.min() - pad), max(xmax, vx.max() + pad))
        ax.set_ylim(min(ymin, vy.min() - pad), max(ymax, vy.max() + pad))
        ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    out = "phase2_demo.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  Saved {out}")
    try:
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()
