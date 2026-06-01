import argparse
import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fps import farthest_point_sampling
from primitives import (
    cell_membership,
    cell_occupancy,
    covering_radius,
    delaunay_neighbors,
    min_pairwise_distance,
)

NUM_SAMPLES = 64


def generate_mimic_2d(N: int = 10_000, seed: int = 42) -> torch.Tensor:
    """
    Synthetic 2D point cloud mimicking a top-down LiDAR scene:
    """
    rng = np.random.default_rng(seed)
    pts = []

    # Concentric scan rings
    for r in np.linspace(5, 45, 10):
        n = max(40, int(N * 0.04))
        theta = rng.uniform(0, 2 * np.pi, n)
        noise = rng.normal(0, 0.4, n)
        x = (r + noise) * np.cos(theta)
        y = (r + noise) * np.sin(theta)
        pts.append(np.stack([x, y], axis=1))

    # Dense obstacle clusters
    for cx, cy, spread, frac in [
        (15,  10,  2.0, 0.15),
        (-20, -5,  3.0, 0.10),
        (5,  -25,  1.5, 0.08),
    ]:
        count = int(N * frac)
        x = rng.normal(cx, spread, count)
        y = rng.normal(cy, spread, count)
        pts.append(np.stack([x, y], axis=1))

    # Linear walls
    for x0, y0, x1, y1 in [(-30, 20, 30, 20), (-10, -10, -10, 30)]:
        n = int(N * 0.05)
        t = rng.uniform(0, 1, n)
        x = x0 + t * (x1 - x0) + rng.normal(0, 0.25, n)
        y = y0 + t * (y1 - y0) + rng.normal(0, 0.25, n)
        pts.append(np.stack([x, y], axis=1))

    pts = np.vstack(pts)
    mask = (np.abs(pts[:, 0]) < 50) & (np.abs(pts[:, 1]) < 50)
    pts = pts[mask]
    idx = rng.choice(len(pts), min(N, len(pts)), replace=False)
    return torch.from_numpy(pts[idx].astype(np.float32))


def load_kitti_bin(path: str) -> torch.Tensor:
    """Load a KITTI .bin file and return only the X,Y columns (2D top-down)."""
    pts = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    return torch.from_numpy(pts[:, :2])


def print_section(title: str) -> None:
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print("─" * 50)


def _voronoi_finite_segments(vor, clip_box):
    """Return finite Voronoi ridge segments clipped to clip_box."""
    xmin, xmax, ymin, ymax = clip_box
    segs = []
    for ridge in vor.ridge_vertices:
        if -1 in ridge:
            continue
        p1, p2 = vor.vertices[ridge[0]], vor.vertices[ridge[1]]
        if (xmin <= p1[0] <= xmax and ymin <= p1[1] <= ymax
                and xmin <= p2[0] <= xmax and ymin <= p2[1] <= ymax):
            segs.append([p1, p2])
    return segs


def main():
    parser = argparse.ArgumentParser(description="Voronoi primitives demo (2-D)")
    parser.add_argument("bin_file", nargs="?", help="KITTI .bin file (X,Y used)")
    parser.add_argument("--samples", type=int, default=NUM_SAMPLES,
                        help="Number of FPS samples (default: 64)")
    args = parser.parse_args()

    if args.bin_file:
        print(f"Loading LiDAR frame from {args.bin_file} …")
        P = load_kitti_bin(args.bin_file)
        print(f"  Loaded {P.shape[0]:,} points (X,Y projection)")
    else:
        print("No .bin file provided — generating mimic 2-D LiDAR scene (N≈10,000).")
        P = generate_mimic_2d(N=10_000)
        print(f"  Generated {P.shape[0]:,} points")

    M = args.samples

    # ── FPS ─────────────────────────────────────────────────────────────────────
    print_section(f"FPS: selecting M={M} samples")
    fps_idx = farthest_point_sampling(P, M, seed=42)
    S = P[fps_idx]
    print(f"  Done. S shape: {S.shape}")

    # ── 1. cell_membership ──────────────────────────────────────────────────────
    print_section("1. cell_membership")
    cell_ids, distances = cell_membership(P, S)
    print(f"  cell_ids  : shape={cell_ids.shape}, dtype={cell_ids.dtype}")
    print(f"  distances : min={distances.min():.4f}  max={distances.max():.4f}"
          f"  mean={distances.mean():.4f}")

    # ── 2. covering_radius ──────────────────────────────────────────────────────
    print_section("2. covering_radius")
    radii = covering_radius(cell_ids, distances, M)
    print(f"  radii     : max={radii.max():.4f}  mean={radii.mean():.4f}"
          f"  min(non-zero)={radii[radii > 0].min():.4f}")

    # ── 3. min_pairwise_distance ────────────────────────────────────────────────
    print_section("3. min_pairwise_distance")
    min_dist, pair = min_pairwise_distance(S)
    i, j = pair[0].item(), pair[1].item()
    print(f"  min distance : {min_dist:.6f}  between samples {i} and {j}")
    print(f"  S[{i}] = {S[i].tolist()}")
    print(f"  S[{j}] = {S[j].tolist()}")

    # ── 4. cell_occupancy ───────────────────────────────────────────────────────
    print_section("4. cell_occupancy")
    counts = cell_occupancy(cell_ids, M)
    empty = (counts == 0).sum().item()
    print(f"  counts    : min={counts.min().item()}  max={counts.max().item()}"
          f"  mean={counts.float().mean():.1f}")
    print(f"  empty cells : {empty} / {M}")

    # ── 5. delaunay_neighbors ───────────────────────────────────────────────────
    print_section("5. delaunay_neighbors")
    neighbors = delaunay_neighbors(S)
    degrees = [len(n) for n in neighbors]
    print(f"  degree    : min={min(degrees)}  max={max(degrees)}"
          f"  mean={sum(degrees)/len(degrees):.1f}")

    # ── Visualization ────────────────────────────────────────────────────────────
    print_section("Visualization")

    from scipy.spatial import Voronoi

    Pnp = P.numpy()
    Snp = S.numpy()
    cell_np = cell_ids.numpy()
    radii_np = radii.numpy()
    counts_np = counts.numpy()

    pad = 2.0
    xmin = Pnp[:, 0].min() - pad;  xmax = Pnp[:, 0].max() + pad
    ymin = Pnp[:, 1].min() - pad;  ymax = Pnp[:, 1].max() + pad
    clip_box = (xmin, xmax, ymin, ymax)

    vor = Voronoi(Snp)
    vor_segs = _voronoi_finite_segments(vor, clip_box)

    cmap = plt.get_cmap("tab20")
    cell_colors = cell_np % 20

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        f"Voronoi Primitives Demo  |  N={P.shape[0]:,} pts  ·  M={M} FPS samples",
        fontsize=14, fontweight="bold",
    )

    def _lim(ax):
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")

    # ── Panel 1: raw point cloud ─────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.scatter(Pnp[:, 0], Pnp[:, 1], s=0.5, c="steelblue", alpha=0.35, linewidths=0)
    ax.scatter(Snp[:, 0], Snp[:, 1], s=35, c="red", zorder=5,
               edgecolors="white", linewidths=0.5, label=f"FPS ({M})")
    ax.set_title("Raw Point Cloud + FPS Samples")
    ax.legend(markerscale=1, fontsize=8, loc="upper right")
    _lim(ax)

    # ── Panel 2: Voronoi cells ───────────────────────────────────────────────────
    ax = axes[0, 1]
    ax.scatter(Pnp[:, 0], Pnp[:, 1], s=0.5, c=cell_colors, cmap="tab20",
               alpha=0.45, linewidths=0, vmin=0, vmax=19)
    if vor_segs:
        lc = LineCollection(vor_segs, colors="0.25", linewidths=0.7, zorder=3)
        ax.add_collection(lc)
    ax.scatter(Snp[:, 0], Snp[:, 1], s=40, c="red", zorder=6,
               edgecolors="white", linewidths=0.5, label="FPS")
    ax.set_title("Voronoi Cells (coloured by membership)")
    ax.legend(markerscale=1, fontsize=8, loc="upper right")
    _lim(ax)

    # ── Panel 3: Delaunay graph ──────────────────────────────────────────────────
    ax = axes[0, 2]
    ax.scatter(Pnp[:, 0], Pnp[:, 1], s=0.3, c="lightgray", alpha=0.25, linewidths=0)
    del_segs = []
    for vi, nbrs in enumerate(neighbors):
        for vj in nbrs:
            if vj > vi:
                del_segs.append([Snp[vi], Snp[vj]])
    if del_segs:
        lc = LineCollection(del_segs, colors="royalblue", linewidths=0.8,
                            alpha=0.7, zorder=3)
        ax.add_collection(lc)
    ax.scatter(Snp[:, 0], Snp[:, 1], s=40, c="red", zorder=6,
               edgecolors="white", linewidths=0.5, label="FPS")
    ax.set_title("Delaunay Graph on FPS Samples")
    ax.legend(markerscale=1, fontsize=8, loc="upper right")
    _lim(ax)

    # ── Panel 4: covering radius circles ────────────────────────────────────────
    ax = axes[1, 0]
    ax.scatter(Pnp[:, 0], Pnp[:, 1], s=0.3, c="lightgray", alpha=0.25, linewidths=0)
    for vi in range(M):
        r = float(radii_np[vi])
        if r > 0:
            circle = plt.Circle(
                (Snp[vi, 0], Snp[vi, 1]), r,
                color=cmap(vi % 20), alpha=0.18, zorder=2,
            )
            ax.add_patch(circle)
            circle_edge = plt.Circle(
                (Snp[vi, 0], Snp[vi, 1]), r,
                fill=False, edgecolor=cmap(vi % 20), linewidth=0.5,
                alpha=0.6, zorder=3,
            )
            ax.add_patch(circle_edge)
    ax.scatter(Snp[:, 0], Snp[:, 1], s=40, c="red", zorder=6,
               edgecolors="white", linewidths=0.5, label="FPS")
    # mark closest pair
    ax.plot([Snp[i, 0], Snp[j, 0]], [Snp[i, 1], Snp[j, 1]],
            "k--", lw=1.2, zorder=7, label=f"min dist={min_dist:.2f}")
    ax.set_title("Covering Radius per Cell")
    ax.legend(markerscale=1, fontsize=8, loc="upper right")
    _lim(ax)

    # ── Panel 5: cell occupancy histogram ───────────────────────────────────────
    ax = axes[1, 1]
    ax.hist(counts_np, bins=min(30, M // 2 + 1),
            color="steelblue", edgecolor="white", linewidth=0.5)
    ax.axvline(counts_np.mean(), color="red", lw=1.5,
               label=f"mean = {counts_np.mean():.1f}")
    ax.set_xlabel("Points per cell")
    ax.set_ylabel("Number of cells")
    ax.set_title("Cell Occupancy Distribution")
    ax.legend(fontsize=9)

    # ── Panel 6: covering radius histogram ──────────────────────────────────────
    ax = axes[1, 2]
    nz = radii_np[radii_np > 0]
    ax.hist(nz, bins=min(30, M // 2 + 1),
            color="darkorange", edgecolor="white", linewidth=0.5)
    ax.axvline(nz.mean(), color="red", lw=1.5,
               label=f"mean = {nz.mean():.3f}")
    ax.set_xlabel("Covering radius")
    ax.set_ylabel("Number of cells")
    ax.set_title("Covering Radius Distribution")
    ax.legend(fontsize=9)

    plt.tight_layout()
    out = "voronoi_demo.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved {out}")
    try:
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()
