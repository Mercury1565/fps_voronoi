import argparse
import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "phase_1"))
sys.path.insert(0, os.path.join(_HERE, "..", "phase_2"))
sys.path.insert(0, os.path.join(_HERE, ".."))

from data_io import list_frames, load_lidar_bin, chamfer_distance  # noqa: E402
from fps import farthest_point_sampling             # noqa: E402
from correct import correct                         # noqa: E402

NUM_SAMPLES = 1024

def _resize_to(P, S, target):
    """Hold the sample count at exactly ``target``.

    If ``S`` is short, greedily append the cloud points farthest from the current
    samples (an FPS continuation — fills the worst-covered spots). If ``S`` has too
    many, drop the most redundant samples one at a time (smallest sample-to-sample
    nearest-neighbour distance first). Returns a tensor with ``target`` rows.
    """
    n = S.shape[0]
    if n == target:
        return S
    S = S.clone()
    if n < target:                       # top up at the worst-covered points
        for _ in range(target - n):
            d = torch.cdist(P, S).min(dim=1).values
            j = int(torch.argmax(d))
            S = torch.cat([S, P[j : j + 1]], dim=0)
    else:                                # trim the most redundant samples
        for _ in range(n - target):
            dd = torch.cdist(S, S)
            dd.fill_diagonal_(float("inf"))
            j = int(torch.argmin(dd.min(dim=1).values))
            S = torch.cat([S[:j], S[j + 1 :]], dim=0)
    return S


def main():
    ap = argparse.ArgumentParser(description="Frame-to-frame sampling stability")
    ap.add_argument("--dataset", choices=["nuscenes", "kitti"], default="nuscenes")
    ap.add_argument("--start", type=int, default=0, help="First frame index")
    ap.add_argument("--num-frames", type=int, default=30)
    ap.add_argument("--dims", type=int, choices=[2, 3], default=3)
    ap.add_argument("--max-range", type=float, default=40.0)
    ap.add_argument("--min-z", type=float, default=-1.5)
    ap.add_argument("--samples", type=int, default=NUM_SAMPLES)
    ap.add_argument("--fixed-samples", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Hold the sample count constant at --samples every frame "
                         "(top up worst-covered spots / trim most-redundant samples "
                         "after each correction). Use --no-fixed-samples to let M "
                         "drift as insertions/evictions fall out.")
    ap.add_argument("--budget", type=int, default=40,
                    help="Max edits/frame before falling back to full FPS.")
    ap.add_argument("--coverage-factor", type=float, default=2.0)
    ap.add_argument("--separation-factor", type=float, default=0.5)
    ap.add_argument("--min-occupancy", type=int, default=1)
    ap.add_argument("--baseline", action="store_true",
                    help="Also FPS-rebuild each frame at equal M and report "
                         "fps_cham (P<->Sf), the ratio (P<->S / P<->Sf), and the "
                         "direct carried-vs-fresh distance (S<->Sf).")
    ap.add_argument("--out", default="temporal_demo.png")
    args = ap.parse_args()

    paths = list_frames(args.dataset)[args.start : args.start + args.num_frames]
    if len(paths) < 2:
        raise SystemExit(f"Need >= 2 frames, found {len(paths)} for {args.dataset}")

    def load(path):
        return load_lidar_bin(path, dims=args.dims,
                              max_range=args.max_range, min_z=args.min_z)

    # ── Frame 0: fresh FPS baseline ──────────────────────────────────────────
    P = load(paths[0])
    S = P[farthest_point_sampling(P, args.samples, seed=42)].clone()
    print(f"\n  {args.dataset}  frames {args.start}..{args.start + len(paths) - 1}"
          f"  ·  {args.dims}-D  ·  M0={S.shape[0]} samples")

    base_cols = f" {'fps_cham':>9} {'ratio':>6} {'direct_fps':>8}" if args.baseline else ""
    header = (f"  {'frame':>5} {'pts':>7} {'M':>4} {'edits':>6} {'ins':>4} "
              f"{'evt':>4} {'carried_over_chamfer':>8}{base_cols}  note")
    print(header)
    print("  " + "─" * (len(header) - 2))
    # Frame 0 baseline: the fresh FPS sampling vs its own cloud.
    cham0, _, _ = chamfer_distance(P, S)
    base_pad = f"{'-':>9} {'-':>6} {'-':>8}" if args.baseline else ""
    print(f"  {args.start:>5} {P.shape[0]:>7} {S.shape[0]:>4} {'-':>6} {'-':>4} "
          f"{'-':>4} {cham0:>20.3f}"
          f"{(' ' + base_pad) if args.baseline else ''}  FPS baseline")

    rec = {k: [] for k in
           ("frame", "pts", "M", "edits", "ins", "evt",
            "carried_over_chamfer", "fps_cham", "ratio", "direct_fps", "fallback")}

    for t in range(1, len(paths)):
        P = load(paths[t])

        res = correct(
            P, S,
            budget=args.budget,
            fps_seed=42,
            coverage_factor=args.coverage_factor,
            separation_factor=args.separation_factor,
            min_occupancy=args.min_occupancy,
        )
        S = res.S  # carry the corrected sampling into the next frame
        if args.fixed_samples:
            S = _resize_to(P, S, args.samples)  # pin M = --samples every frame

        # Headline metric: how close is the carried sampling to the actual cloud?
        cham, _, _ = chamfer_distance(P, S)

        base_str = ""
        fps_cham = ratio = direct = float("nan")
        if args.baseline:
            # Fair fight: fresh FPS at the SAME sample count on this frame.
            Sf = P[farthest_point_sampling(P, S.shape[0], seed=42)]
            fps_cham, _, _ = chamfer_distance(P, Sf)
            ratio = cham / fps_cham if fps_cham > 0 else float("nan")
            # Direct sample-to-sample distance (for comparison only — see README).
            direct, _, _ = chamfer_distance(S, Sf)
            base_str = f" {fps_cham:>9.3f} {ratio:>6.2f} {direct:>8.3f}"

        note = "FULL FPS REBUILD" if res.fallback else ""
        print(f"  {args.start + t:>5} {P.shape[0]:>7} {S.shape[0]:>4} "
              f"{res.n_requested:>6} {res.n_inserted:>4} {res.n_evicted:>4} "
              f"{cham:>20.3f}{base_str}  {note}")

        rec["frame"].append(args.start + t)
        rec["pts"].append(P.shape[0])
        rec["M"].append(S.shape[0])
        rec["edits"].append(res.n_requested)
        rec["ins"].append(res.n_inserted if not res.fallback else 0)
        rec["evt"].append(res.n_evicted if not res.fallback else 0)
        rec["carried_over_chamfer"].append(cham)
        rec["fps_cham"].append(fps_cham)
        rec["ratio"].append(ratio)
        rec["direct_fps"].append(direct)
        rec["fallback"].append(res.fallback)

    _plot(rec, args)


def _plot(rec, args):
    f = np.array(rec["frame"])
    edits = np.array(rec["edits"])
    ins = np.array(rec["ins"])
    evt = np.array(rec["evt"])
    fb = np.array(rec["fallback"])

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    fig.suptitle(
        f"Sampling stability across consecutive frames  |  {args.dataset}  ·  "
        f"{args.dims}-D  ·  budget={args.budget}",
        fontsize=13, fontweight="bold",
    )

    # ── Top: edits requested per frame (insert / evict split) + fallbacks ─────
    ax0.bar(f, ins, color="seagreen", label="insertions")
    ax0.bar(f, evt, bottom=ins, color="indianred", label="evictions")
    ax0.axhline(args.budget, color="0.4", ls="--", lw=1,
                label=f"budget ({args.budget})")
    if fb.any():
        ax0.scatter(f[fb], edits[fb], marker="v", s=90, color="black", zorder=5,
                    label="full FPS rebuild")
    ax0.set_ylabel("edits requested")
    ax0.set_title("How much the sampling must adapt each frame "
                  "(higher = bigger scene change)")
    ax0.legend(fontsize=8, loc="upper right")
    ax0.grid(axis="y", alpha=0.3)

    # ── Bottom: chamfer comparisons (carried vs fresh, against the cloud) ──────
    cham = np.array(rec["carried_over_chamfer"])
    ax1.plot(f, cham, color="purple", lw=1.8, marker="o", ms=3,
             label="chamfer  P ↔ carried S")
    if np.isfinite(rec["fps_cham"]).any():
        fps_cham = np.array(rec["fps_cham"])
        direct = np.array(rec["direct_fps"])
        ax1.plot(f, fps_cham, color="0.4", lw=1.5, ls=":", marker="s", ms=3,
                 label="chamfer  P ↔ fresh FPS Sf (equal M)")
        ax1.plot(f, direct, color="darkorange", lw=1.3, ls="--", marker="^", ms=3,
                 label="chamfer  carried S ↔ fresh FPS Sf (direct)")
    ax1.set_ylabel("chamfer distance (m)")
    ax1.set_xlabel("frame index")
    ax1.set_title("Reuse vs. fresh FPS, judged against the cloud "
                  "(lower = closer; gap P↔S to P↔Sf = cost of reuse)")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(alpha=0.3)

    # Ratio on a secondary axis (dimensionless; 1.0 = reuse as good as rebuild).
    if np.isfinite(rec["ratio"]).any():
        ratio = np.array(rec["ratio"])
        axr = ax1.twinx()
        axr.plot(f, ratio, color="crimson", lw=1.0, alpha=0.55)
        axr.axhline(1.0, color="crimson", ls=":", lw=0.8, alpha=0.5)
        axr.set_ylabel("ratio = P↔S / P↔Sf", color="crimson")
        axr.tick_params(axis="y", labelcolor="crimson")

    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\n  Saved {args.out}")
    try:
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()
