"""
Temporal demo — how much do *consecutive* LiDAR frames differ, measured by how
much the sampling has to adapt?

The single-frame engine (Phases 1-3) tells us whether one cloud is sampled well.
This driver turns that into a frame-to-frame signal:

    1. FPS the first frame  → sample set S.
    2. For each later frame: carry S forward and run Phase 3's `correct()` on the
       new cloud. It diagnoses the carried samples against the new frame and
       patches them (insert at coverage gaps, evict vanished / redundant cells).
    3. The number of edits the frame *asks for* (`n_requested`) is the headline
       signal: a near-static scene needs almost no edits; a fast-changing one
       needs many. When a frame needs more edits than the budget allows, Phase 3
       gives up patching and does a full FPS rebuild — itself a "scene changed
       too much" flag.

So we never compare raw clouds directly; the *sampling's instability* is the
proxy for how much the scene moved. As a continuous companion we also report the
pre-correction misfit: the mean/max distance from the new cloud's points to the
*previous* frame's samples (how badly the old sampling fits before we fix it).

Run (nuScenes, 30 frames, full 3-D, cropped + ground-removed):
    python phase_3/temporal_demo.py --dataset nuscenes --num-frames 30 \
        --dims 3 --max-range 40 --min-z -1.5
"""

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

from data_io import list_frames, load_lidar_bin   # noqa: E402
from fps import farthest_point_sampling             # noqa: E402
from primitives import cell_membership              # noqa: E402
from correct import correct                         # noqa: E402

NUM_SAMPLES = 64


def main():
    ap = argparse.ArgumentParser(description="Frame-to-frame sampling stability")
    ap.add_argument("--dataset", choices=["nuscenes", "kitti"], default="nuscenes")
    ap.add_argument("--start", type=int, default=0, help="First frame index")
    ap.add_argument("--num-frames", type=int, default=30)
    ap.add_argument("--dims", type=int, choices=[2, 3], default=3)
    ap.add_argument("--max-range", type=float, default=40.0)
    ap.add_argument("--min-z", type=float, default=-1.5)
    ap.add_argument("--samples", type=int, default=NUM_SAMPLES)
    ap.add_argument("--budget", type=int, default=40,
                    help="Max edits/frame before falling back to full FPS.")
    ap.add_argument("--coverage-factor", type=float, default=2.0)
    ap.add_argument("--separation-factor", type=float, default=0.5)
    ap.add_argument("--min-occupancy", type=int, default=1)
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
          f"  ·  {args.dims}-D  ·  M0={S.shape[0]} samples\n")
    header = (f"  {'frame':>5} {'pts':>7} {'M':>4} {'edits':>6} {'ins':>4} "
              f"{'evt':>4} {'misfit_mean':>11} {'misfit_max':>10}  note")
    print(header)
    print("  " + "─" * (len(header) - 2))
    print(f"  {args.start:>5} {P.shape[0]:>7} {S.shape[0]:>4} {'-':>6} {'-':>4} "
          f"{'-':>4} {'-':>11} {'-':>10}  FPS baseline")

    rec = {k: [] for k in
           ("frame", "pts", "M", "edits", "ins", "evt",
            "misfit_mean", "misfit_max", "fallback")}

    for t in range(1, len(paths)):
        P = load(paths[t])

        # Pre-correction misfit: how well do the carried-forward samples already
        # cover this new cloud? (distance from each new point to nearest sample)
        _, d = cell_membership(P, S)
        misfit_mean, misfit_max = float(d.mean()), float(d.max())

        res = correct(
            P, S,
            budget=args.budget,
            fps_seed=42,
            coverage_factor=args.coverage_factor,
            separation_factor=args.separation_factor,
            min_occupancy=args.min_occupancy,
        )
        S = res.S  # carry the corrected sampling into the next frame

        note = "FULL FPS REBUILD" if res.fallback else ""
        print(f"  {args.start + t:>5} {P.shape[0]:>7} {S.shape[0]:>4} "
              f"{res.n_requested:>6} {res.n_inserted:>4} {res.n_evicted:>4} "
              f"{misfit_mean:>11.3f} {misfit_max:>10.3f}  {note}")

        rec["frame"].append(args.start + t)
        rec["pts"].append(P.shape[0])
        rec["M"].append(S.shape[0])
        rec["edits"].append(res.n_requested)
        rec["ins"].append(res.n_inserted if not res.fallback else 0)
        rec["evt"].append(res.n_evicted if not res.fallback else 0)
        rec["misfit_mean"].append(misfit_mean)
        rec["misfit_max"].append(misfit_max)
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

    # ── Bottom: pre-correction misfit (continuous companion) ──────────────────
    mm = np.array(rec["misfit_mean"])
    mx = np.array(rec["misfit_max"])
    ax1.plot(f, mx, color="darkorange", lw=1.5, marker="o", ms=3,
             label="max misfit")
    ax1.plot(f, mm, color="steelblue", lw=1.5, marker="o", ms=3,
             label="mean misfit")
    ax1.set_ylabel("dist. of new cloud\nto previous samples (m)")
    ax1.set_xlabel("frame index")
    ax1.set_title("Pre-correction misfit — how badly last frame's samples fit "
                  "the new cloud")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\n  Saved {args.out}")
    try:
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()
