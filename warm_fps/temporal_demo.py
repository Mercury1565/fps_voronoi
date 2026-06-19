import argparse
import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))

from resample import warm_resample, fps_continue              # noqa: E402
from data_io import list_frames, load_lidar_bin, chamfer_distance  # noqa: E402
import config                                                  # noqa: E402


def _median_nn(S: torch.Tensor) -> float:
    """Median nearest-neighbour distance among the samples (a length scale)."""
    if S.shape[0] < 2:
        return 0.0
    D = torch.cdist(S, S)
    D.fill_diagonal_(float("inf"))
    return float(D.min(dim=1).values.median())


def main():
    ap = argparse.ArgumentParser(description="Warm-started FPS resampling over frames")
    ap.add_argument("--dataset", choices=["nuscenes", "kitti"], default="nuscenes")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--num-frames", type=int, default=30)
    ap.add_argument("--dims", type=int, choices=[2, 3], default=config.DIMS)
    ap.add_argument("--max-range", type=float, default=config.MAX_RANGE)
    ap.add_argument("--min-z", type=float, default=config.MIN_Z)
    ap.add_argument("--samples", type=int, default=config.SAMPLES)
    # ── validity thresholds (data-adaptive: factor x median sample spacing) ──
    ap.add_argument("--min-occupancy", type=int, default=1,
                    help="Drop a sample whose cell holds fewer points than this.")
    ap.add_argument("--stale-factor", type=float, default=2.0,
                    help="Drop a sample sitting farther than factor x median "
                         "sample spacing from any real point (0 disables).")
    ap.add_argument("--separation-factor", type=float, default=0.5,
                    help="Drop the lesser of two samples closer than factor x "
                         "median sample spacing (0 disables).")
    ap.add_argument("--baseline", action="store_true",
                    help="Also cold-rebuild a fresh FPS each frame at equal M and "
                         "report fps_cham (P<->Sf), ratio, and direct (S<->Sf).")
    ap.add_argument("--out", default="warm_fps_demo.png")
    args = ap.parse_args()

    paths = list_frames(args.dataset)[args.start : args.start + args.num_frames]
    if len(paths) < 2:
        raise SystemExit(f"Need >= 2 frames, found {len(paths)} for {args.dataset}")

    def load(path):
        return load_lidar_bin(path, dims=args.dims,
                              max_range=args.max_range, min_z=args.min_z)

    def thresholds(S):
        """Turn the relative factors into absolute metres for this frame."""
        scale = _median_nn(S)
        stale = args.stale_factor * scale if args.stale_factor > 0 else None
        sep = args.separation_factor * scale if args.separation_factor > 0 else None
        return stale, sep

    # ── Frame 0: cold FPS baseline ────────────────────────────────────────────
    P = load(paths[0])
    S = fps_continue(P, None, args.samples, seed=42)
    print(f"\n  {args.dataset}  frames {args.start}..{args.start + len(paths) - 1}"
          f"  ·  {args.dims}-D  ·  M={args.samples} (constant)  ·  warm-FPS resample")
    print("  kept/drop/refill = carried samples reused / discarded / re-sampled by FPS")
    print("  chamfer = P <-> carried S   fps_cham = P <-> fresh cold FPS   "
          "ratio = chamfer/fps_cham\n")

    base_cols = f" {'fps_cham':>9} {'ratio':>6} {'direct':>7}" if args.baseline else ""
    header = (f"  {'frame':>5} {'pts':>7} {'M':>5} {'kept':>5} {'drop':>5} "
              f"{'refill':>6} {'chamfer':>8}{base_cols}")
    print(header)
    print("  " + "─" * (len(header) - 2))

    cham0, _, _ = chamfer_distance(P, S)
    base_pad = f"{'-':>9} {'-':>6} {'-':>7}" if args.baseline else ""
    print(f"  {args.start:>5} {P.shape[0]:>7} {S.shape[0]:>5} {'-':>5} {'-':>5} "
          f"{'-':>6} {cham0:>8.3f}{(' ' + base_pad) if args.baseline else ''}")

    rec = {k: [] for k in ("frame", "kept", "drop", "refill",
                           "chamfer", "fps_cham", "ratio", "direct")}

    for t in range(1, len(paths)):
        P = load(paths[t])
        stale, sep = thresholds(S)

        res = warm_resample(
            P, S, args.samples,
            min_occupancy=args.min_occupancy,
            stale_dist=stale,
            separation_min=sep,
        )
        S = res.S  # carry forward

        cham, _, _ = chamfer_distance(P, S)

        base_str = ""
        fps_cham = ratio = direct = float("nan")
        if args.baseline:
            Sf = fps_continue(P, None, args.samples, seed=42)   # cold rebuild, equal M
            fps_cham, _, _ = chamfer_distance(P, Sf)
            ratio = cham / fps_cham if fps_cham > 0 else float("nan")
            direct, _, _ = chamfer_distance(S, Sf)
            base_str = f" {fps_cham:>9.3f} {ratio:>6.2f} {direct:>7.3f}"

        print(f"  {args.start + t:>5} {P.shape[0]:>7} {S.shape[0]:>5} "
              f"{res.n_kept:>5} {res.n_dropped:>5} {res.n_refilled:>6} "
              f"{cham:>8.3f}{base_str}")

        rec["frame"].append(args.start + t)
        rec["kept"].append(res.n_kept)
        rec["drop"].append(res.n_dropped)
        rec["refill"].append(res.n_refilled)
        rec["chamfer"].append(cham)
        rec["fps_cham"].append(fps_cham)
        rec["ratio"].append(ratio)
        rec["direct"].append(direct)

    _plot(rec, args)


def _plot(rec, args):
    f = np.array(rec["frame"])
    kept = np.array(rec["kept"])
    refill = np.array(rec["refill"])

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    fig.suptitle(f"Warm-FPS resampling across frames  |  {args.dataset}  ·  "
                 f"{args.dims}-D  ·  M={args.samples}",
                 fontsize=13, fontweight="bold")

    # ── Top: how the constant-M budget splits into reused vs re-sampled ───────
    ax0.bar(f, kept, color="seagreen", label="kept (reused FPS seeds)")
    ax0.bar(f, refill, bottom=kept, color="indianred", label="refilled (fresh FPS)")
    ax0.set_ylabel("samples")
    ax0.set_title("Validity split each frame (more red = more of the scene changed)")
    ax0.legend(fontsize=8, loc="lower right")
    ax0.grid(axis="y", alpha=0.3)

    # ── Bottom: chamfer of the warm sampling vs a cold rebuild ────────────────
    cham = np.array(rec["chamfer"])
    ax1.plot(f, cham, color="purple", lw=1.8, marker="o", ms=3,
             label="chamfer  P ↔ warm S")
    if np.isfinite(rec["fps_cham"]).any():
        ax1.plot(f, np.array(rec["fps_cham"]), color="0.4", lw=1.5, ls=":",
                 marker="s", ms=3, label="chamfer  P ↔ cold FPS (equal M)")
        ax1.plot(f, np.array(rec["direct"]), color="darkorange", lw=1.3, ls="--",
                 marker="^", ms=3, label="chamfer  warm S ↔ cold FPS (direct)")
    ax1.set_ylabel("chamfer distance (m)")
    ax1.set_xlabel("frame index")
    ax1.set_title("Warm sampling vs. cold rebuild (lower = closer)")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(alpha=0.3)

    if np.isfinite(rec["ratio"]).any():
        axr = ax1.twinx()
        axr.plot(f, np.array(rec["ratio"]), color="crimson", lw=1.0, alpha=0.55)
        axr.axhline(1.0, color="crimson", ls=":", lw=0.8, alpha=0.5)
        axr.set_ylabel("ratio = warm / cold", color="crimson")
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
