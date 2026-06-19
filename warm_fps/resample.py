from dataclasses import dataclass
import torch


# Low-level geometry helpers (chunked so they scale to large clouds)
def _min_dist_to(P: torch.Tensor, C: torch.Tensor, chunk: int = 8192) -> torch.Tensor:
    """For each point in ``P``, distance to its nearest point in ``C``. Shape (N,)."""
    out = torch.empty(P.shape[0], dtype=torch.float32, device=P.device)
    for lo in range(0, P.shape[0], chunk):
        out[lo : lo + chunk] = torch.cdist(P[lo : lo + chunk], C).min(dim=1).values
    return out


def _cell_stats(P: torch.Tensor, S: torch.Tensor, chunk: int = 8192):
    """One fused pass over the cloud, returning the signals validity needs:

        occupancy[m]  — how many cloud points fall in sample m's Voronoi cell
        faith[m]      — distance from sample m to its NEAREST cloud point
                        (large => the sample floats in empty space => stale)

    ``faith`` is the per-sample 'faithfulness' half of the Chamfer distance.
    """
    N, M = P.shape[0], S.shape[0]
    nearest_id = torch.empty(N, dtype=torch.long, device=P.device)
    faith = torch.full((M,), float("inf"), dtype=torch.float32, device=P.device)
    for lo in range(0, N, chunk):
        d = torch.cdist(P[lo : lo + chunk], S)         # (chunk, M)
        nearest_id[lo : lo + chunk] = d.argmin(dim=1)  # owning cell per point
        faith = torch.minimum(faith, d.min(dim=0).values)
    occupancy = torch.bincount(nearest_id, minlength=M)
    return occupancy, faith


# Step 1 — validity classifier
def valid_mask(
    P: torch.Tensor,
    S: torch.Tensor,
    min_occupancy: int = 1,
    stale_dist: float | None = None,
    separation_min: float | None = None,
):
    """Boolean keep-mask over the carried samples ``S`` (shape (M,)).

    A sample is dropped if it is:
      • **vanished**   — its cell holds fewer than ``min_occupancy`` points;
      • **stale**      — it sits farther than ``stale_dist`` from any real point
                         (only checked when ``stale_dist`` is given);
      • **redundant**  — it lies within ``separation_min`` of another kept sample;
                         of a too-close pair the lower-occupancy one is dropped
                         (only checked when ``separation_min`` is given).

    Returns ``(keep, occupancy, faith)``.
    """
    M = S.shape[0]
    occupancy, faith = _cell_stats(P, S)

    keep = occupancy >= min_occupancy                  # vanished cells out
    if stale_dist is not None:
        keep &= faith <= stale_dist                    # floating/stale samples out

    if separation_min is not None and M > 1:
        # Greedily thin redundant clusters: walk samples from least to most
        # populated, dropping any that still sit too close to a surviving one.
        D = torch.cdist(S, S)
        D.fill_diagonal_(float("inf"))
        for i in torch.argsort(occupancy).tolist():
            if not keep[i]:
                continue
            close = (D[i] < separation_min) & keep
            close[i] = False
            if bool(close.any()):
                keep[i] = False

    return keep


# Step 3 — warm-started Farthest-Point Sampling
def fps_continue(
    P: torch.Tensor,
    seeds: torch.Tensor | None,
    num_samples: int,
    seed: int | None = None,
    chunk: int = 8192,
) -> torch.Tensor:
    """Farthest-Point Sampling that *continues* from an existing set of centres.

    ``seeds`` are kept centres (positions, not necessarily members of ``P``); the
    greedy FPS loop runs from them, each new pick being the cloud point farthest
    from everything chosen so far, until ``num_samples`` centres exist. The kept
    seeds occupy the first rows of the result, so they are preserved exactly.

    With ``seeds=None`` (or empty) this degrades to ordinary cold-start FPS, with
    ``seed`` selecting the random first pick. The warm continuation itself is
    deterministic — no RNG once there is at least one centre.
    """
    N = P.shape[0]
    if num_samples > N:
        raise ValueError(f"num_samples ({num_samples}) must be <= N ({N}).")

    have = 0 if seeds is None else seeds.shape[0]
    if have >= num_samples:
        return seeds[:num_samples].clone()

    if have == 0:                                      # cold start: random first pick
        if seed is not None:
            torch.manual_seed(seed)
        first = int(torch.randint(0, N, (1,), device=P.device))
        centres = [P[first : first + 1].clone()]
        min_dist = torch.norm(P - P[first], dim=1)
        have = 1
    else:                                              # warm start from the seeds
        centres = [seeds.clone()]
        min_dist = _min_dist_to(P, seeds, chunk)

    while have < num_samples:
        j = int(torch.argmax(min_dist))                # worst-covered cloud point
        centres.append(P[j : j + 1])
        min_dist = torch.minimum(min_dist, torch.norm(P - P[j], dim=1))
        have += 1

    return torch.cat(centres, dim=0)


# The per-frame operation: classify -> keep -> refill
@dataclass
class WarmResampleResult:
    """Outcome of one warm resample.

    S               new sample set, exactly ``num_samples`` rows
    n_kept          carried samples judged valid and reused (the FPS seeds)
    n_dropped       carried samples discarded (vanished / stale / redundant)
    n_refilled      points added by the FPS continuation (= num_samples - n_kept)
    valid_fraction  n_kept / len(S_prev) — how much of the old sampling survived
    """

    S: torch.Tensor
    n_kept: int
    n_dropped: int
    n_refilled: int
    valid_fraction: float


def warm_resample(
    P: torch.Tensor,
    S_prev: torch.Tensor,
    num_samples: int,
    min_occupancy: int = 1,
    stale_dist: float | None = None,
    separation_min: float | None = None,
    seed: int | None = None,
) -> WarmResampleResult:
    """Carry ``S_prev`` into the new cloud ``P`` by partial rebuild.

    Drops the carried samples that are no longer valid, then warm-starts FPS from
    the survivors to refill back to ``num_samples``. Returns the new sampling and
    the keep/drop/refill counts.
    """
    keep = valid_mask(
        P, S_prev,
        min_occupancy=min_occupancy,
        stale_dist=stale_dist,
        separation_min=separation_min,
    )
    S_valid = S_prev[keep]
    if S_valid.shape[0] > num_samples:                 # never seed past the target
        S_valid = S_valid[:num_samples]

    S_new = fps_continue(
        P, S_valid if S_valid.shape[0] > 0 else None, num_samples, seed=seed,
    )

    n_kept = S_valid.shape[0]
    n_dropped = int(S_prev.shape[0] - int(keep.sum()))
    return WarmResampleResult(
        S=S_new,
        n_kept=n_kept,
        n_dropped=n_dropped,
        n_refilled=num_samples - n_kept,
        valid_fraction=n_kept / max(1, S_prev.shape[0]),
    )
