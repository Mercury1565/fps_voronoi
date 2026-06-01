import torch
from scipy.spatial import Delaunay, QhullError

_CHUNK = 4096


def cell_membership(P: torch.Tensor, S: torch.Tensor):
    """
    For each cloud point, find the nearest sample in S.
    """

    M = S.shape[0]
    if M == 0:
        raise ValueError("S must contain at least one sample (M > 0).")

    N = P.shape[0]
    cell_ids = torch.empty(N, dtype=torch.int64, device=P.device)
    distances = torch.empty(N, dtype=torch.float32, device=P.device)

    for lo in range(0, N, _CHUNK):
        hi = min(lo + _CHUNK, N)
        d = torch.cdist(P[lo:hi], S)
        idx = d.argmin(dim=1)
        cell_ids[lo:hi] = idx
        distances[lo:hi] = d.gather(1, idx.unsqueeze(1)).squeeze(1)

    return cell_ids, distances


def covering_radius(
    cell_ids: torch.Tensor,
    distances: torch.Tensor,
    num_samples: int,
) -> torch.Tensor:
    """
    Max distance from any cell member to the cell's sample.
    """

    radii = torch.zeros(num_samples, dtype=torch.float32, device=distances.device)
    if cell_ids.shape[0] == 0:
        return radii

    ids = cell_ids.long()
    try:
        radii.scatter_reduce_(0, ids, distances, reduce="amax", include_self=True)
    except (AttributeError, TypeError, RuntimeError):
        # Fallback for PyTorch < 2.0
        for i in range(ids.shape[0]):
            c = ids[i].item()
            d = distances[i].item()
            if d > radii[c].item():
                radii[c] = d

    return radii


def min_pairwise_distance(S: torch.Tensor):
    """
    Minimum Euclidean distance between any two distinct samples.
    """

    M = S.shape[0]
    if M < 2:
        raise ValueError("S must contain at least two samples (M >= 2).")

    dists = torch.cdist(S, S)           # (M, M)
    dists.fill_diagonal_(float("inf"))

    flat_idx = int(dists.reshape(-1).argmin().item())
    i, j = flat_idx // M, flat_idx % M
    if i > j:
        i, j = j, i

    dist = dists[i, j].float()
    pair = torch.tensor([i, j], dtype=torch.int64, device=S.device)
    return dist, pair


def cell_occupancy(cell_ids: torch.Tensor, num_samples: int) -> torch.Tensor:
    """
    Number of cloud points in each Voronoi cell.
    """
    try:
        counts = torch.bincount(cell_ids, minlength=num_samples)
    except RuntimeError:
        # bincount requires CPU on some older PyTorch builds
        counts = torch.bincount(
            cell_ids.cpu(), minlength=num_samples
        ).to(cell_ids.device)
    return counts.to(torch.int64)


def delaunay_neighbors(S: torch.Tensor):
    """
    Delaunay adjacency graph of the sample set.
    """
    M = S.shape[0]
    pts = S.detach().cpu().numpy()

    try:
        tri = Delaunay(pts)
    except QhullError as e:
        raise ValueError(f"S is degenerate (Qhull error): {e}") from e

    adjacency = [set() for _ in range(M)]
    for simplex in tri.simplices:
        for a in simplex:
            for b in simplex:
                if a != b:
                    adjacency[a].add(int(b))

    return [sorted(nbrs) for nbrs in adjacency]
