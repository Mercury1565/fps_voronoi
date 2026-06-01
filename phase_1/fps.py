import torch

def farthest_point_sampling(
    P: torch.Tensor,
    num_samples: int,
    seed: int = None,
) -> torch.Tensor:
    """
    Select num_samples points from P via greedy FPS.
    """

    N = P.shape[0]
    if num_samples > N:
        raise ValueError(f"num_samples ({num_samples}) must be <= N ({N}).")

    if seed is not None:
        torch.manual_seed(seed)

    selected = torch.zeros(num_samples, dtype=torch.long, device=P.device)
    min_dists = torch.full((N,), float("inf"), dtype=torch.float32, device=P.device)

    selected[0] = torch.randint(0, N, (1,), device=P.device).item()

    for k in range(1, num_samples):
        last = P[selected[k - 1]] # (3,)
        d = torch.norm(P - last, dim=1) # (N,)
        min_dists = torch.minimum(min_dists, d)
        selected[k] = min_dists.argmax()

    return selected
