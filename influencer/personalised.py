"""
Personalised authority scoring for the influencer package.

Provides a personalised-PageRank-style variant of `socialAU` that biases
user authority toward a query user's neighbourhood via teleportation, plus
a helper to derive a seed set from a query user's ego network.
"""

import torch

from .torch_centrality import safe_normalize


def socialAU_personalised(
    mu: torch.Tensor, mi: torch.Tensor, mw: torch.Tensor,
    T: torch.Tensor,
    seed_users: list,
    alpha: float = 0.15,
    epsilon: float = 0.001,
    device=0,
):
    """
    Personalised variant of `socialAU` that biases user authority toward a
    seed set via teleportation, analogous to personalised PageRank.

    Parameters
    -------------------
    mu, mi, mw, T: see `socialAU`.
    seed_users: list of int. User indices defining the personalisation
        context. The seed vector s has s[i] = 1/len(seed_users) for
        i in seed_users, else 0.
    alpha: float. Default 0.15. Teleportation probability. At each
        iteration, right after steps 4-5 of the paper compute
        a_U = (1 - alpha) * a_U + alpha * s and renormalise. alpha=0
        skips this step entirely, reducing exactly to `socialAU`.
    epsilon, device: see `socialAU`.

    Returns
    -------------------
    u, v, w: torch tensor. Same shapes as `socialAU`'s output.
    """
    if not torch.cuda.is_available():
        device = "cpu"
        print("GPU not available")

    device = torch.device(device)
    mu = mu.to(device)
    mi = mi.to(device)
    mw = mw.to(device)
    T = T.to(device)

    n, m, r = T.shape

    s = torch.zeros(n, device=device)
    if len(seed_users) > 0:
        seed_idx = torch.tensor(seed_users, device=device, dtype=torch.long)
        s[seed_idx] = 1.0 / len(seed_users)

    a_U = torch.ones(n, device=device)
    h_U = torch.ones(n, device=device)

    a_I = torch.ones(m, device=device)
    h_I = torch.ones(m, device=device)

    a_k = torch.ones(r, device=device)
    h_k = torch.ones(r, device=device)

    h = torch.ones(n, device=device)
    a = torch.ones(m, device=device)
    w = torch.ones(r, device=device)

    lambda_prev = 0.0

    while True:
        # Steps 4-5: HITS for users layer
        h_U = torch.mv(mu, a_U)
        a_U = torch.mv(mu.T, h_U)

        # Personalisation: teleport authority toward the seed users, then
        # renormalise immediately so step 10 below sees the biased value.
        # Guarded on alpha != 0 so this reduces exactly to plain socialAU
        # (which leaves a_U unnormalised here, normalising only at step 14).
        if alpha != 0:
            a_U = (1 - alpha) * a_U + alpha * s
            a_U = safe_normalize(a_U)

        # Steps 6-7: HITS for items layer
        h_I = torch.mv(mi, a_I)
        a_I = torch.mv(mi.T, h_I)

        # Steps 8-9: HITS for keywords layer
        h_k = torch.mv(mw, a_k)
        a_k = torch.mv(mw.T, h_k)

        # Step 10: h^(t+1) = A x2 a^(t) x3 w^(t) + h_U^(t+1) + a_U^(t+1)
        tensor_part = torch.tensordot(T, a, dims=([1], [0]))             # (N, R)
        tensor_part = torch.tensordot(tensor_part, w, dims=([1], [0]))   # (N,)
        h = tensor_part + h_U + a_U

        # Steps 11-12: compute T x1 h^(t+1) once, reuse for a and w updates
        a_prev = a
        Th = torch.tensordot(T, h, dims=([0], [0]))                # (M, R)
        a = torch.tensordot(Th, w, dims=([1], [0]))                # (M,)
        w = torch.tensordot(Th, a_prev, dims=([0], [0])) + a_k     # (R,)

        # Step 13: lambda = ||h|| ||a|| ||w|| (before normalisation)
        lambda_current = (
            torch.linalg.norm(h) * torch.linalg.norm(a) * torch.linalg.norm(w)
        ).item()

        # Step 14: normalise all vectors
        h = safe_normalize(h)
        a = safe_normalize(a)
        w = safe_normalize(w)

        a_U = safe_normalize(a_U)
        h_U = safe_normalize(h_U)
        a_I = safe_normalize(a_I)
        h_I = safe_normalize(h_I)
        a_k = safe_normalize(a_k)
        h_k = safe_normalize(h_k)

        # Steps 15-16: check convergence
        if lambda_current - lambda_prev <= epsilon:
            break

        lambda_prev = lambda_current

    return h.unsqueeze(0), a.unsqueeze(0), w.unsqueeze(0)


def get_ego_network_seeds(
    mu: torch.Tensor,
    query_user: int,
    hops: int = 1,
):
    """
    Users within `hops` directed steps of `query_user` in the graph
    represented by `mu` (mu[i, j] != 0 means an edge i -> j), including
    `query_user` itself. Computed via layered BFS using boolean tensor
    operations (no networkx).

    Parameters
    -------------------
    mu: torch tensor. NxN adjacency matrix, as in `socialAU`.
    query_user: int. Index of the user to centre the ego network on.
    hops: int. Default 1. Maximum number of directed hops to expand outward.

    Returns
    -------------------
    list of int. Sorted user indices in the ego network.
    """
    adjacent = mu != 0
    n = mu.shape[0]

    visited = torch.zeros(n, dtype=torch.bool, device=mu.device)
    visited[query_user] = True
    frontier = visited.clone()

    for _ in range(hops):
        if not frontier.any():
            break
        neighbours = adjacent[frontier].any(dim=0)
        frontier = neighbours & ~visited
        visited |= frontier

    return visited.nonzero(as_tuple=True)[0].tolist()
