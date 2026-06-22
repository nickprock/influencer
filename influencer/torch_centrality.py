"""
Created on Fri Jan  28 09:56:07 2022
Last update on Sun Aug 03 15:02:36 2025

@author: nico
"""

import torch


def safe_normalize(vec: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.norm(vec)
    return vec / norm if norm > 0 else torch.zeros_like(vec)


def _resolve_init(
    init: "torch.Tensor | None", size: int, device: torch.device, name: str
) -> torch.Tensor:
    """
    Resolve one of `socialAU`'s optional h_init/a_init/w_init arguments.

    Falls back to `torch.ones` (the original, no-prior behaviour) when
    `init` is None. Otherwise validates its shape and applies
    `safe_normalize`, so a supplied prior starts on equal footing with the
    all-ones default.
    """
    if init is None:
        return torch.ones(size, device=device)
    if tuple(init.shape) != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {tuple(init.shape)}")
    return safe_normalize(init.to(device=device, dtype=torch.float32))


def hits(adjMatrix: torch.Tensor, p: int = 100, device=0):
    """
    Compute HITS (Hub and Authority) scores on a graph represented by an adjacency matrix.

    Parameters
    ----------
    adjMatrix : torch.Tensor
        Square adjacency matrix of shape (N, N).
    p : int, optional
        Number of rows in the output history tensors (default is 100).
        The algorithm runs p-1 iterations starting from an all-ones initialisation.
    device : int or str, optional
        GPU device ID or "cpu". If GPU is unavailable, it defaults to "cpu".

    Returns
    -------
    hub : dict
        Final hub scores as a dictionary of node index -> score.
    authority : dict
        Final authority scores as a dictionary of node index -> score.
    h_all : torch.Tensor
        Hub scores over iterations, shape (p, N).
    a_all : torch.Tensor
        Authority scores over iterations, shape (p, N).
    """
    if not torch.cuda.is_available():
        device = "cpu"
        print("GPU not available")

    device = torch.device(device)
    adjMatrix = adjMatrix.to(device)
    n = adjMatrix.shape[0]

    # Pre-allocate history tensors (avoids O(p²) torch.vstack allocations)
    h_all = torch.empty((p, n), device=device)
    a_all = torch.empty((p, n), device=device)
    h_all[0] = 1.0
    a_all[0] = 1.0

    for t in range(1, p):
        h_all[t] = safe_normalize(torch.mv(adjMatrix, a_all[t - 1]))
        a_all[t] = safe_normalize(torch.mv(adjMatrix.T, h_all[t]))

    hub = {str(i): h_all[-1, i].item() for i in range(n)}
    authority = {str(i): a_all[-1, i].item() for i in range(n)}

    return hub, authority, h_all, a_all


def tophits(T: torch.Tensor, epsilon: float = 1e-3, max_iter: int = 1000, device=0):
    """
    Compute TOPHITS scores for a 3D tensor using iterative decomposition.

    Parameters
    ----------
    T : torch.Tensor
        A 3D tensor of shape (N, M, Q).
    epsilon : float, optional
        Convergence threshold on lambda difference (default is 1e-3).
    max_iter : int, optional
        Maximum number of iterations to prevent infinite loops (default is 1000).
    device : int or str, optional
        GPU device ID or "cpu". If GPU is unavailable, it defaults to "cpu".

    Returns
    -------
    u : torch.Tensor
        Score vector for the first dimension (users), shape (N,).
    v : torch.Tensor
        Score vector for the second dimension (items), shape (M,).
    w : torch.Tensor
        Score vector for the third dimension (keywords), shape (Q,).
    """
    if not torch.cuda.is_available():
        device = "cpu"
        print("GPU not available")

    device = torch.device(device)
    T = T.to(device)

    N, M, Q = T.shape

    u = safe_normalize(torch.ones(N, device=device))
    v = safe_normalize(torch.ones(M, device=device))
    w = safe_normalize(torch.ones(Q, device=device))

    lambda_prev = 0.0

    for _ in range(max_iter):
        # Update u: T ×₂ v ×₃ w
        uv = torch.tensordot(T, v, dims=([1], [0]))       # (N, Q)
        u_new = torch.tensordot(uv, w, dims=([1], [0]))   # (N,)

        # Compute T ×₁ u_new once and reuse for both v and w updates
        Tu = torch.tensordot(T, u_new, dims=([0], [0]))   # (M, Q)

        # Update v: T ×₁ u_new ×₃ w
        v_new = torch.tensordot(Tu, w, dims=([1], [0]))   # (M,)

        # Update w: T ×₁ u_new ×₂ v_new  (reuses Tu)
        w_new = torch.tensordot(Tu, v_new, dims=([0], [0]))  # (Q,)

        # Compute lambda BEFORE normalization (paper step 13: λ = ||h|| ||a|| ||w||)
        lambda_curr = (
            torch.linalg.norm(u_new) *
            torch.linalg.norm(v_new) *
            torch.linalg.norm(w_new)
        ).item()

        # Normalize (paper step 14) and update before convergence check
        u = safe_normalize(u_new)
        v = safe_normalize(v_new)
        w = safe_normalize(w_new)

        if lambda_curr - lambda_prev < epsilon:
            break

        lambda_prev = lambda_curr

    return u, v, w


def socialAU(
    mu, mi, mw, T,
    epsilon: float = 0.001,
    device=0,
    h_init: "torch.Tensor | None" = None,
    a_init: "torch.Tensor | None" = None,
    w_init: "torch.Tensor | None" = None,
):
    """
    Calculate the socialAU score in a 3 layer net and detect the influencer.

    Parameters
    -------------------
    mu, mi, mw: torch tensor. These are 3 adjacency matrices with dimension NxN, MxM, QxQ.
    T: torch tensor. A 3D tensor with dimension NxMxQ.
    epsilon: float. Default 0.001. Stop criteria. Algorithm stops when
        lambda(t) - lambda(t-1) <= epsilon (paper step 15).
    device: Default 0. Set the GPU. If GPU is not available it is set to "cpu" automatically.
    h_init, a_init, w_init: optional torch tensor of shape (n,), (m,), (r,)
        respectively. Default None, which reproduces the original
        all-ones initialisation. When provided, used instead of
        `torch.ones` to seed h, a, w and normalised via `safe_normalize`
        before the first iteration, so content-based embeddings can serve
        as a warm-start prior for nodes with little or no graph
        connectivity. Raises ValueError on a shape mismatch.

    Returns
    -------------------
    u, v, w: torch tensor. The socialAU scores for each node about, respectively,
        the first, second and third dimension of tensor (3 social networks).
    """
    if not torch.cuda.is_available():
        device = "cpu"
        print("GPU not available")

    device = torch.device(device)
    mu = mu.to(device)
    mi = mi.to(device)
    mw = mw.to(device)
    T = T.to(device)

    # Step 1: initialise all vectors to ones
    n, m, r = T.shape

    a_U = torch.ones(n, device=device)
    h_U = torch.ones(n, device=device)

    a_I = torch.ones(m, device=device)
    h_I = torch.ones(m, device=device)

    a_k = torch.ones(r, device=device)
    h_k = torch.ones(r, device=device)

    h = _resolve_init(h_init, n, device, "h_init")
    a = _resolve_init(a_init, m, device, "a_init")
    w = _resolve_init(w_init, r, device, "w_init")

    # Step 2: initialise lambda
    lambda_prev = 0.0

    # Step 3: iterate until convergence
    while True:
        # Steps 4-5: HITS for users layer
        h_U = torch.mv(mu, a_U)
        a_U = torch.mv(mu.T, h_U)

        # Steps 6-7: HITS for items layer (computed per pseudocode; h_I/a_I not used in score update)
        h_I = torch.mv(mi, a_I)
        a_I = torch.mv(mi.T, h_I)

        # Steps 8-9: HITS for keywords layer
        h_k = torch.mv(mw, a_k)
        a_k = torch.mv(mw.T, h_k)

        # Step 10: h^(t+1) = A ×₂ a^(t) ×₃ w^(t) + h_U^(t+1) + a_U^(t+1)
        tensor_part = torch.tensordot(T, a, dims=([1], [0]))      # (N, R)
        tensor_part = torch.tensordot(tensor_part, w, dims=([1], [0]))  # (N,)
        h = tensor_part + h_U + a_U

        # Steps 11-12: compute T ×₁ h^(t+1) once, reuse for a and w updates
        # a^(t+1) = A ×₁ h^(t+1) ×₃ w^(t)
        # w^(t+1) = A ×₁ h^(t+1) ×₂ a^(t) + a_k^(t+1)   [uses old a per paper step 12]
        a_prev = a
        Th = torch.tensordot(T, h, dims=([0], [0]))               # (M, R)
        a = torch.tensordot(Th, w, dims=([1], [0]))                # (M,)
        w = torch.tensordot(Th, a_prev, dims=([0], [0])) + a_k    # (R,)

        # Step 13: λ = ||h|| ||a|| ||w||  (before normalisation)
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

        # Steps 15-16: check convergence (paper: if λ₁ − λ ≤ ε)
        if lambda_current - lambda_prev <= epsilon:
            break

        lambda_prev = lambda_current

    # Step 19: return rank-1 approximation
    return h.unsqueeze(0), a.unsqueeze(0), w.unsqueeze(0)
