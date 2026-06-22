"""
Sparse tensor support for the influencer package.

Provides sparse-compatible variants of `tophits` and `socialAU` that exploit
the sparsity of the user-item-keyword tensor T, avoiding the memory cost of
materialising T as a dense tensor for large real-world networks.
"""

import torch

from .torch_centrality import safe_normalize, tophits, socialAU


def to_sparse_tensor(
    indices: torch.LongTensor,
    values: torch.FloatTensor,
    size: tuple,
) -> torch.Tensor:
    """
    Build a coalesced sparse COO tensor from explicit indices and values.

    Parameters
    ----------
    indices : torch.LongTensor
        Shape (3, nnz). Rows are (user_ids, item_ids, keyword_ids).
    values : torch.FloatTensor
        Shape (nnz,). Non-zero values.
    size : tuple[int, int, int]
        Dense shape (N, M, Q) of the tensor.

    Returns
    -------
    torch.Tensor
        A coalesced sparse COO tensor of shape `size`.
    """
    return torch.sparse_coo_tensor(indices, values, size=size).coalesce()


def _build_unfold(T_sparse: torch.Tensor, mode: int):
    """
    Build the mode-`mode` unfolding of a sparse 3D tensor, already
    transposed to shape (other_size, contract_size) so contracting it
    against a vector is a direct `torch.sparse.mm` with no extra
    transpose.

    This unfolding depends only on `T_sparse` and `mode`, never on the
    vector being contracted against, so callers that contract the same
    mode repeatedly across iterations (e.g. power iteration in
    `tophits_sparse`/`socialAU_sparse`) should build it once via this
    helper and reuse it, instead of paying its index-arithmetic and
    coalesce cost on every iteration.

    Returns
    -------
    unfold_t : torch.Tensor
        Coalesced sparse matrix of shape (other_size, contract_size).
    out_shape : tuple[int, int]
        Shape to reshape a contraction result into: (M, Q) for mode=0,
        (N, Q) for mode=1, or (N, M) for mode=2.
    """
    if mode not in (0, 1, 2):
        raise ValueError(f"mode must be 0, 1, or 2, got {mode}")

    T_sparse = T_sparse.coalesce()
    N, M, Q = T_sparse.shape
    indices = T_sparse.indices()
    values = T_sparse.values()
    i0, i1, i2 = indices[0], indices[1], indices[2]

    if mode == 0:
        contract_idx, other_idx = i0, i1 * Q + i2
        contract_size, out_shape = N, (M, Q)
    elif mode == 1:
        contract_idx, other_idx = i1, i0 * Q + i2
        contract_size, out_shape = M, (N, Q)
    else:  # mode == 2
        contract_idx, other_idx = i2, i0 * M + i1
        contract_size, out_shape = Q, (N, M)

    other_size = out_shape[0] * out_shape[1]

    unfold_t = torch.sparse_coo_tensor(
        torch.stack([other_idx, contract_idx]),
        values,
        size=(other_size, contract_size),
        device=T_sparse.device,
    ).coalesce()

    return unfold_t, out_shape


def _apply_unfold(unfold_t: torch.Tensor, out_shape: tuple, vec: torch.Tensor) -> torch.Tensor:
    """Contract a precomputed unfolding (from `_build_unfold`) against `vec`."""
    vec = vec.to(device=unfold_t.device, dtype=unfold_t.dtype)
    result = torch.sparse.mm(unfold_t, vec.reshape(-1, 1)).squeeze(1)
    return result.reshape(out_shape)


def sparse_mode_product(
    T_sparse: torch.Tensor,
    vec: torch.Tensor,
    mode: int,
) -> torch.Tensor:
    """
    Contract a sparse 3D tensor along `mode` with a dense vector.

    Parameters
    ----------
    T_sparse : torch.Tensor
        Sparse COO tensor of shape (N, M, Q).
    vec : torch.Tensor
        Dense vector matching the size of dimension `mode`.
    mode : int
        Dimension to contract: 0, 1, or 2.

    Returns
    -------
    torch.Tensor
        Dense tensor of shape (M, Q) for mode=0, (N, Q) for mode=1,
        or (N, M) for mode=2.

    Notes
    -----
    Rebuilds the mode unfolding from scratch on every call. Callers that
    contract the same mode repeatedly with different vectors (power
    iteration) should use `_build_unfold`/`_apply_unfold` directly to
    build the unfolding once and reuse it.
    """
    unfold_t, out_shape = _build_unfold(T_sparse, mode)
    return _apply_unfold(unfold_t, out_shape, vec)


def tophits_sparse(T: torch.Tensor, epsilon: float = 1e-3, max_iter: int = 1000, device=0):
    """
    Sparse-compatible variant of `tophits`. Falls back to the dense
    implementation when T is not sparse.
    """
    if not T.is_sparse:
        return tophits(T, epsilon=epsilon, max_iter=max_iter, device=device)

    if not torch.cuda.is_available():
        device = "cpu"
        print("GPU not available")

    device = torch.device(device)
    T = T.to(device)

    N, M, Q = T.shape

    u = safe_normalize(torch.ones(N, device=device))
    v = safe_normalize(torch.ones(M, device=device))
    w = safe_normalize(torch.ones(Q, device=device))

    # Mode-0/mode-1 unfoldings depend only on T, not on the per-iteration
    # vectors below, so build each once and reuse it across iterations
    # instead of re-deriving and re-coalescing it on every call.
    mode1_unfold = _build_unfold(T, mode=1)
    mode0_unfold = _build_unfold(T, mode=0)

    lambda_prev = 0.0

    for _ in range(max_iter):
        # Update u: T ×₂ v ×₃ w
        uv = _apply_unfold(*mode1_unfold, v)               # (N, Q)
        u_new = torch.mv(uv, w)                            # (N,)

        # Compute T ×₁ u_new once and reuse for both v and w updates
        Tu = _apply_unfold(*mode0_unfold, u_new)            # (M, Q)

        # Update v: T ×₁ u_new ×₃ w
        v_new = torch.mv(Tu, w)                              # (M,)

        # Update w: T ×₁ u_new ×₂ v_new  (reuses Tu)
        w_new = torch.mv(Tu.t(), v_new)                      # (Q,)

        lambda_curr = (
            torch.linalg.norm(u_new) *
            torch.linalg.norm(v_new) *
            torch.linalg.norm(w_new)
        ).item()

        u = safe_normalize(u_new)
        v = safe_normalize(v_new)
        w = safe_normalize(w_new)

        if lambda_curr - lambda_prev < epsilon:
            break

        lambda_prev = lambda_curr

    return u, v, w


def socialAU_sparse(mu, mi, mw, T, epsilon: float = 0.001, device=0):
    """
    Sparse-compatible variant of `socialAU`. Falls back to the dense
    implementation when T is not sparse.
    """
    if not T.is_sparse:
        return socialAU(mu, mi, mw, T, epsilon=epsilon, device=device)

    if not torch.cuda.is_available():
        device = "cpu"
        print("GPU not available")

    device = torch.device(device)
    mu = mu.to(device)
    mi = mi.to(device)
    mw = mw.to(device)
    T = T.to(device)

    n, m, r = T.shape

    a_U = torch.ones(n, device=device)
    h_U = torch.ones(n, device=device)

    a_I = torch.ones(m, device=device)
    h_I = torch.ones(m, device=device)

    a_k = torch.ones(r, device=device)
    h_k = torch.ones(r, device=device)

    h = torch.ones(n, device=device)
    a = torch.ones(m, device=device)
    w = torch.ones(r, device=device)

    # See `tophits_sparse`: these unfoldings depend only on T, so build
    # them once and reuse across iterations rather than per-call.
    mode1_unfold = _build_unfold(T, mode=1)
    mode0_unfold = _build_unfold(T, mode=0)

    lambda_prev = 0.0

    while True:
        h_U = torch.mv(mu, a_U)
        a_U = torch.mv(mu.T, h_U)

        h_I = torch.mv(mi, a_I)
        a_I = torch.mv(mi.T, h_I)

        h_k = torch.mv(mw, a_k)
        a_k = torch.mv(mw.T, h_k)

        # Step 10: h^(t+1) = A ×₂ a^(t) ×₃ w^(t) + h_U^(t+1) + a_U^(t+1)
        tensor_part = _apply_unfold(*mode1_unfold, a)       # (N, R)
        tensor_part = torch.mv(tensor_part, w)               # (N,)
        h = tensor_part + h_U + a_U

        # Steps 11-12: compute T ×₁ h^(t+1) once, reuse for a and w updates
        a_prev = a
        Th = _apply_unfold(*mode0_unfold, h)                 # (M, R)
        a = torch.mv(Th, w)                                   # (M,)
        w = torch.mv(Th.t(), a_prev) + a_k                    # (R,)

        lambda_current = (
            torch.linalg.norm(h) * torch.linalg.norm(a) * torch.linalg.norm(w)
        ).item()

        h = safe_normalize(h)
        a = safe_normalize(a)
        w = safe_normalize(w)

        a_U = safe_normalize(a_U)
        h_U = safe_normalize(h_U)
        a_I = safe_normalize(a_I)
        h_I = safe_normalize(h_I)
        a_k = safe_normalize(a_k)
        h_k = safe_normalize(h_k)

        if lambda_current - lambda_prev <= epsilon:
            break

        lambda_prev = lambda_current

    return h.unsqueeze(0), a.unsqueeze(0), w.unsqueeze(0)
