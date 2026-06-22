"""
Cold-start initialisation helpers for the influencer tensor model.

Builds warm-start prior vectors for `socialAU`'s h_init/a_init/w_init
arguments from text content (user bios, item descriptions, keyword
definitions, ...) via a caller-supplied embedding function, so the package
itself stays model-agnostic: `embed_fn` may wrap a sentence-transformer, an
embedding API call, or any custom encoder.
"""

from typing import Callable

import torch


def _first_principal_component(X: torch.Tensor, n_iter: int = 100, seed: int = 42) -> torch.Tensor:
    """
    Project each row of `X` (N, D) onto the first principal component of
    the (centred) data, via power iteration on the covariance matrix. No
    sklearn dependency.

    Returns
    -------------------
    torch.Tensor. Shape (N,).
    """
    N, D = X.shape
    centered = X - X.mean(dim=0, keepdim=True)
    cov = centered.T @ centered  # (D, D)

    generator = torch.Generator(device=X.device).manual_seed(seed)
    v = torch.randn(D, generator=generator, device=X.device)
    v = v / torch.linalg.norm(v).clamp_min(1e-12)

    for _ in range(n_iter):
        v_new = cov @ v
        norm = torch.linalg.norm(v_new)
        if norm <= 1e-12:
            break
        v = v_new / norm

    return centered @ v


def make_embedding_init(
    texts: list[str],
    embed_fn: Callable[[list[str]], torch.Tensor],
    aggregation: str = "mean",
) -> torch.Tensor:
    """
    Build a warm-start initial vector for `socialAU`'s h_init/a_init/w_init
    from text content, via a caller-supplied embedding function.

    Parameters
    -------------------
    texts: list[str]. One text per node (user bio, item description,
        keyword definition, ...), length N.
    embed_fn: Callable[[list[str]], torch.Tensor]. Maps `texts` to a dense
        (N, D) embedding tensor. Could wrap a sentence-transformer, an
        embedding API call, or a custom model — the package itself never
        depends on a specific embedding library.
    aggregation: str, default "mean". One of:
        - "mean": reduce the (N, D) embeddings to (N,) by projecting onto
          their first principal component (power iteration, no sklearn).
        - "norm": L2-normalise each embedding row and return the (N, D)
          matrix as-is; the caller is responsible for projecting it to R^N.

    Returns
    -------------------
    torch.Tensor. float32. Shape (N,) for "mean", shape (N, D) for "norm".
    """
    if aggregation not in ("mean", "norm"):
        raise ValueError(f"aggregation must be 'mean' or 'norm', got {aggregation!r}")

    embeddings = embed_fn(texts).float()

    if aggregation == "norm":
        norms = embeddings.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return (embeddings / norms).float()

    return _first_principal_component(embeddings).float()
