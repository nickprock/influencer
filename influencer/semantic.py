"""
Semantic keyword layer construction for the influencer tensor model.

Builds a keyword-keyword adjacency matrix (the `mw` argument of `socialAU`)
from pre-computed keyword embeddings via cosine similarity, instead of
co-occurrence counts. Also provides k-means clustering of keywords (to
collapse a large vocabulary into fewer, denser effective keywords) and a
helper to remap a user-item-keyword tensor onto the resulting clusters.
"""

import torch

# Row-chunk size used when building the (R, R) similarity matrix, so the
# per-chunk topk/threshold work never materialises more than one
# (chunk, R) block at a time, even for R up to 50,000.
_DEFAULT_CHUNK_SIZE = 4096


def build_semantic_keyword_matrix(
    embeddings: torch.Tensor,
    threshold: float = 0.7,
    top_k: int | None = None,
) -> torch.Tensor:
    """
    Build a keyword-keyword adjacency matrix from cosine similarity of
    pre-computed keyword embeddings.

    Parameters
    -------------------
    embeddings: torch.Tensor. Shape (R, D), one embedding per keyword.
    threshold: float, default 0.7. Keep MK[i, j] = sim[i, j] when
        sim[i, j] >= threshold, else 0. Ignored when `top_k` is set.
    top_k: int or None, default None. When set, keep for each keyword
        only its `top_k` most similar neighbours (the rest are zeroed).
        Takes precedence over `threshold`.

    Notes
    -------------------
    The (R, R) similarity matrix is built via batched matrix
    multiplication in row chunks, so no Python loop over keyword pairs
    is required even for R up to 50,000.

    Returns
    -------------------
    MK: torch.Tensor. Symmetric float32 tensor of shape (R, R) with a
        zero diagonal (no self-loops).
    """
    R = embeddings.shape[0]
    device = embeddings.device

    if R == 0:
        return torch.zeros((0, 0), dtype=torch.float32, device=device)

    embeddings = embeddings.float()
    norms = embeddings.norm(dim=1, keepdim=True).clamp_min(1e-12)
    normalized = embeddings / norms

    MK = torch.zeros((R, R), dtype=torch.float32, device=device)
    chunk_size = min(_DEFAULT_CHUNK_SIZE, R)
    k = min(top_k, R - 1) if top_k is not None else None

    for start in range(0, R, chunk_size):
        end = min(start + chunk_size, R)

        # Batched matmul: (chunk, D) @ (D, R) -> (chunk, R)
        sim_chunk = (normalized[start:end] @ normalized.T).clamp(-1.0, 1.0)

        # Exclude self-similarity from neighbour selection / thresholding.
        rows = torch.arange(end - start, device=device)
        sim_chunk[rows, torch.arange(start, end, device=device)] = float("-inf")

        if k is not None:
            if k > 0:
                _, idx = torch.topk(sim_chunk, k, dim=1)
                mask = torch.zeros_like(sim_chunk, dtype=torch.bool)
                mask.scatter_(1, idx, True)
                sim_chunk = sim_chunk.masked_fill(~mask, 0.0)
            else:
                sim_chunk = torch.zeros_like(sim_chunk)
        else:
            sim_chunk = torch.where(sim_chunk >= threshold, sim_chunk, torch.zeros_like(sim_chunk))

        MK[start:end] = sim_chunk

    # top_k neighbour selection can be asymmetric (i picks j without j
    # picking i); symmetrize so the result is always usable as an
    # undirected adjacency matrix. A no-op for the threshold path, since
    # the underlying similarity matrix is already symmetric.
    MK = torch.maximum(MK, MK.T)
    MK.fill_diagonal_(0.0)
    return MK


def cluster_keywords(
    embeddings: torch.Tensor,
    n_clusters: int,
    n_iter: int = 100,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run k-means clustering on keyword embeddings, using only torch.

    Parameters
    -------------------
    embeddings: torch.Tensor. Shape (R, D), one embedding per keyword.
    n_clusters: int. Number of clusters, must be in [1, R].
    n_iter: int, default 100. Maximum number of Lloyd's-algorithm
        iterations. Stops early once assignments stop changing.
    seed: int, default 42. Seed for the k-means++ initialisation, so
        results are reproducible.

    Returns
    -------------------
    labels: torch.LongTensor. Shape (R,), cluster assignment per keyword.
    centroids: torch.FloatTensor. Shape (n_clusters, D).
    """
    R, D = embeddings.shape
    device = embeddings.device

    if not (1 <= n_clusters <= R):
        raise ValueError(f"n_clusters must be in [1, {R}], got {n_clusters}")

    embeddings = embeddings.float()
    generator = torch.Generator(device=device).manual_seed(seed)

    def sq_dists(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_sq = (x * x).sum(dim=1, keepdim=True)
        y_sq = (y * y).sum(dim=1)
        return (x_sq + y_sq - 2.0 * (x @ y.T)).clamp_min(0.0)

    # k-means++ initialisation: pick the first centroid uniformly, then
    # each subsequent one with probability proportional to its squared
    # distance from the nearest centroid chosen so far.
    centroids = torch.empty((n_clusters, D), dtype=torch.float32, device=device)
    first_idx = torch.randint(0, R, (1,), generator=generator, device=device).item()
    centroids[0] = embeddings[first_idx]
    closest_sq_dist = sq_dists(embeddings, centroids[:1]).squeeze(1)

    for c in range(1, n_clusters):
        weights = closest_sq_dist.clamp_min(1e-12)
        if weights.sum() <= 0:
            next_idx = torch.randint(0, R, (1,), generator=generator, device=device).item()
        else:
            next_idx = torch.multinomial(weights, 1, generator=generator).item()
        centroids[c] = embeddings[next_idx]
        new_dist = sq_dists(embeddings, centroids[c : c + 1]).squeeze(1)
        closest_sq_dist = torch.minimum(closest_sq_dist, new_dist)

    # Lloyd's algorithm: alternate nearest-centroid assignment (E-step)
    # and centroid recomputation (M-step) until labels stop changing.
    labels = torch.full((R,), -1, dtype=torch.long, device=device)
    ones = torch.ones(R, device=device)

    for _ in range(n_iter):
        new_labels = sq_dists(embeddings, centroids).argmin(dim=1)
        if torch.equal(new_labels, labels):
            break
        labels = new_labels

        counts = torch.zeros(n_clusters, device=device)
        counts.index_add_(0, labels, ones)

        new_centroids = torch.zeros_like(centroids)
        new_centroids.index_add_(0, labels, embeddings)

        nonempty = counts > 0
        new_centroids[nonempty] /= counts[nonempty].unsqueeze(1)
        new_centroids[~nonempty] = centroids[~nonempty]  # keep stale centroid, avoid NaN

        centroids = new_centroids

    return labels, centroids


def remap_tensor_to_clusters(
    T: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Collapse the keyword dimension of a user-item-keyword tensor onto
    keyword clusters, summing entries within each cluster.

    Parameters
    -------------------
    T: torch.Tensor. Shape (N, M, R).
    labels: torch.Tensor. Shape (R,), cluster assignment per keyword, as
        returned by `cluster_keywords`.

    Returns
    -------------------
    T_clustered: torch.Tensor. Shape (N, M, n_clusters), where
        T_clustered[u, i, c] = sum(T[u, i, k] for k where labels[k] == c).
    """
    N, M, R = T.shape
    if labels.shape != (R,):
        raise ValueError(f"labels must have shape ({R},), got {tuple(labels.shape)}")

    labels = labels.to(device=T.device, dtype=torch.long)
    n_clusters = int(labels.max().item()) + 1

    T_clustered = torch.zeros((N, M, n_clusters), dtype=T.dtype, device=T.device)
    T_clustered.index_add_(2, labels, T)
    return T_clustered
