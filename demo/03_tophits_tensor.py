import torch
from influencer import tophits
from influencer.torch_centrality import safe_normalize

torch.manual_seed(0)

# Build a rank-2 tensor as a weighted sum of two outer products
u1 = torch.tensor([1., 0., 0., 0., 0.]); v1 = torch.tensor([1., 0., 0., 0.]); w1 = torch.tensor([1., 0., 0.])
u2 = torch.tensor([0., 1., 0., 0., 0.]); v2 = torch.tensor([0., 1., 0., 0.]); w2 = torch.tensor([0., 1., 0.])
T = 10 * torch.einsum('i,j,k->ijk', u1, v1, w1) + 3 * torch.einsum('i,j,k->ijk', u2, v2, w2)
print(f"T shape: {list(T.shape)}  |  T[0,0,0]={T[0,0,0].item():.0f} (dominant)  T[1,1,1]={T[1,1,1].item():.0f} (weak)\n")

# Run TOPHITS with tight convergence and print recovered vectors
u, v, w = tophits(T, epsilon=1e-8)
print("Recovered vectors (epsilon=1e-8):")
print(f"  u = {u.tolist()}")
print(f"  v = {v.tolist()}")
print(f"  w = {w.tolist()}\n")

# Compute cosine alignment with the dominant direction (u1, v1, w1 are already unit vectors)
cos_u = torch.dot(u, safe_normalize(u1)).item()
cos_v = torch.dot(v, safe_normalize(v1)).item()
cos_w = torch.dot(w, safe_normalize(w1)).item()
print("Cosine alignment with dominant direction [u1, v1, w1]:")
print(f"  cos(u, u1) = {cos_u:.6f}")
print(f"  cos(v, v1) = {cos_v:.6f}")
print(f"  cos(w, w1) = {cos_w:.6f}")
assert cos_u > 0.99, f"u alignment too low: {cos_u:.4f}"
assert cos_v > 0.99, f"v alignment too low: {cos_v:.4f}"
assert cos_w > 0.99, f"w alignment too low: {cos_w:.4f}"
print("  All alignments > 0.99 -- dominant component recovered correctly.\n")


def tophits_counted(T: torch.Tensor, epsilon: float, max_iter: int = 1000):
    """Mirror of tophits() that also returns the iteration count and lambda history."""
    u = safe_normalize(torch.ones(T.shape[0]))
    v = safe_normalize(torch.ones(T.shape[1]))
    w = safe_normalize(torch.ones(T.shape[2]))
    lambda_prev, iters, lam_history = 0.0, 0, []
    for iters in range(1, max_iter + 1):
        uv  = torch.tensordot(T, v,    dims=([1], [0]))
        u_n = torch.tensordot(uv, w,   dims=([1], [0]))
        Tu  = torch.tensordot(T, u_n,  dims=([0], [0]))
        v_n = torch.tensordot(Tu, w,   dims=([1], [0]))
        w_n = torch.tensordot(Tu, v_n, dims=([0], [0]))
        lam = (torch.linalg.norm(u_n) * torch.linalg.norm(v_n) * torch.linalg.norm(w_n)).item()
        lam_history.append((iters, lam, lam - lambda_prev))
        u, v, w = safe_normalize(u_n), safe_normalize(v_n), safe_normalize(w_n)
        if lam - lambda_prev < epsilon:
            break
        lambda_prev = lam
    return u, v, w, iters, lam_history


# Compare iteration counts for tight vs loose epsilon and show lambda trace
_, _, _, iters_tight, history = tophits_counted(T, epsilon=1e-8)
_, _, _, iters_loose, _       = tophits_counted(T, epsilon=0.5)

print("Lambda trace (epsilon=1e-8):")
print(f"  {'iter':>4}  {'lambda':>12}  {'delta':>12}")
for it, lam, delta in history:
    print(f"  {it:>4}  {lam:>12.6f}  {delta:>12.6f}")

print(f"\nConvergence: epsilon=1e-8 -> {iters_tight} iters | epsilon=0.5 -> {iters_loose} iters")
print("Note: this orthogonal tensor converges so fast the delta drops below 0.5 by the same")
print("      iteration as 1e-8. To demonstrate early stopping we force max_iter=1 instead:\n")

u_early, v_early, w_early, iters_forced, _ = tophits_counted(T, epsilon=1e-8, max_iter=1)
print(f"  max_iter=1 (forced early stop) -- iters={iters_forced}")
print(f"  cos(u_early, u1) = {torch.dot(u_early, safe_normalize(u1)).item():.6f}  (vs 1.000000 at convergence)")
print(f"  cos(v_early, v1) = {torch.dot(v_early, safe_normalize(v1)).item():.6f}")
print(f"  cos(w_early, w1) = {torch.dot(w_early, safe_normalize(w1)).item():.6f}\n")

print(
    "Lambda convergence in the PARAFAC context\n"
    "------------------------------------------\n"
    "In TOPHITS, lambda is computed at each step as ||u|| * ||v|| * ||w|| before\n"
    "normalisation. It approximates the leading singular value of the tensor -- the scale\n"
    "of the rank-1 component u(x)v(x)w that best fits T. Each iteration contracts T along\n"
    "two modes to update the third vector, converging toward the dominant PARAFAC factor.\n"
    "Lambda grows until the rank-1 approximation stops improving: once the vectors are\n"
    "well aligned with the true dominant direction, successive contractions barely change\n"
    "their norms, so lambda(t) - lambda(t-1) drops below epsilon and the algorithm halts.\n"
    "A tight epsilon (1e-8) forces near-exact alignment; a loose epsilon (0.5) stops as\n"
    "soon as the coarse direction is found. For this tensor the dominant signal (weight 10)\n"
    "is well-separated from the secondary one (weight 3), so convergence is extremely fast."
)
