import time
import io
import contextlib
import torch
from influencer import tophits, socialAU

torch.manual_seed(99)


def make_random_inputs(n, m, r, density=0.01):
    MU = torch.bernoulli(torch.full((n, n), density))
    MI = torch.bernoulli(torch.full((m, m), density))
    MK = torch.bernoulli(torch.full((r, r), density))
    mask = torch.bernoulli(torch.full((n, m, r), density))
    T = mask * torch.randint(1, 6, (n, m, r)).float()
    return MU, MI, MK, T


def run_timed(fn, *args, **kwargs):
    """Call fn, suppress stdout (hides 'GPU not available'), return elapsed seconds."""
    with contextlib.redirect_stdout(io.StringIO()):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - t0
    return result, elapsed


SIZES = [(10, 8, 12), (50, 40, 60), (200, 150, 250), (500, 400, 600)]

print("Running benchmarks ...")
rows = []
for n, m, r in SIZES:
    MU, MI, MK, T = make_random_inputs(n, m, r)
    _, t_th  = run_timed(tophits,  T, epsilon=1e-3)
    _, t_sau = run_timed(socialAU, MU, MI, MK, T, epsilon=1e-3)
    ratio = t_sau / t_th if t_th > 0 else float("inf")
    rows.append((n, m, r, t_th, t_sau, ratio))
    print(f"  ({n:>3},{m:>3},{r:>3})  tophits={t_th:.3f}s  socialAU={t_sau:.3f}s  overhead={ratio:.1f}x")

print()
hdr = f"{'n':>5} | {'m':>5} | {'r':>5} | {'TOPHITS (s)':>11} | {'SocialAU (s)':>12} | {'Overhead':>8}"
sep = f"{'-----':>5} | {'-----':>5} | {'-----':>5} | {'-----------':>11} | {'------------':>12} | {'--------':>8}"
print(hdr)
print(sep)
for n, m, r, t_th, t_sau, ratio in rows:
    print(f"{n:>5} | {m:>5} | {r:>5} | {t_th:>11.3f} | {t_sau:>12.3f} | {ratio:>7.1f}x")

# SocialAU overhead vs TOPHITS comes from the 3 HITS matrix-vector products per iteration.
# On sparse random tensors SocialAU often converges in fewer iterations than TOPHITS,
# so total wall-clock time can be lower despite the higher per-iteration cost.
