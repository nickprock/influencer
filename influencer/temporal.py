"""
Temporal extension of the influencer tensor model.

Builds a dynamic (N, M, R) tensor from a stream of timestamped
(user, item, keyword) events, applying exponential decay to older
events, then reuses `socialAU` for centrality scoring.
"""

from datetime import datetime
from typing import Any

import torch

from .torch_centrality import socialAU

_SECONDS_PER_UNIT = {
    "seconds": 1,
    "minutes": 60,
    "hours": 3600,
    "days": 86400,
    "weeks": 604800,
}


def _age_in_units(reference_time: Any, timestamp: Any, time_unit: str) -> float:
    """Age of `timestamp`, in `time_unit`s before `reference_time`."""
    if isinstance(timestamp, datetime):
        delta = reference_time - timestamp
        return delta.total_seconds() / _SECONDS_PER_UNIT[time_unit]
    return float(reference_time - timestamp)


def build_temporal_tensor(
    triples: list,
    n: int, m: int, r: int,
    window: int = 7,
    decay: float = 0.9,
    reference_time: Any = None,
    time_unit: str = "days",
) -> torch.Tensor:
    """
    Build a dense (N, M, R) tensor from timestamped events, decaying
    contributions from older events.

    Parameters
    -------------------
    triples: list of (user_id, item_id, keyword_id, timestamp) tuples.
        `timestamp` is either a `datetime.datetime` or a numeric value
        expressed in `time_unit`s since epoch.
    n, m, r: int. Output tensor dimensions.
    window: int, default 7. Number of buckets. Events whose age is
        >= window `time_unit`s before `reference_time` are ignored.
    decay: float, default 0.9. Per-bucket decay factor.
    reference_time: default None. The "now" instant buckets are measured
        from. Defaults to the maximum timestamp found in `triples`.
    time_unit: str, default "days". One of "seconds", "minutes", "hours",
        "days", "weeks". Only used to convert `datetime` timestamps into
        a unit-less age; numeric timestamps are assumed to already be
        expressed in this unit.

    Notes
    -------------------
    An event's age (in `time_unit`s before `reference_time`) is floored
    to obtain its bucket index: bucket 0 covers the most recent unit
    (age in [0, 1)), bucket 1 the unit before that (age in [1, 2)), and
    so on up to bucket (window - 1). Events at or after `reference_time`
    are clamped to bucket 0. An event in bucket t contributes `decay**t`
    to T[user_id, item_id, keyword_id].

    Returns
    -------------------
    T: torch.Tensor. Dense float32 tensor of shape (n, m, r).
    """
    if time_unit not in _SECONDS_PER_UNIT:
        raise ValueError(
            f"time_unit must be one of {sorted(_SECONDS_PER_UNIT)}, got {time_unit!r}"
        )

    if reference_time is None:
        reference_time = max(timestamp for _, _, _, timestamp in triples)

    T = torch.zeros((n, m, r), dtype=torch.float32)

    for user_id, item_id, keyword_id, timestamp in triples:
        age = _age_in_units(reference_time, timestamp, time_unit)
        if age < 0:
            age = 0.0

        bucket = int(age)
        if bucket >= window:
            continue

        T[user_id, item_id, keyword_id] += decay ** bucket

    return T


def socialAU_temporal(
    mu: torch.Tensor, mi: torch.Tensor, mw: torch.Tensor,
    triples: list,
    n: int, m: int, r: int,
    window: int = 7, decay: float = 0.9,
    reference_time: Any = None, time_unit: str = "days",
    epsilon: float = 0.001, device=0,
):
    """
    Build a temporal tensor from timestamped events and run `socialAU` on it.

    Parameters
    -------------------
    mu, mi, mw: torch tensor. Adjacency matrices, as in `socialAU`.
    triples, n, m, r, window, decay, reference_time, time_unit: see
        `build_temporal_tensor`.
    epsilon, device: see `socialAU`.

    Returns
    -------------------
    u, v, w: torch tensor. Identical in shape to `socialAU`'s output.
    """
    T = build_temporal_tensor(
        triples, n, m, r,
        window=window, decay=decay,
        reference_time=reference_time, time_unit=time_unit,
    )
    return socialAU(mu, mi, mw, T, epsilon=epsilon, device=device)
