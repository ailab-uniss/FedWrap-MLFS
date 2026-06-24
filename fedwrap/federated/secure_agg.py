"""Additive pairwise-masking secure aggregation for the count-only wrapper protocol.

The server only ever needs the *sum* of the clients' label-wise counts (TP/FP/FN), so it does not need
to see any individual client's vector. We hide each client's contribution with pairwise additive masks
that cancel in the sum (the classic one-time-pad / Bonawitz-style construction), evaluated over the
ring Z_{2^64} so cancellation is exact:

    masked_i = count_i + sum_{j>i} PRG(s_ij) - sum_{j<i} PRG(s_ij)   (mod 2^64)

For any pair (i,j) the two masks are added by one client and subtracted by the other, so
``sum_i masked_i = sum_i count_i (mod 2^64)``. Because the true total is far below 2^64, the modular
sum equals the integer sum exactly---hence the reconstructed global TP/FP/FN, and the macro/micro-F1
computed from them, are *bitwise identical* to the non-secure protocol. Each masked vector on its own
is uniform over the ring, so the coordinator learns nothing about an individual client beyond the sum.

Scope (kept deliberately honest): this protects per-client contributions against an honest-but-curious
coordinator for one aggregation. We do not implement dropout recovery (the scheduler's mass quorum
already tolerates non-responders by aggregating over the responding subset) and we do not claim
record-level privacy---that is the orthogonal differential-privacy add-on. The shared seeds s_ij stand
in for a key-agreement step (e.g. Diffie--Hellman) done once per federation.
"""
from __future__ import annotations

import numpy as np

_MOD = np.uint64  # arithmetic over Z_{2^64} via native uint64 wraparound


def pair_seed(base: int, i: int, j: int) -> int:
    """Deterministic shared seed for the unordered pair {i,j} (stands in for DH key agreement)."""
    a, b = (i, j) if i < j else (j, i)
    return (int(base) * 1_000_003 + a * 1009 + b) & 0xFFFFFFFFFFFFFFFF


def _prg(seed: int, dim: int) -> np.ndarray:
    """Pseudo-random mask vector over Z_{2^64} from a shared seed."""
    return np.random.default_rng(seed).integers(0, 2**64, size=dim, dtype=np.uint64)


def mask_client(vec: np.ndarray, i: int, K: int, base_seed: int) -> np.ndarray:
    """Mask client ``i``'s non-negative integer count vector so the masks cancel in the K-party sum."""
    out = vec.astype(np.uint64, copy=True)
    for j in range(K):
        if j == i:
            continue
        m = _prg(pair_seed(base_seed, i, j), vec.shape[0])
        if j > i:
            out = out + m       # uint64 wraparound = mod 2^64
        else:
            out = out - m
    return out


def secure_sum(vectors: list[np.ndarray], base_seed: int = 0) -> np.ndarray:
    """Aggregate per-client count vectors through pairwise masking; returns the exact integer sum.

    ``vectors`` are non-negative integer arrays (e.g. concatenated TP/FP/FN for a batch of candidates).
    The result equals ``sum(vectors)`` exactly, but the server only ever adds the *masked* vectors."""
    K = len(vectors)
    dim = int(vectors[0].shape[0])
    acc = np.zeros(dim, dtype=np.uint64)
    for i, v in enumerate(vectors):
        acc = acc + mask_client(np.asarray(v), i, K, base_seed)   # server sees only masked vectors
    return acc.astype(np.int64)   # masks cancelled; true total fits well below 2^64
