"""Flat binary-mask genotype used by FedWrap-MLFS.

A candidate is a flat feature mask ``m in {0,1}^D``. The federation-aware operators that drive the
search (disagreement-guided mutation, client-stability ranking) live in
:mod:`fedwrap.fedaware`; this module provides the encoding and its plain genetic operators.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BitstringConfig:
    """Parameters for the flat binary-mask genotype."""
    init_prob: float = 0.1            # probability each feature is ON at init
    bitflip_prob: float = 0.01        # symmetric per-bit flip probability
    bitflip_prob_on: float | None = None   # asymmetric 1→0 flip (overrides bitflip_prob)
    bitflip_prob_off: float | None = None  # asymmetric 0→1 flip (overrides bitflip_prob)


def init_bitstring(n_features: int, cfg: BitstringConfig, rng: np.random.Generator) -> np.ndarray:
    """Create a random binary mask with P(bit=1) = init_prob."""
    mask = rng.random(n_features) < float(cfg.init_prob)
    if mask.sum() == 0:
        mask[rng.integers(0, n_features)] = True
    return mask


def bitstring_crossover(
    a: np.ndarray, b: np.ndarray, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Uniform crossover: each bit is independently taken from either parent."""
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    m = rng.random(a.size) < 0.5
    c1, c2 = np.where(m, a, b), np.where(m, b, a)
    for c in (c1, c2):
        if c.sum() == 0:
            cands = np.flatnonzero(a | b)
            idx = int(rng.choice(cands)) if cands.size else int(rng.integers(0, a.size))
            c[idx] = True
    return c1, c2


def bitstring_mutate(a: np.ndarray, cfg: BitstringConfig, rng: np.random.Generator) -> np.ndarray:
    """Bit-flip mutation with optional asymmetric probabilities."""
    a = np.asarray(a, dtype=bool).copy()
    p_on = float(cfg.bitflip_prob_on if cfg.bitflip_prob_on is not None else cfg.bitflip_prob)
    p_off = float(cfg.bitflip_prob_off if cfg.bitflip_prob_off is not None else cfg.bitflip_prob)
    r = rng.random(a.size)
    flips = (a & (r < p_on)) | ((~a) & (r < p_off))
    a ^= flips.astype(bool, copy=False)
    if a.sum() == 0:
        a[rng.integers(0, a.size)] = True
    return a
