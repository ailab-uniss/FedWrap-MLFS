"""FedAware-NSGA-II: federation-aware operators for the flat binary-mask wrapper search.

The genotype is a flat feature mask ``m in {0,1}^D``; the search itself is made federation-aware:

  * federated relevance sketch  R[l, j]: each client computes a local per-label feature
    relevance (class-conditional standardized mean difference) and the server aggregates the
    sufficient statistics into a global relevance matrix -- no raw data leaves a client.
  * disagreement-guided mutation: globally-hard labels (low population F1) bias mutation toward
    turning ON features with high relevance for those labels and OFF features with low aggregate
    relevance, instead of flipping bits uniformly.
  * client-stability tie-break (applied in nsga2): among solutions of equal Pareto rank and
    similar crowding, prefer the one whose per-client macro-F1 has the smallest dispersion, so the
    selected subset does not work only on a dominant client.

These reuse the per-client / per-label sufficient statistics the federated evaluator already
returns; exact full-evaluation archive certification is handled by the evaluator and the NSGA-II loop.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .genotypes import BitstringConfig, bitstring_crossover, bitstring_mutate
from .nsga2 import Variation


@dataclass
class FedAwareConfig:
    enabled: bool = True
    stability_tiebreak: bool = True
    disagreement_mutation: bool = True
    disagreement_prob: float = 0.5        # P(a mutation call is disagreement-guided vs plain bitflip)
    relevance_pool: int = 20              # candidate pool size of top-relevance features per move
    hardness_temperature: float = 0.5     # softmax temperature over label hardness
    relevance_warmstart: bool = True      # seed part of the initial population from the relevance sketch
    warmstart_frac: float = 0.3           # fraction of the initial population built from the sketch
    warmstart_jitter: float = 0.10        # expected fraction of seeded features randomly perturbed
    filter_seed: bool = False             # also seed the population with strong federated-filter masks
    swap_prob: float = 0.0                # P(a mutation is a sparsity-preserving random swap)


def federated_relevance(clients, n_features: int, n_labels: int,
                        hard_client_weight: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate a per-label feature-relevance matrix R[L, D] from client-side sufficient
    statistics (class-conditional means). Each client contributes Y^T X (L x D), sum X (D),
    label counts (L) and n; the server forms the standardized mean-difference relevance. No raw
    features leave a client. Returns (R[L, D], global_relevance[D]).
    """
    YtX = np.zeros((n_labels, n_features), dtype=np.float64)
    sumX = np.zeros(n_features, dtype=np.float64)
    sumX2 = np.zeros(n_features, dtype=np.float64)
    pos = np.zeros(n_labels, dtype=np.float64)
    n_tot = 0
    for ci, c in enumerate(clients):
        X = c.x_train.tocsr(); Y = c.y_train.tocsr()
        w = 1.0 if hard_client_weight is None else float(hard_client_weight[ci])
        YtX += w * np.asarray((Y.T @ X).todense(), dtype=np.float64)
        Xd = np.asarray(X.sum(axis=0)).ravel()
        sumX += w * Xd
        sumX2 += w * np.asarray(X.multiply(X).sum(axis=0)).ravel()
        pos += w * np.asarray(Y.sum(axis=0)).ravel()
        n_tot += w * X.shape[0]
    n_tot = max(n_tot, 1.0)
    neg = np.maximum(n_tot - pos, 1.0)
    pos_safe = np.maximum(pos, 1.0)
    mean = sumX / n_tot
    var = np.maximum(sumX2 / n_tot - mean ** 2, 1e-9)
    sd = np.sqrt(var)
    pos_mean = YtX / pos_safe[:, None]                       # E[x_j | y_l=1]
    neg_mean = (sumX[None, :] - YtX) / neg[:, None]          # E[x_j | y_l=0]
    R = np.abs(pos_mean - neg_mean) / sd[None, :]            # standardized mean difference
    R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
    global_rel = R.mean(axis=0)
    return R.astype(np.float32), global_rel.astype(np.float32)


class FedAwareVariation(Variation):
    """Flat-mask variation with federation-aware (disagreement-guided) mutation.

    ``update_hardness`` is called once per generation with the population's per-label F1 to refocus
    mutation on labels the current front handles worst.
    """
    def __init__(self, bcfg: BitstringConfig, facfg: FedAwareConfig,
                 R: np.ndarray, global_rel: np.ndarray, n_labels: int):
        self.bcfg = bcfg
        self.fa = facfg
        self.R = R                     # (L, D) relevance
        self.global_rel = global_rel   # (D,)
        self.n_labels = int(n_labels)
        self.hard_weights = np.ones(n_labels, dtype=np.float64) / max(1, n_labels)
        self._rank = {int(l): np.argsort(-R[l]) for l in range(n_labels)}  # features by desc relevance

    def _coverage_mask(self, k: int, rng) -> np.ndarray:
        """Build a length-D mask of ~k features by round-robin over each label's most relevant
        features. Round-robin (rather than top-k of the global score) guarantees that rare/hard
        labels contribute their informative features, which a global ranking would drown out."""
        D = int(self.R.shape[1])
        k = int(np.clip(k, 1, D))
        chosen: list[int] = []
        seen = np.zeros(D, dtype=bool)
        pos = np.zeros(self.n_labels, dtype=np.int64)
        labs = np.arange(self.n_labels); rng.shuffle(labs)
        while len(chosen) < k:
            progressed = False
            for l in labs:
                order = self._rank[int(l)]
                while pos[l] < order.size:
                    f = int(order[int(pos[l])]); pos[l] += 1
                    if not seen[f]:
                        seen[f] = True; chosen.append(f); progressed = True; break
                if len(chosen) >= k:
                    break
            if not progressed:
                break
        m = np.zeros(D, dtype=bool); m[np.asarray(chosen[:k], dtype=np.int64)] = True
        # jitter: perturb a few bits so the seeded block keeps genetic diversity
        j = float(self.fa.warmstart_jitter)
        if j > 0.0:
            flip = rng.random(D) < (j * k / D)
            m ^= flip
        if not m.any():
            m[int(rng.integers(0, D))] = True
        return m

    def seed_population(self, pop_size: int, rng, max_ratio: float = 0.25) -> list[np.ndarray]:
        """Initial population: a ``warmstart_frac`` block of relevance-guided masks spanning
        densities up to ``max_ratio`` (log-uniform, so the sparse end of the Pareto front is
        covered), plus a random remainder built from the plain bitstring initialiser for
        exploration/diversity. The sketch is one-time federated information, so this adds no rounds."""
        from .genotypes import init_bitstring
        D = int(self.R.shape[1])
        n_seed = int(round(float(self.fa.warmstart_frac) * pop_size)) if self.fa.relevance_warmstart else 0
        n_seed = int(np.clip(n_seed, 0, pop_size))
        pop: list[np.ndarray] = []
        if n_seed > 0:
            lo = np.log(max(2.0, 0.02 * D)); hi = np.log(max(3.0, float(max_ratio) * D))
            ks = np.exp(np.linspace(lo, hi, n_seed))
            for k in ks:
                pop.append(self._coverage_mask(int(round(float(k))), rng))
        for _ in range(pop_size - n_seed):
            pop.append(np.asarray(init_bitstring(D, self.bcfg, rng), dtype=bool))
        rng.shuffle(pop)
        return pop

    def update_hardness(self, label_f1: np.ndarray) -> None:
        """Refocus on hard labels: weight ∝ softmax((1 - F1_l) / T)."""
        if label_f1 is None or label_f1.size != self.n_labels:
            return
        hard = 1.0 - np.clip(np.asarray(label_f1, dtype=np.float64), 0.0, 1.0)
        t = max(1e-3, float(self.fa.hardness_temperature))
        z = np.exp((hard - hard.max()) / t)
        self.hard_weights = z / max(z.sum(), 1e-12)

    def crossover(self, a, b, rng):
        return bitstring_crossover(np.asarray(a, bool), np.asarray(b, bool), rng)

    def _guided_swap(self, a, rng):
        """Relevance/hard-label-guided swap: add a high-relevance feature a hard label misses, drop a
        selected feature of lowest aggregate relevance (sparsity-neutral)."""
        m = a.copy()
        l = int(rng.choice(self.n_labels, p=self.hard_weights))
        pool = self._rank[l][: max(1, int(self.fa.relevance_pool))]
        cand = pool[~m[pool]]
        if cand.size:
            m[int(rng.choice(cand))] = True
        sel = np.flatnonzero(m)
        if sel.size > 1:
            worst = sel[int(np.argmin(self.global_rel[sel]))]
            m[worst] = False
        if not m.any():
            m[int(rng.integers(0, m.size))] = True
        return m

    def _plain_swap(self, a, rng):
        """Sparsity-preserving neighborhood move: turn one active feature off and one inactive on,
        without relevance guidance---explores the neighborhood of a good mask (e.g. a filter seed)
        when individual-feature relevance signals are noisy."""
        m = a.copy(); on = np.flatnonzero(m); off = np.flatnonzero(~m)
        if on.size and off.size:
            m[int(rng.choice(on))] = False
            m[int(rng.choice(off))] = True
        if not m.any():
            m[int(rng.integers(0, m.size))] = True
        return m

    def mutate(self, a, rng):
        a = np.asarray(a, bool)
        if not self.fa.disagreement_mutation:
            return bitstring_mutate(a, self.bcfg, rng)
        # operator portfolio: plain swap (neighborhood), guided swap (relevance), or plain bit-flip
        u = rng.random()
        if u < float(self.fa.swap_prob):
            return self._plain_swap(a, rng)
        if u < float(self.fa.swap_prob) + float(self.fa.disagreement_prob):
            return self._guided_swap(a, rng)
        return bitstring_mutate(a, self.bcfg, rng)
