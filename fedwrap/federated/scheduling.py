"""Resource-aware, straggler-tolerant scheduling for the federated wrapper-evaluation loop.

In a compute-continuum deployment the clients of a federation live on heterogeneous tiers---edge
devices, on-premise servers, cloud/HPC nodes---with very different compute speeds and availability.
Every wrapper evaluation is a synchronous round (broadcast a feature mask, each client returns its
label-wise TP/FP/FN, the server sums them), so the round's critical-path latency is set by the
slowest participating client. Over the thousands of rounds an evolutionary search needs, stragglers
dominate wall-clock time.

This module provides the scheduling layer that the workflow uses to stay efficient under tier
heterogeneity, and the machinery to evaluate it. Two observations make it clean:

* The global objective is a *sum* of per-client sufficient statistics. Aggregating over any subset
  of responding clients therefore yields the **exact** global micro/macro-F1 *for that subset*---
  there is no imputation and no bias in the metric itself, only the sampling variance of which
  clients participated. ``aggregate_subset`` returns exactly this.
* Because the metric is a deterministic function of summed counts, a scheduling policy can be
  studied by precomputing each client's counts for each candidate mask **once** and then replaying
  policies analytically over those counts (``replay_policy``), instead of re-running the federation
  per policy.

The latency model (``tier_latencies``) is an explicit emulation of tier heterogeneity: per-client
compute time scales with the client's data size and a per-tier speed factor, plus a fixed
communication term. It is calibrated to plausible edge/server/cloud ratios and is clearly an
emulation, not measured hardware; callers may instead pass measured per-client times.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .metrics import compute_macro_f1, compute_micro_f1


# Plausible relative compute-speed factors for a three-tier continuum (time multipliers; larger =
# slower). Edge devices are ~an order of magnitude slower than a cloud/HPC node per unit of data.
DEFAULT_TIERS = {"edge": 8.0, "server": 3.0, "cloud": 1.0}


@dataclass
class TierModel:
    """Emulated per-client latency under tier heterogeneity.

    ``compute_per_sample`` and ``comm_fixed`` are in arbitrary time units; only their ratios and the
    tier factors affect the reported speed-ups. Assignments are round-robin over the tier names
    unless ``assignment`` is given explicitly (client index -> tier name)."""
    tiers: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_TIERS))
    compute_per_sample: float = 1e-4
    comm_fixed: float = 0.05
    assignment: dict[int, str] | None = None

    def assign(self, n_clients: int) -> list[str]:
        if self.assignment is not None:
            return [self.assignment.get(i, "cloud") for i in range(n_clients)]
        names = list(self.tiers)
        return [names[i % len(names)] for i in range(n_clients)]

    def latencies(self, client_sizes: np.ndarray) -> np.ndarray:
        """Per-client round latency (seconds, emulated) for the given per-client data sizes."""
        names = self.assign(len(client_sizes))
        factor = np.array([self.tiers[n] for n in names], dtype=float)
        return factor * (self.compute_per_sample * np.asarray(client_sizes, float)) + self.comm_fixed


def aggregate_subset(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray, idx: np.ndarray) -> dict:
    """Exact global micro/macro-F1 over the responding subset ``idx`` of clients.

    ``tp/fp/fn`` are (K, L) per-client label-wise counts; summing the rows in ``idx`` and scoring is
    identical to scoring the concatenation of those clients' predictions (no imputation)."""
    s_tp = tp[idx].sum(0); s_fp = fp[idx].sum(0); s_fn = fn[idx].sum(0)
    return {"macro": float(compute_macro_f1(s_tp, s_fp, s_fn)),
            "micro": float(compute_micro_f1(s_tp, s_fp, s_fn))}


def select_clients(policy: str, sizes: np.ndarray, latencies: np.ndarray, quorum: float,
                   rng: np.random.Generator, deadline_q: float = 1.0) -> np.ndarray:
    """Choose the participating clients for one round, combining two orthogonal levers.

    A *mass quorum* (``quorum``, the target fraction of total validation mass) controls how much
    compute and communication a round spends; a *latency deadline* (``deadline_q``, a quantile of the
    client latency distribution) controls round latency by not waiting for the slow tail. Either can
    be relaxed (set to ``1.0``) to isolate the other.
      * ``full``           : all clients (the synchronous baseline).
      * ``uniform``        : random clients until the mass quorum is met.
      * ``resource_aware`` : add clients in increasing latency-per-sample order (cheap, data-dense
                             clients first) until the mass quorum is met---cheap yet representative.
    The latency deadline is then applied to whichever set the policy chose: clients slower than the
    ``deadline_q`` quantile are dropped (they would miss the round). Aggregation over the survivors is
    still exact for that subset.
    """
    K = len(sizes); total = float(sizes.sum()); target = quorum * total
    if policy == "full":
        chosen = list(range(K))
    elif policy in ("uniform", "resource_aware"):
        order = rng.permutation(K) if policy == "uniform" else np.argsort(latencies / np.maximum(sizes, 1.0))
        chosen, acc = [], 0.0
        for i in order:
            chosen.append(int(i)); acc += sizes[i]
            if acc >= target:
                break
    else:
        raise ValueError(f"unknown policy {policy!r}")
    chosen = np.array(chosen, dtype=int)
    if deadline_q < 1.0:
        deadline = float(np.quantile(latencies, deadline_q))
        kept = chosen[latencies[chosen] <= deadline]
        if kept.size:  # never drop everyone
            chosen = kept
    return chosen


def replay_policy(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray, sizes: np.ndarray,
                  latencies: np.ndarray, policy: str, quorum: float, *,
                  deadline_q: float = 1.0, seed: int = 0) -> dict:
    """Replay a scheduling policy over precomputed per-client counts for a set of masks.

    ``tp/fp/fn`` are (M, K, L): for each of M evaluated masks, each client's label-wise counts.
    Returns the mean macro/micro-F1 under the policy, the mean absolute deviation from
    full-participation macro-F1, the emulated critical-path wall-time (sum over rounds of the slowest
    surviving participant's latency), the mean fraction of clients that participated, and the mean
    communication (clients that uploaded counters).
    """
    M = tp.shape[0]; rng = np.random.default_rng(seed)
    macro, micro, dev, wall, frac = [], [], [], [], []
    full = np.arange(tp.shape[1])
    for m in range(M):
        full_macro = aggregate_subset(tp[m], fp[m], fn[m], full)["macro"]
        idx = select_clients(policy, sizes, latencies, quorum, rng, deadline_q=deadline_q)
        a = aggregate_subset(tp[m], fp[m], fn[m], idx)
        macro.append(a["macro"]); micro.append(a["micro"]); dev.append(abs(a["macro"] - full_macro))
        wall.append(float(latencies[idx].max())); frac.append(len(idx) / len(full))
    return {"policy": policy, "quorum": quorum, "deadline_q": deadline_q, "n_masks": M,
            "macro": float(np.mean(macro)), "micro": float(np.mean(micro)),
            "mae_vs_full": float(np.mean(dev)), "walltime": float(np.sum(wall)),
            "mean_participation": float(np.mean(frac))}
