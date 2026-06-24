"""Federated feature-selection baselines.

These produce a *feature ranking* (or directly a mask) without centralising raw
data: each client computes a local relevance score; the server aggregates. Masks
are then scored by the same :class:`FederatedEvaluator` protocol used by
Fed-CC-FedWrap-MLFS, so the comparison is apples-to-apples.

Implemented:
- ``all_features``      : sanity upper bound on feature count.
- ``random_subset``     : random mask at a fixed feature ratio.
- ``fed_rank_relevance``: FedAvg-Rank. Each client computes per-feature relevance
                          (summed ANOVA F-score over labels); the server sums them
                          weighted by client size -> global ranking.
- ``local_topk_union``  : each client selects its local top-k features; the union
                          is taken (tends to select many features).
- ``topk_frequency``    : features ranked by how many clients put them in their
                          local top-k (a simple but strong baseline).
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse
from sklearn.feature_selection import f_classif

from .client import FederatedClient


def all_features_mask(n_features: int) -> np.ndarray:
    return np.ones(int(n_features), dtype=bool)


def random_subset_mask(n_features: int, ratio: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    k = max(1, int(round(float(ratio) * n_features)))
    idx = rng.choice(n_features, size=min(k, n_features), replace=False)
    mask = np.zeros(int(n_features), dtype=bool)
    mask[idx] = True
    return mask


def _local_relevance(x: sparse.csr_matrix, y: sparse.csr_matrix) -> np.ndarray:
    """Per-feature relevance = sum over labels of the ANOVA F-score.

    Robust to negative features (unlike chi2) and cheap on sparse data.
    """
    xd = x.toarray() if sparse.issparse(x) else np.asarray(x)
    yd = (y.toarray() > 0).astype(int) if sparse.issparse(y) else (np.asarray(y) > 0).astype(int)
    n_features = xd.shape[1]
    scores = np.zeros(n_features, dtype=float)
    for l in range(yd.shape[1]):
        col = yd[:, l]
        if col.sum() == 0 or col.sum() == col.shape[0]:
            continue  # label has a single class on this client -> uninformative
        with np.errstate(all="ignore"):
            f, _ = f_classif(xd, col)
        f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
        scores += f
    return scores


def fed_rank_relevance(clients: list[FederatedClient]) -> np.ndarray:
    """Global feature ranking via size-weighted aggregation of local relevance."""
    n_features = clients[0].x_train.shape[1]
    agg = np.zeros(n_features, dtype=float)
    total = 0
    for c in clients:
        w = c.n_train
        agg += w * _local_relevance(c.x_train, c.y_train)
        total += w
    if total > 0:
        agg /= total
    return agg  # higher = more relevant


def _local_mi_relevance(x: sparse.csr_matrix, y: sparse.csr_matrix) -> np.ndarray:
    """Per-feature relevance = sum over labels of MI(feature, label) (FMLFS relevance)."""
    from sklearn.feature_selection import mutual_info_classif
    xd = x.toarray() if sparse.issparse(x) else np.asarray(x)
    yd = (y.toarray() > 0).astype(int) if sparse.issparse(y) else (np.asarray(y) > 0).astype(int)
    rel = np.zeros(xd.shape[1], dtype=float)
    for l in range(yd.shape[1]):
        col = yd[:, l]
        if col.sum() == 0 or col.sum() == col.shape[0]:
            continue
        with np.errstate(all="ignore"):
            mi = mutual_info_classif(xd, col, discrete_features=False, random_state=0)
        rel += np.nan_to_num(mi, nan=0.0)
    return rel


def _local_redundancy(x: sparse.csr_matrix) -> np.ndarray:
    """Feature-feature redundancy = |Pearson correlation| matrix (FMLFS uses a
    correlation-distance redundancy). This is O(D^2): the cost that prevents
    information-theoretic mRMR filters such as FMLFS from scaling to high D."""
    xd = x.toarray() if sparse.issparse(x) else np.asarray(x)
    with np.errstate(all="ignore"):
        c = np.corrcoef(xd, rowvar=False)
    c = np.nan_to_num(np.abs(c), nan=0.0)
    np.fill_diagonal(c, 0.0)
    return c


def fmlfs(clients: list[FederatedClient], ratio: float,
          max_features_for_redundancy: int = 6000) -> np.ndarray:
    """FMLFS-style federated mRMR (Anonymous 2024, the only prior federated
    multi-label FS): size-weighted aggregation of local MI relevance and
    feature-feature redundancy, then greedy mRMR selection of k = ratio*D
    features. Raises at high D where the O(D^2) redundancy matrix is intractable
    (the scalability gap this paper's wrapper closes)."""
    n_features = clients[0].x_train.shape[1]
    if n_features > int(max_features_for_redundancy):
        raise MemoryError(
            f"FMLFS redundancy is O(D^2): D={n_features} exceeds the feasible cap "
            f"{max_features_for_redundancy} (a {n_features**2*8/1e9:.1f} GB matrix).")
    rel = np.zeros(n_features, dtype=float)
    red = np.zeros((n_features, n_features), dtype=float)
    total = 0
    for c in clients:
        w = c.n_train
        rel += w * _local_mi_relevance(c.x_train, c.y_train)
        red += w * _local_redundancy(c.x_train)
        total += w
    if total > 0:
        rel /= total
        red /= total
    # Greedy mRMR: maximise relevance - mean redundancy with the selected set.
    k = max(1, int(round(float(ratio) * n_features)))
    selected: list[int] = [int(np.argmax(rel))]
    remaining = set(range(n_features)) - set(selected)
    while len(selected) < k and remaining:
        rem = np.fromiter(remaining, dtype=int)
        mrmr = rel[rem] - red[np.ix_(rem, selected)].mean(axis=1)
        best = int(rem[int(np.argmax(mrmr))])
        selected.append(best)
        remaining.discard(best)
    mask = np.zeros(n_features, dtype=bool)
    mask[selected] = True
    return mask


def ranking_to_mask(scores: np.ndarray, ratio: float) -> np.ndarray:
    n_features = scores.shape[0]
    k = max(1, int(round(float(ratio) * n_features)))
    top = np.argsort(-scores)[:k]
    mask = np.zeros(n_features, dtype=bool)
    mask[top] = True
    return mask


def local_topk_union(clients: list[FederatedClient], ratio: float) -> np.ndarray:
    n_features = clients[0].x_train.shape[1]
    k = max(1, int(round(float(ratio) * n_features)))
    mask = np.zeros(n_features, dtype=bool)
    for c in clients:
        s = _local_relevance(c.x_train, c.y_train)
        top = np.argsort(-s)[:k]
        mask[top] = True
    return mask


def topk_frequency_scores(clients: list[FederatedClient], ratio: float) -> np.ndarray:
    """Count, per feature, how many clients rank it in their local top-(ratio)."""
    n_features = clients[0].x_train.shape[1]
    k = max(1, int(round(float(ratio) * n_features)))
    votes = np.zeros(n_features, dtype=float)
    for c in clients:
        s = _local_relevance(c.x_train, c.y_train)
        top = np.argsort(-s)[:k]
        votes[top] += 1.0
    return votes


def build_baseline_masks(
    clients: list[FederatedClient],
    n_features: int,
    ratios: list[float],
    seed: int = 0,
    fmlfs_max_features: int | None = None,
) -> dict[str, dict[float, np.ndarray]]:
    """Return {method: {ratio: mask}} for the ranking-based baselines.

    ``fmlfs_max_features`` skips the O(D^2) FMLFS build above that dimensionality (it becomes
    computationally infeasible); the omission is reported by the caller as an infeasibility result.
    """
    out: dict[str, dict[float, np.ndarray]] = {}

    # Rankings computed once, then thresholded at each ratio.
    fed_scores = fed_rank_relevance(clients)
    out["fed_rank_relevance"] = {r: ranking_to_mask(fed_scores, r) for r in ratios}

    # FMLFS-style federated mRMR (the prior federated multi-label FS baseline).
    # O(D^2) redundancy -> only feasible at low/medium D; skipped/raised otherwise.
    if fmlfs_max_features is not None and n_features > fmlfs_max_features:
        print(f"[baselines] FMLFS skipped at D={n_features} (> {fmlfs_max_features}, O(D^2) infeasible)")
    else:
        try:
            out["fmlfs"] = {r: fmlfs(clients, r) for r in ratios}
        except Exception as e:
            print(f"[baselines] FMLFS infeasible at D={n_features}: {e}")

    out["topk_frequency"] = {}
    out["local_topk_union"] = {}
    out["random_subset"] = {}
    for r in ratios:
        freq = topk_frequency_scores(clients, r)
        out["topk_frequency"][r] = ranking_to_mask(freq, r)
        out["local_topk_union"][r] = local_topk_union(clients, r)
        out["random_subset"][r] = random_subset_mask(n_features, r, seed=seed)

    out["all_features"] = {1.0: all_features_mask(n_features)}
    return out
