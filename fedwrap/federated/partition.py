"""Federated partitioning of multi-label datasets into virtual clients.

Implements the partition scenarios required by the Fed-CC-FedWrap-MLFS research plan
(section 12):

- ``iid``                   : random split (experimental control).
- ``label_skew_dirichlet``  : each label is distributed across clients with a
                              Dirichlet(alpha) prior. The most important scenario
                              for multi-label non-IID.
- ``quantity_skew``         : clients receive very different amounts of data
                              (lognormal size distribution).
- ``label_quantity_skew``   : label-skew combined with quantity-skew.
- ``natural_silo``          : use a natural grouping field (e.g. hospital, language).

The partitioner is deliberately data-agnostic: it operates on the binary label
matrix ``y`` (and an optional group vector) and returns, for each client, the
array of row indices it owns. The same partitioner instance produces *consistent*
client structure for the train and validation matrices via :meth:`partition_paired`,
so that a client's training and validation data follow the same skew.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import sparse


def _as_dense_labels(y: Any) -> np.ndarray:
    if sparse.issparse(y):
        return (y.toarray() > 0).astype(np.int8)
    return (np.asarray(y) > 0).astype(np.int8)


@dataclass
class PartitionConfig:
    n_clients: int = 10
    kind: str = "iid"  # iid | label_skew_dirichlet | quantity_skew | label_quantity_skew | natural_silo
    seed: int = 0
    dirichlet_alpha: float = 0.3
    min_samples_per_client: int = 16
    min_positive_labels_per_client: int = 1
    size_distribution: str = "lognormal"  # for quantity skew
    size_sigma: float = 1.0  # lognormal sigma for quantity skew
    # natural silo group ids (per-sample), set externally; not serialised here
    groups: Any = field(default=None, repr=False)


class FederatedPartitioner:
    """Assigns dataset rows to virtual federated clients.

    For label-skew the per-label client proportions are drawn once at
    construction time so that train and validation splits share the same skew.
    """

    def __init__(self, config: PartitionConfig, n_labels: int) -> None:
        self.cfg = config
        self.n_labels = int(n_labels)
        self.n_clients = int(config.n_clients)
        self._rng = np.random.default_rng(int(config.seed))

        # Pre-draw structural randomness shared across train/val.
        self._label_proportions: np.ndarray | None = None
        self._client_size_weights: np.ndarray | None = None

        if self.cfg.kind in ("label_skew_dirichlet", "label_quantity_skew"):
            # proportions[l] is a distribution over clients for label l.
            alpha = float(self.cfg.dirichlet_alpha)
            self._label_proportions = self._rng.dirichlet(
                np.full(self.n_clients, alpha), size=self.n_labels
            )  # shape (n_labels, n_clients)
        if self.cfg.kind in ("quantity_skew", "label_quantity_skew"):
            if self.cfg.size_distribution == "lognormal":
                w = self._rng.lognormal(mean=0.0, sigma=float(self.cfg.size_sigma), size=self.n_clients)
            else:
                w = self._rng.uniform(0.5, 1.5, size=self.n_clients)
            self._client_size_weights = w / w.sum()

    # ------------------------------------------------------------------
    def partition_paired(
        self,
        y_train: Any,
        y_val: Any,
        groups_train: Any = None,
        groups_val: Any = None,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Partition train and validation rows consistently across clients."""
        if self.cfg.kind == "natural_silo":
            # Use a SHARED group vocabulary so client k = same group in train & val,
            # even when a group is absent from one split (then that shard is empty).
            if groups_train is None or groups_val is None:
                raise ValueError("natural_silo requires groups for train and val")
            gtr = np.asarray(groups_train)
            gva = np.asarray(groups_val)
            vocab = list(dict.fromkeys(gtr.tolist() + gva.tolist()))
            self.n_clients = len(vocab)
            train_idx = [np.flatnonzero(gtr == g) for g in vocab]
            val_idx = [np.flatnonzero(gva == g) for g in vocab]
            return train_idx, val_idx
        train_idx = self._assign(_as_dense_labels(y_train), groups_train)
        val_idx = self._assign(_as_dense_labels(y_val), groups_val)
        return train_idx, val_idx

    def partition(self, y: Any, groups: Any = None) -> list[np.ndarray]:
        return self._assign(_as_dense_labels(y), groups)

    # ------------------------------------------------------------------
    def _assign(self, y: np.ndarray, groups: Any = None) -> list[np.ndarray]:
        n = int(y.shape[0])
        kind = self.cfg.kind
        if kind == "natural_silo":
            idx_lists = self._assign_natural(n, groups)
        elif kind == "iid":
            idx_lists = self._assign_iid(n)
        elif kind == "quantity_skew":
            idx_lists = self._assign_quantity(n)
        elif kind in ("label_skew_dirichlet", "label_quantity_skew"):
            idx_lists = self._assign_label_skew(y, quantity=(kind == "label_quantity_skew"))
        else:
            raise ValueError(f"Unknown partition kind: {kind!r}")
        idx_lists = self._enforce_min_samples(idx_lists, n)
        return idx_lists

    def _assign_iid(self, n: int) -> list[np.ndarray]:
        perm = self._rng.permutation(n)
        return [np.sort(a) for a in np.array_split(perm, self.n_clients)]

    def _assign_quantity(self, n: int) -> list[np.ndarray]:
        perm = self._rng.permutation(n)
        w = self._client_size_weights
        counts = np.maximum(1, np.round(w * n).astype(int))
        # Adjust to sum to n.
        while counts.sum() > n:
            counts[np.argmax(counts)] -= 1
        while counts.sum() < n:
            counts[np.argmin(counts)] += 1
        out, start = [], 0
        for c in counts:
            out.append(np.sort(perm[start:start + c]))
            start += c
        return out

    def _assign_label_skew(self, y: np.ndarray, quantity: bool) -> list[np.ndarray]:
        """Assign each sample to a client using Dirichlet per-label proportions.

        Score(i, c) = sum_{l in labels(i)} P[l, c]; samples with no active label
        fall back to a uniform draw. Optionally modulated by quantity weights.
        """
        n = int(y.shape[0])
        P = self._label_proportions  # (n_labels, n_clients)
        assert P is not None
        scores = y.astype(float) @ P  # (n, n_clients)
        # Samples with no positive label: assign uniformly at random.
        no_label = scores.sum(axis=1) <= 0
        if no_label.any():
            scores[no_label] = self._rng.random((int(no_label.sum()), self.n_clients))
        if quantity and self._client_size_weights is not None:
            scores = scores * self._client_size_weights[None, :]
        # Probabilistic assignment (softmax-free): normalise rows and sample.
        row_sums = scores.sum(axis=1, keepdims=True)
        probs = scores / np.where(row_sums > 0, row_sums, 1.0)
        assignment = np.array(
            [self._rng.choice(self.n_clients, p=probs[i]) for i in range(n)]
        )
        return [np.flatnonzero(assignment == c) for c in range(self.n_clients)]

    def _assign_natural(self, n: int, groups: Any) -> list[np.ndarray]:
        if groups is None:
            groups = self.cfg.groups
        if groups is None:
            raise ValueError("natural_silo partition requires a per-sample group vector")
        groups = np.asarray(groups)
        if groups.shape[0] != n:
            raise ValueError(
                f"group vector length {groups.shape[0]} != n_samples {n}"
            )
        uniq = list(dict.fromkeys(groups.tolist()))  # preserve first-seen order
        self.n_clients = len(uniq)
        return [np.flatnonzero(groups == g) for g in uniq]

    def _enforce_min_samples(self, idx_lists: list[np.ndarray], n: int) -> list[np.ndarray]:
        """Move samples from large clients to under-filled ones (best effort)."""
        min_s = int(self.cfg.min_samples_per_client)
        if min_s <= 0:
            return idx_lists
        idx_lists = [np.asarray(a, dtype=int) for a in idx_lists]
        for c in range(len(idx_lists)):
            while len(idx_lists[c]) < min_s:
                donor = int(np.argmax([len(a) for a in idx_lists]))
                if donor == c or len(idx_lists[donor]) <= min_s:
                    break
                take = idx_lists[donor][-1]
                idx_lists[donor] = idx_lists[donor][:-1]
                idx_lists[c] = np.append(idx_lists[c], take)
        return [np.sort(a) for a in idx_lists]


def partition_summary(client_indices: list[np.ndarray], y: Any) -> dict[str, Any]:
    """Diagnostic summary of a partition (sizes, label coverage per client)."""
    yd = _as_dense_labels(y)
    sizes = [int(len(c)) for c in client_indices]
    label_coverage = []
    for c in client_indices:
        if len(c) == 0:
            label_coverage.append(0)
            continue
        present = (yd[c].sum(axis=0) > 0).sum()
        label_coverage.append(int(present))
    return {
        "n_clients": len(client_indices),
        "sizes": sizes,
        "min_size": int(min(sizes)) if sizes else 0,
        "max_size": int(max(sizes)) if sizes else 0,
        "label_coverage_per_client": label_coverage,
        "n_labels": int(yd.shape[1]),
    }
