"""Server-side federated evaluator: a drop-in replacement for ml_eval.Evaluator.

It exposes the same surface used by the experiment runners --
``evaluate_mask(mask) -> (objectives, MLResult)``, ``batch_evaluate_masks`` and a
``_cache`` keyed by ``mask.tobytes()`` -- but computes the objectives via
*federated evaluation*: the mask is broadcast to virtual clients, each returns
label-wise TP/FP/FN on its local validation set, and the server aggregates them
into GLOBAL micro/macro F1 (never an average of local F1 scores).

Evaluations are cached (elitism-safe, persistent, keyed by the mask) and may be promoted by the
caller into the Pareto archive.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse

from ..metrics import MLResult
from ..ml_eval import EvalConfig
from .client import ClientEvalConfig, FederatedClient
from .communication import CommunicationEstimator
from .metrics import FederatedMetricsAggregator, AggregatedStats, compute_micro_f1, compute_macro_f1
from .partition import FederatedPartitioner, PartitionConfig, partition_summary


@dataclass
class FederatedConfig:
    enabled: bool = True
    n_clients: int = 10
    partition: str = "iid"
    dirichlet_alpha: float = 0.3
    min_samples_per_client: int = 16
    size_distribution: str = "lognormal"
    client_fraction_full: float = 1.0
    final_eval_all_clients: bool = True
    return_client_metrics: bool = True
    estimate_communication: bool = True
    n_jobs: int = 1  # client-level parallelism (threads); 1 = sequential
    seed: int = 0
    # natural-silo group vectors (per-sample), optional
    groups_train: Any = None
    groups_val: Any = None
    # objective aggregation: "global" reconstructs exact global F1 from summed counts (default);
    # "local_avg" optimizes the (biased) mean of per-client F1 -- the Local-avg wrapper baseline.
    objective_aggregation: str = "global"


_CANON_ALIASES = {
    "hamming_loss": "hamming",
    "ranking_loss": "ranking",
    "micro_f1_loss": "one_minus_micro_f1",
    "macro_f1_loss": "one_minus_macro_f1",
    "avg_precision_loss": "one_minus_avg_precision",
    "gmean_f1_loss": "one_minus_gmean_f1",
    "harmonic_f1_loss": "one_minus_harmonic_f1",
}


def _canon(name: str) -> str:
    """Canonicalise an objective name (portable across implementations)."""
    n = str(name).strip().lower()
    return _CANON_ALIASES.get(n, n)


# Objectives that are decomposable from label-wise TP/FP/FN sufficient statistics.
_FEDERATED_OBJECTIVES = {
    "one_minus_macro_f1",
    "one_minus_micro_f1",
    "feature_ratio",
    "hamming",
    "one_minus_gmean_f1",
    "one_minus_harmonic_f1",
}


class FederatedEvaluator:
    def __init__(
        self,
        x_train: sparse.csr_matrix,
        y_train: sparse.csr_matrix,
        x_val: sparse.csr_matrix,
        y_val: sparse.csr_matrix,
        eval_cfg: EvalConfig,
        fed_cfg: FederatedConfig,
    ) -> None:
        self.config = eval_cfg
        self.fed = fed_cfg
        self.n_features = int(x_train.shape[1])
        self.n_labels = int(y_train.shape[1])

        self._validate_objectives()

        # Partition data into virtual clients (the server holds it only for
        # simulation; conceptually each client owns its own shard).
        pcfg = PartitionConfig(
            n_clients=int(fed_cfg.n_clients),
            kind=str(fed_cfg.partition),
            seed=int(fed_cfg.seed),
            dirichlet_alpha=float(fed_cfg.dirichlet_alpha),
            min_samples_per_client=int(fed_cfg.min_samples_per_client),
            size_distribution=str(fed_cfg.size_distribution),
        )
        partitioner = FederatedPartitioner(pcfg, n_labels=self.n_labels)
        train_idx, val_idx = partitioner.partition_paired(
            y_train, y_val,
            groups_train=fed_cfg.groups_train,
            groups_val=fed_cfg.groups_val,
        )
        self.n_clients = len(train_idx)

        ccfg = ClientEvalConfig(
            kind=str(getattr(eval_cfg, "kind", "mlknn")),
            k=int(eval_cfg.k),
            s=float(eval_cfg.s),
            mlknn_backend=str(eval_cfg.mlknn_backend),
            mlknn_device=str(eval_cfg.mlknn_device),
            cv_folds=int(getattr(eval_cfg, "cv_folds", 1)),
        )
        x_train = x_train.tocsr(); y_train = y_train.tocsr()
        x_val = x_val.tocsr(); y_val = y_val.tocsr()
        self.clients: list[FederatedClient] = []
        for cid in range(self.n_clients):
            tr, va = train_idx[cid], val_idx[cid]
            # A client with no validation rows cannot contribute statistics; a
            # client with no training rows cannot fit a model. Skip degenerate ones.
            if len(va) == 0 or len(tr) == 0:
                continue
            self.clients.append(
                FederatedClient(
                    client_id=cid,
                    x_train=x_train[tr], y_train=y_train[tr],
                    x_val=x_val[va], y_val=y_val[va],
                    n_labels=self.n_labels, cfg=ccfg,
                )
            )
        if not self.clients:
            raise ValueError("Federated partition produced no usable clients")

        self.partition_info = {
            "train": partition_summary(train_idx, y_train),
            "val": partition_summary(val_idx, y_val),
            "n_usable_clients": len(self.clients),
        }

        self.aggregator = FederatedMetricsAggregator(self.n_labels)
        self.comm = CommunicationEstimator(self.n_features, self.n_labels)
        self._rng = np.random.default_rng(int(fed_cfg.seed))

        # Full-eval cache (elitism-safe, persistent). Keyed by mask.tobytes().
        self._cache: dict[bytes, tuple[np.ndarray, MLResult]] = {}
        # Per-mask aggregated client stats from the last full evaluation.
        self._stats_cache: dict[bytes, AggregatedStats] = {}

        self.counters = {"full_evals": 0, "full_cache_hits": 0}

        # Persistent thread pool for client-level parallelism. Created once and
        # reused across the (thousands of) evaluate_mask calls to avoid per-call
        # pool spin-up overhead. Threads are effective because the per-client
        # ML-kNN work (sklearn kNN, NumPy) releases the GIL.
        self._n_jobs = int(getattr(fed_cfg, "n_jobs", 1))
        self._pool = None
        if self._n_jobs and self._n_jobs != 1:
            from concurrent.futures import ThreadPoolExecutor
            workers = self._n_jobs if self._n_jobs > 0 else (len(self.clients) or 1)
            self._pool = ThreadPoolExecutor(max_workers=min(workers, max(1, len(self.clients))))
        # Only parallelise when enough clients to amortise dispatch overhead.
        self._parallel_min_clients = 6

    # ------------------------------------------------------------------
    def _validate_objectives(self) -> None:
        names = self.config.objective_names
        if not names:
            return
        for n in names:
            canon = _canon(n)
            if canon not in _FEDERATED_OBJECTIVES:
                raise ValueError(
                    f"Objective {n!r} (canonical {canon!r}) is not decomposable from "
                    f"label-wise TP/FP/FN and cannot be used in federated mode. "
                    f"Supported: {sorted(_FEDERATED_OBJECTIVES)}"
                )

    def _sample_clients(self, client_fraction: float | None, mode: str) -> list[FederatedClient]:
        if client_fraction is None:
            client_fraction = self.fed.client_fraction_full
        cf = float(client_fraction)
        if cf >= 1.0 or mode == "full" and self.fed.final_eval_all_clients and cf >= 1.0:
            return self.clients
        k = max(1, int(round(cf * len(self.clients))))
        idx = self._rng.choice(len(self.clients), size=k, replace=False)
        return [self.clients[i] for i in idx]

    # ------------------------------------------------------------------
    def evaluate_mask(
        self,
        feature_mask: np.ndarray,
        mode: str = "full",
        client_fraction: float | None = None,
    ) -> tuple[np.ndarray, MLResult]:
        mask = np.asarray(feature_mask, dtype=bool)
        key = mask.tobytes()

        if key in self._cache:
            self.counters["full_cache_hits"] += 1
            return self._cache[key]

        clients = self._sample_clients(client_fraction, mode)
        if self._pool is not None and len(clients) >= self._parallel_min_clients:
            # Reuse the persistent thread pool; clients are independent and their
            # ML-kNN releases the GIL.
            local_results = list(self._pool.map(
                lambda c: c.evaluate_mask(mask, mode="full"),
                clients,
            ))
        else:
            local_results = [
                c.evaluate_mask(mask, mode="full")
                for c in clients
            ]
        stats = self.aggregator.aggregate(local_results)

        feature_ratio = float(mask.sum() / self.n_features)
        objectives = self._build_objectives(stats, feature_ratio)
        ml = self._build_mlresult(stats)

        if self.fed.estimate_communication:
            self.comm.record_round(int(mask.sum()), len(clients))

        self.counters["full_evals"] += 1
        self._cache[key] = (objectives, ml)
        self._stats_cache[key] = stats
        return objectives, ml

    # ------------------------------------------------------------------
    def _build_objectives(self, stats: AggregatedStats, feature_ratio: float) -> np.ndarray:
        names = self.config.objective_names
        if str(getattr(self.fed, "objective_aggregation", "global")) == "local_avg":
            # Local-avg wrapper baseline: optimize the mean of per-client F1 (biased under skew).
            micro = float(np.mean(stats.client_micro_f1)) if stats.client_micro_f1 else 0.0
            macro = float(np.mean(stats.client_macro_f1)) if stats.client_macro_f1 else 0.0
        else:
            micro, macro = stats.micro_f1, stats.macro_f1
        if not names:
            # Legacy 2-objective: (1 - macro_f1, feature_ratio).
            return np.array([1.0 - macro, feature_ratio], dtype=float)
        obj: list[float] = []
        for n in names:
            canon = _canon(n)
            if canon == "one_minus_macro_f1":
                obj.append(1.0 - macro)
            elif canon == "one_minus_micro_f1":
                obj.append(1.0 - micro)
            elif canon == "feature_ratio":
                obj.append(feature_ratio)
            elif canon == "hamming":
                denom = max(1, stats.n_val * self.n_labels)
                obj.append(float(np.sum(stats.fp + stats.fn)) / denom)
            elif canon == "one_minus_gmean_f1":
                obj.append(1.0 - float(np.sqrt(max(micro, 0.0) * max(macro, 0.0))))
            elif canon == "one_minus_harmonic_f1":
                h = 2 * micro * macro / (micro + macro) if (micro + macro) > 0 else 0.0
                obj.append(1.0 - h)
            else:
                raise ValueError(f"Unsupported federated objective: {n}")
        return np.array(obj, dtype=float)

    def _build_mlresult(self, stats: AggregatedStats) -> MLResult:
        denom = max(1, stats.n_val * self.n_labels)
        hamming = float(np.sum(stats.fp + stats.fn)) / denom
        # ranking/avg_precision/one_error are not decomposable from TP/FP/FN and
        # are not transmitted by clients; left at neutral values.
        return MLResult(
            hamming=hamming,
            ranking=0.0,
            avg_precision=0.0,
            f1_micro=float(stats.micro_f1),
            f1_macro=float(stats.macro_f1),
            one_error=0.0,
            zero_one_loss=0.0,
        )

    # ------------------------------------------------------------------
    def batch_evaluate_masks(self, masks: list[np.ndarray]) -> list[tuple[np.ndarray, MLResult]]:
        return [self.evaluate_mask(m) for m in masks]

    def client_metrics_for(self, mask: np.ndarray) -> AggregatedStats | None:
        return self._stats_cache.get(np.asarray(mask, dtype=bool).tobytes())

    def summary(self) -> dict[str, Any]:
        out = {
            "n_clients": self.n_clients,
            "n_usable_clients": len(self.clients),
            "partition": self.fed.partition,
            "counters": dict(self.counters),
        }
        if self.fed.estimate_communication:
            out["communication"] = self.comm.summary()
        return out
