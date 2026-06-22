"""Federated simulation layer for Fed-CC-FedWrap-MLFS.

The server evolves feature masks; virtual clients keep their data local and
return only label-wise sufficient statistics. See ``federated_cc_research_plan.md``.
"""
from __future__ import annotations

from typing import Any

from scipy import sparse

from ..ml_eval import EvalConfig, Evaluator
from .client import ClientEvalConfig, FederatedClient
from .communication import CommunicationEstimator
from .evaluator import FederatedConfig, FederatedEvaluator
from .metrics import (
    AggregatedStats,
    FederatedMetricsAggregator,
    compute_macro_f1,
    compute_micro_f1,
    per_label_f1,
)
from .partition import FederatedPartitioner, PartitionConfig, partition_summary

__all__ = [
    "ClientEvalConfig",
    "FederatedClient",
    "FederatedEvaluator",
    "FederatedConfig",
    "FederatedMetricsAggregator",
    "AggregatedStats",
    "FederatedPartitioner",
    "PartitionConfig",
    "CommunicationEstimator",
    "compute_macro_f1",
    "compute_micro_f1",
    "per_label_f1",
    "partition_summary",
    "federated_config_from_dict",
    "is_federated_enabled",
    "make_evaluator",
]


def _get(d: dict, dotted: str, default=None):
    cur: Any = d
    for p in dotted.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def is_federated_enabled(config: dict[str, Any]) -> bool:
    return bool(_get(config, "federated.enabled", False))


def federated_config_from_dict(config: dict[str, Any], seed: int) -> FederatedConfig:
    f = config.get("federated", {}) if isinstance(config, dict) else {}
    return FederatedConfig(
        enabled=bool(f.get("enabled", False)),
        n_clients=int(f.get("n_clients", 10)),
        partition=str(f.get("partition", "iid")),
        dirichlet_alpha=float(f.get("dirichlet_alpha", 0.3)),
        min_samples_per_client=int(f.get("min_samples_per_client", 16)),
        size_distribution=str(f.get("size_distribution", "lognormal")),
        client_fraction_bite=float(f.get("client_fraction_bite", 0.30)),
        client_fraction_full=float(f.get("client_fraction_full", 1.0)),
        final_eval_all_clients=bool(f.get("final_eval_all_clients", True)),
        return_client_metrics=bool(f.get("return_client_metrics", True)),
        estimate_communication=bool(f.get("estimate_communication", True)),
        n_jobs=int(f.get("n_jobs", 1)),
        seed=int(f.get("seed", seed)),
        objective_aggregation=str(f.get("objective_aggregation", "global")),
    )


def load_fed_natural_split(root: str, name: str, seed: int, val_size: float = 0.25):
    """Load a prepared real federated dataset and split trainval→train/val while
    tracking the per-row natural-silo group vectors (for natural_silo CC runs).

    Returns a dict with x/y train/val/test and aligned groups_{train,val,test} plus
    groups_trainval (for the final train+val→test evaluation).
    """
    from pathlib import Path
    import numpy as np
    from .. import datasets as _ds  # host package datasets (has _load_npz_any/_as_csr)

    fold = Path(root) / name / "fold0"
    x_tv, y_tv = _ds._load_npz_any(fold / "trainval.npz")
    x_te, y_te = _ds._load_npz_any(fold / "test.npz")
    x_tv, y_tv, x_te, y_te = x_tv.tocsr(), y_tv.tocsr(), x_te.tocsr(), y_te.tocsr()
    g_tv = np.load(fold / "trainval_groups.npy", allow_pickle=True)
    g_te = np.load(fold / "test_groups.npy", allow_pickle=True)

    rng = np.random.default_rng(int(seed))
    n = x_tv.shape[0]
    perm = rng.permutation(n)
    n_val = int(float(val_size) * n)
    val_i = np.sort(perm[:n_val]); tr_i = np.sort(perm[n_val:])
    return {
        "x_train": x_tv[tr_i], "y_train": y_tv[tr_i],
        "x_val": x_tv[val_i], "y_val": y_tv[val_i],
        "x_test": x_te, "y_test": y_te,
        "groups_train": g_tv[tr_i], "groups_val": g_tv[val_i],
        "groups_trainval": g_tv, "groups_test": g_te,
    }


def make_evaluator(
    x_train: sparse.csr_matrix,
    y_train: sparse.csr_matrix,
    x_val: sparse.csr_matrix,
    y_val: sparse.csr_matrix,
    eval_cfg: EvalConfig,
    config: dict[str, Any],
    seed: int,
    groups: tuple | None = None,
):
    """Return a FederatedEvaluator when ``federated.enabled`` is set, else Evaluator.

    ``groups=(groups_train, groups_val)`` supplies the per-row natural-silo group
    vectors for ``partition: natural_silo`` (otherwise they come from the config).
    Both evaluators share the ``evaluate_mask`` / ``batch_evaluate_masks`` / ``_cache``
    surface used by the experiment runners, so they are interchangeable.
    """
    if is_federated_enabled(config):
        fed_cfg = federated_config_from_dict(config, seed=seed)
        if groups is not None:
            fed_cfg.groups_train, fed_cfg.groups_val = groups
        return FederatedEvaluator(x_train, y_train, x_val, y_val, eval_cfg, fed_cfg)
    return Evaluator(x_train, y_train, x_val, y_val, eval_cfg)
