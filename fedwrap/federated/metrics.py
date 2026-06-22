"""Federated metric aggregation from label-wise sufficient statistics.

The server never sees raw data: clients return per-label ``TP``, ``FP``, ``FN``
counts on their local validation set. The server sums them and computes *global*
micro-F1 and macro-F1. Per the research plan (section 5.2) we must NOT average
local F1 scores, because that biases the estimate under quantity- and label-skew.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def compute_micro_f1(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> float:
    tp_s = float(np.sum(tp))
    fp_s = float(np.sum(fp))
    fn_s = float(np.sum(fn))
    denom = 2.0 * tp_s + fp_s + fn_s
    if denom <= 0.0:
        return 0.0
    return (2.0 * tp_s) / denom


def compute_macro_f1(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> float:
    tp = np.asarray(tp, dtype=float)
    fp = np.asarray(fp, dtype=float)
    fn = np.asarray(fn, dtype=float)
    denom = 2.0 * tp + fp + fn
    f1 = np.zeros_like(denom)
    valid = denom > 0.0
    f1[valid] = (2.0 * tp[valid]) / denom[valid]
    if f1.size == 0:
        return 0.0
    # Labels with denom==0 (never predicted and never present) contribute F1=0,
    # matching sklearn macro-F1 with zero_division=0.
    return float(np.mean(f1))


def per_label_f1(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> np.ndarray:
    tp = np.asarray(tp, dtype=float)
    fp = np.asarray(fp, dtype=float)
    fn = np.asarray(fn, dtype=float)
    denom = 2.0 * tp + fp + fn
    f1 = np.zeros_like(denom)
    valid = denom > 0.0
    f1[valid] = (2.0 * tp[valid]) / denom[valid]
    return f1


@dataclass
class AggregatedStats:
    tp: np.ndarray
    fp: np.ndarray
    fn: np.ndarray
    n_val: int
    clients_used: int
    micro_f1: float
    macro_f1: float
    # Per-client F1 (computed locally, used only as external robustness metrics)
    client_micro_f1: list[float]
    client_macro_f1: list[float]
    client_sizes: list[int]

    @property
    def worst_client_macro_f1(self) -> float:
        return float(min(self.client_macro_f1)) if self.client_macro_f1 else 0.0

    @property
    def std_client_macro_f1(self) -> float:
        return float(np.std(self.client_macro_f1)) if self.client_macro_f1 else 0.0

    @property
    def std_client_micro_f1(self) -> float:
        return float(np.std(self.client_micro_f1)) if self.client_micro_f1 else 0.0


class FederatedMetricsAggregator:
    """Aggregates per-client label-wise statistics into global metrics."""

    def __init__(self, n_labels: int) -> None:
        self.n_labels = int(n_labels)

    def aggregate(self, local_results: list[dict]) -> AggregatedStats:
        tp = np.zeros(self.n_labels, dtype=np.int64)
        fp = np.zeros(self.n_labels, dtype=np.int64)
        fn = np.zeros(self.n_labels, dtype=np.int64)
        n_val = 0
        client_micro: list[float] = []
        client_macro: list[float] = []
        client_sizes: list[int] = []

        for r in local_results:
            tp += np.asarray(r["tp"], dtype=np.int64)
            fp += np.asarray(r["fp"], dtype=np.int64)
            fn += np.asarray(r["fn"], dtype=np.int64)
            n_val += int(r["n_val"])
            client_micro.append(compute_micro_f1(r["tp"], r["fp"], r["fn"]))
            client_macro.append(compute_macro_f1(r["tp"], r["fp"], r["fn"]))
            client_sizes.append(int(r["n_val"]))

        return AggregatedStats(
            tp=tp,
            fp=fp,
            fn=fn,
            n_val=n_val,
            clients_used=len(local_results),
            micro_f1=compute_micro_f1(tp, fp, fn),
            macro_f1=compute_macro_f1(tp, fp, fn),
            client_micro_f1=client_micro,
            client_macro_f1=client_macro,
            client_sizes=client_sizes,
        )
