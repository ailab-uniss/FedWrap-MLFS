"""Revision experiment (Reviewer 2, comment 4a): secure aggregation applied to every server-visible quantity.

In the submitted manuscript secure aggregation was verified on the per-evaluation TP/FP/FN counts; the
one-time relevance sketch was summed in plaintext. All quantities that the server needs are additive, so
the same pairwise masking over Z_2^64 (fedwrap.federated.secure_agg) applies to:

  S0  relevance sketch: Y^T X, sum X, sum X^2 (fixed point, F_BITS fractional bits), label counts, n;
  S1  filter seeds: size-weighted local relevance n_i * F_i (fixed point) and the local top-k indicator
      vectors at each seeding ratio (their sum gives top-frequency votes; votes > 0 gives the union);
  S2  every mask evaluation: TP/FP/FN and n_val (integers, exact) plus sum_i F1_i and sum_i F1_i^2
      (fixed point, 32 fractional bits) for the client-stability criterion.

The server-side code below only ever touches the masked sums. A complete FedAware search is run in this
mode and compared with the plaintext run of the same split and seed.

Usage: python scripts/revision/run_secure_e2e.py <dataset> <seed> <reference_run_dir>
"""
from __future__ import annotations

import copy
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
import fedwrap.experiment as E
import fedwrap.federated.baselines as B
from fedwrap.fedaware import federated_relevance as plain_relevance
from fedwrap.federated.metrics import (AggregatedStats, FederatedMetricsAggregator, compute_macro_f1,
                                       compute_micro_f1)
from fedwrap.federated.secure_agg import secure_sum

DS, SEED, REF = sys.argv[1], int(sys.argv[2]), sys.argv[3]
OUT = f"runs/revision/secure/{DS}_faport_s{SEED}"
F_BITS = 20                  # fractional bits for real-valued sketch statistics
F1_BITS = 32                 # fractional bits for per-client F1 sums
REPORT: dict = {"dataset": DS, "seed": SEED, "f_bits": F_BITS, "secure_sum_calls": 0, "secure_sum_s": 0.0}
_seed = [0]


def ssum(vecs):
    """Server-visible operation: add the masked client vectors (masks cancel in the sum)."""
    _seed[0] += 1
    t0 = time.perf_counter()
    out = secure_sum(vecs, base_seed=_seed[0])
    REPORT["secure_sum_s"] += time.perf_counter() - t0
    REPORT["secure_sum_calls"] += 1
    return out


def enc(x, bits):
    v = np.asarray(x, dtype=np.float64) * float(1 << bits)
    assert np.all(np.abs(v) < 2.0 ** 58), "fixed-point overflow: reduce the number of fractional bits"
    return np.rint(v).astype(np.int64)


# ── S0: relevance sketch from masked sums ─────────────────────────────────────────────────────────
def secure_relevance(clients, n_features, n_labels, hard_client_weight=None):
    D, L = n_features, n_labels
    vecs = []
    for c in clients:                                   # client side
        X = c.x_train.tocsr(); Y = c.y_train.tocsr()
        real = np.concatenate([np.asarray((Y.T @ X).todense(), dtype=np.float64).ravel(),
                               np.asarray(X.sum(axis=0), dtype=np.float64).ravel(),
                               np.asarray(X.multiply(X).sum(axis=0), dtype=np.float64).ravel()])
        ints = np.concatenate([np.asarray(Y.sum(axis=0)).ravel(), [X.shape[0]]]).astype(np.int64)
        vecs.append(np.concatenate([enc(real, F_BITS), ints]))
    tot = ssum(vecs)                                    # server side: masked sum only
    scale = float(1 << F_BITS)
    YtX = tot[: L * D].reshape(L, D) / scale
    sumX = tot[L * D: L * D + D] / scale
    sumX2 = tot[L * D + D: L * D + 2 * D] / scale
    pos = tot[L * D + 2 * D: L * D + 2 * D + L].astype(np.float64)
    n_tot = max(float(tot[-1]), 1.0)
    neg = np.maximum(n_tot - pos, 1.0); pos_safe = np.maximum(pos, 1.0)
    mean = sumX / n_tot; sd = np.sqrt(np.maximum(sumX2 / n_tot - mean ** 2, 1e-9))
    R = np.abs(YtX / pos_safe[:, None] - (sumX[None, :] - YtX) / neg[:, None]) / sd[None, :]
    R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    # diagnostics against the plaintext sketch (not available to a real server)
    Rp, _ = plain_relevance(clients, D, L)
    REPORT["sketch_words_per_client"] = int(vecs[0].size)
    REPORT["sketch_max_abs_dR"] = float(np.max(np.abs(R - Rp)))
    REPORT["sketch_top20_pools_identical"] = bool(all(
        np.array_equal(np.argsort(-R[l])[:20], np.argsort(-Rp[l])[:20]) for l in range(L)))
    REPORT["sketch_full_rank_identical"] = bool(all(
        np.array_equal(np.argsort(-R[l]), np.argsort(-Rp[l])) for l in range(L)))
    return R, R.mean(axis=0).astype(np.float32)


# ── S1: filter seeds from masked sums ─────────────────────────────────────────────────────────────
_plain_local_relevance = B._local_relevance
_plain_fed_rank, _plain_union, _plain_votes = B.fed_rank_relevance, B.local_topk_union, B.topk_frequency_scores
_lrel_cache: dict = {}


def _lrel(c):
    if id(c) not in _lrel_cache:
        _lrel_cache[id(c)] = np.asarray(_plain_local_relevance(c.x_train, c.y_train), dtype=np.float64)
    return _lrel_cache[id(c)]


def secure_fed_rank(clients):
    vecs = [np.concatenate([enc(c.n_train * _lrel(c), 8), [c.n_train]]).astype(np.int64) for c in clients]
    tot = ssum(vecs)
    out = (tot[:-1] / float(1 << 8)) / max(float(tot[-1]), 1.0)
    REPORT["fed_rank_order_identical"] = bool(np.array_equal(np.argsort(-out), np.argsort(-_plain_fed_rank(clients))))
    return out


def secure_votes(clients, ratio):
    D = clients[0].x_train.shape[1]; k = max(1, int(round(float(ratio) * D)))
    vecs = []
    for c in clients:
        ind = np.zeros(D, dtype=np.int64); ind[np.argsort(-_lrel(c))[:k]] = 1
        vecs.append(ind)
    return ssum(vecs).astype(float)


def secure_union(clients, ratio):
    return secure_votes(clients, ratio) > 0


# ── S2: evaluation aggregates from masked sums ────────────────────────────────────────────────────
@dataclass
class SecureStats(AggregatedStats):
    secure_std: float = 0.0

    @property
    def std_client_macro_f1(self) -> float:          # from sum F1_i and sum F1_i^2 only
        return float(self.secure_std)

    @property
    def worst_client_macro_f1(self) -> float:        # not derivable from sums
        return float("nan")


def secure_aggregate(self, local_results):
    L = self.n_labels; vecs = []
    for r in local_results:                              # client side
        f1 = compute_macro_f1(r["tp"], r["fp"], r["fn"])
        vecs.append(np.concatenate([np.asarray(r["tp"], np.int64), np.asarray(r["fp"], np.int64),
                                    np.asarray(r["fn"], np.int64), [int(r["n_val"])],
                                    enc([f1, f1 * f1], F1_BITS)]).astype(np.int64))
    tot = ssum(vecs)                                     # server side
    tp, fp, fn = tot[:L], tot[L:2 * L], tot[2 * L:3 * L]
    K = len(local_results); s1 = tot[-2] / float(1 << F1_BITS); s2 = tot[-1] / float(1 << F1_BITS)
    std = float(np.sqrt(max(s2 / K - (s1 / K) ** 2, 0.0)))
    return SecureStats(tp=tp, fp=fp, fn=fn, n_val=int(tot[3 * L]), clients_used=K,
                       micro_f1=compute_micro_f1(tp, fp, fn), macro_f1=compute_macro_f1(tp, fp, fn),
                       client_micro_f1=[], client_macro_f1=[], client_sizes=[], secure_std=std)


def build_cfg():
    import yaml
    cfg = copy.deepcopy(yaml.safe_load(open("configs/main_bench.yaml")))

    def setk(d, dot, v):
        p = dot.split("."); c = d
        for k in p[:-1]:
            c = c.setdefault(k, {})
        c[p[-1]] = v
    nclients = len(set(np.load(f"data/fed_real/{DS}/fold0/trainval_groups.npy", allow_pickle=True).tolist()))
    setk(cfg, "seed", SEED)
    setk(cfg, "dataset.kind", "prefold"); setk(cfg, "dataset.root", "data/fed_real"); setk(cfg, "dataset.name", DS)
    setk(cfg, "cross_validation.enabled", True); setk(cfg, "cross_validation.n_folds", 1)
    setk(cfg, "model.kind", "mlknn"); setk(cfg, "model.mlknn_backend", "sklearn"); setk(cfg, "model.mlknn_device", "cpu")
    setk(cfg, "model.k", 5); setk(cfg, "model.cv_folds", 1)
    setk(cfg, "evolution.genotype", "bitstring")
    cfg.get("evolution", {}).pop("max_evals_per_feature", None)
    setk(cfg, "evolution.max_evals", 300000)
    setk(cfg, "evolution.early_stopping", {"enabled": True, "mode": "window", "window": 10, "rel_tol": 0.002, "patience": 3})
    setk(cfg, "fedaware.enabled", True)
    setk(cfg, "fedaware.stability_tiebreak", True); setk(cfg, "fedaware.disagreement_mutation", True)
    setk(cfg, "fedaware.relevance_warmstart", True); setk(cfg, "fedaware.warmstart_frac", 0.3)
    setk(cfg, "fedaware.filter_seed", True); setk(cfg, "fedaware.swap_prob", 0.4)
    setk(cfg, "federated", {"enabled": True, "n_clients": nclients, "partition": "natural_silo",
                            "min_samples_per_client": 32, "client_fraction_full": 1.0, "final_eval_all_clients": True,
                            "return_client_metrics": True, "estimate_communication": True, "n_jobs": 8})
    setk(cfg, "reporting.max_feature_ratio", 0.25)
    setk(cfg, "logging.out_dir", OUT)
    return cfg


def main():
    E.federated_relevance = secure_relevance
    B.fed_rank_relevance = secure_fed_rank
    B.topk_frequency_scores = secure_votes
    B.local_topk_union = secure_union
    FederatedMetricsAggregator.aggregate = secure_aggregate
    t0 = time.perf_counter()
    E.run_experiment_from_config(build_cfg(), fold_idx=0)
    REPORT["wall_s"] = time.perf_counter() - t0
    REPORT["secure_overhead_pct"] = 100.0 * REPORT["secure_sum_s"] / REPORT["wall_s"]
    out = Path(OUT + "_fold0")
    a = np.load(out / "population_masks.npz"); b = np.load(Path(REF) / "population_masks.npz")
    sa = json.loads((out / "summary.json").read_text()); sb = json.loads((Path(REF) / "summary.json").read_text())
    REPORT["reference"] = REF
    REPORT["masks_identical"] = bool(a["masks_packed"].shape == b["masks_packed"].shape
                                     and np.array_equal(a["masks_packed"], b["masks_packed"]))
    same = a["val_objs"].shape == b["val_objs"].shape
    REPORT["val_objs_max_abs_diff"] = float(np.max(np.abs(a["val_objs"] - b["val_objs"]))) if same else None
    REPORT["test_objs_max_abs_diff"] = float(np.max(np.abs(a["test_objs"] - b["test_objs"]))) if same else None
    for k in ("gen", "total_evals", "hv_val", "hv_test"):
        REPORT[k] = sa["final"][k]; REPORT[k + "_ref"] = sb["final"][k]
    Path("reports/revision").mkdir(parents=True, exist_ok=True)
    (Path("reports/revision") / f"secure_e2e_{DS}_s{SEED}.json").write_text(json.dumps(REPORT, indent=2))
    print(json.dumps(REPORT, indent=2), flush=True)


if __name__ == "__main__":
    main()
