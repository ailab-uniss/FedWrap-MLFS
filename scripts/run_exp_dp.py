#!/usr/bin/env python3
"""Experiment D --- differential privacy on the federated evaluation.

The only quantity that leaves a client is the label-wise count vector
(TP, FP, FN), so we can release it under (eps, delta)-DP with the Gaussian
mechanism. One record affects at most one of {TP,FP,FN} per label, so the L2
sensitivity of the 3L-vector is sqrt(L). We add calibrated Gaussian noise to the
aggregated counts, recompute global macro/micro-F1, and trace the
privacy--utility curve. Communication is unchanged (still 3L numbers/round)."""
from __future__ import annotations
import sys, csv
import numpy as np
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "."))
from fedwrap.federated import (FederatedClient, ClientEvalConfig,
                             compute_macro_f1, compute_micro_f1, load_fed_natural_split)

def build_clients(d, k=10):
    cfg = ClientEvalConfig(k=k, s=1.0, threshold=0.5,
                           mlknn_backend="sklearn", mlknn_device="cpu")
    L = d["y_train"].shape[1]
    gtr = np.asarray(d["groups_train"]); gva = np.asarray(d["groups_val"])
    clients = []
    for g in sorted(set(gtr.tolist())):
        itr = np.flatnonzero(gtr == g); iva = np.flatnonzero(gva == g)
        if itr.size < k + 1 or iva.size == 0:
            continue
        clients.append(FederatedClient(len(clients), d["x_train"][itr], d["y_train"][itr],
                                        d["x_val"][iva], d["y_val"][iva], L, cfg))
    return clients, L

def topk_mask(d, ratio):
    df = np.asarray((d["x_train"] > 0).sum(axis=0)).ravel()   # document frequency
    k = max(1, int(round(ratio * df.size)))
    idx = np.argsort(-df)[:k]
    m = np.zeros(df.size, dtype=bool); m[idx] = True
    return m

def aggregate(clients, mask):
    L = clients[0].n_labels
    tp = np.zeros(L); fp = np.zeros(L); fn = np.zeros(L)
    for c in clients:
        r = c.evaluate_mask(mask, mode="full")
        tp += r["tp"]; fp += r["fp"]; fn += r["fn"]
    return tp, fp, fn

def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "PubMedMeSH"
    ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.10
    d = load_fed_natural_split("data/fed_real", name, seed=0)
    clients, L = build_clients(d)
    mask = topk_mask(d, ratio)
    tp, fp, fn = aggregate(clients, mask)
    macro0 = compute_macro_f1(tp, fp, fn); micro0 = compute_micro_f1(tp, fp, fn)
    print(f"[D] {name}: {len(clients)} clients, L={L}, ratio={ratio:.2f}, "
          f"NON-PRIVATE macro={macro0:.4f} micro={micro0:.4f}")

    delta = 1e-5
    sens = np.sqrt(L)                       # L2 sensitivity of the 3L count vector
    rng = np.random.default_rng(0)
    epsilons = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    n_trials = 400
    rows = [{"eps": "inf", "macro_mean": round(macro0,4), "macro_std": 0.0,
             "micro_mean": round(micro0,4), "micro_std": 0.0}]
    for eps in epsilons:
        sigma = sens * np.sqrt(2.0 * np.log(1.25 / delta)) / eps
        mas, mis = [], []
        for _ in range(n_trials):
            tpn = np.clip(tp + rng.normal(0, sigma, L), 0, None)
            fpn = np.clip(fp + rng.normal(0, sigma, L), 0, None)
            fnn = np.clip(fn + rng.normal(0, sigma, L), 0, None)
            mas.append(compute_macro_f1(tpn, fpn, fnn))
            mis.append(compute_micro_f1(tpn, fpn, fnn))
        rows.append({"eps": eps, "macro_mean": round(float(np.mean(mas)),4),
                     "macro_std": round(float(np.std(mas)),4),
                     "micro_mean": round(float(np.mean(mis)),4),
                     "micro_std": round(float(np.std(mis)),4)})
        print(f"[D] eps={eps:5.1f} sigma={sigma:7.2f}  macro={np.mean(mas):.4f}±{np.std(mas):.4f}  "
              f"micro={np.mean(mis):.4f}±{np.std(mis):.4f}")
    out = PROJECT / f"reports/exp_dp_{name}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"[D] wrote {out}")

if __name__ == "__main__":
    raise SystemExit(main())
