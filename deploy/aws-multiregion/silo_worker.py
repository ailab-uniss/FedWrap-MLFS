#!/usr/bin/env python3
"""Resident FedWrap-MLFS silo evaluator (a persistent Flower-like client).

Loads its LOCAL shard ONCE, then serves the server over stdin (one line per request):
    prep                      -> per-client SUFFICIENT STATISTICS for the federated relevance sketch
                                 and the filter seeds (Y^T X, sum X, sum X^2, label counts, n, local
                                 relevance). No raw features leave the silo -- only aggregatable stats.
    eval                      -> evaluate the pre-deployed mask.npy (fixed-mask timing runs)
    eval <i,j,k,...>          -> evaluate the subset with those feature indices ON (the search sends this)
    quit
`eval` returns one JSON line with the label-wise TP/FP/FN counters (+ n_val).
"""
import json
import sys

import numpy as np

sys.path.insert(0, ".")
from fedwrap.federated.client import ClientEvalConfig, FederatedClient
from workflow.shard_io import load_shard

x_tr, y_tr, x_va, y_va = load_shard("shard.npz")
D = int(x_tr.shape[1])
cfg = ClientEvalConfig(kind="mlknn", k=5, s=1.0, mlknn_backend="sklearn", mlknn_device="cpu")
client = FederatedClient(0, x_tr, y_tr, x_va, y_va, n_labels=y_tr.shape[1], cfg=cfg)
try:
    default_mask = np.asarray(np.load("mask.npy"), dtype=bool)
except Exception:
    default_mask = None

sys.stdout.write(f"READY {D}\n")
sys.stdout.flush()
for line in sys.stdin:
    line = line.strip()
    if line == "quit":
        break
    if line == "prep":
        from fedwrap.federated.baselines import _local_relevance
        X = x_tr.tocsr(); Y = y_tr.tocsr()
        YtX = np.asarray((Y.T @ X).todense(), dtype=float)
        out = {"YtX": YtX.tolist(),
               "sumX": np.asarray(X.sum(axis=0)).ravel().tolist(),
               "sumX2": np.asarray(X.multiply(X).sum(axis=0)).ravel().tolist(),
               "pos": np.asarray(Y.sum(axis=0)).ravel().tolist(),
               "n": int(X.shape[0]),
               "lrel": [float(v) for v in _local_relevance(x_tr, y_tr)]}
        sys.stdout.write(json.dumps(out) + "\n"); sys.stdout.flush()
        continue
    if not line.startswith("eval"):
        continue
    arg = line[4:].strip()
    if arg:
        mask = np.zeros(D, dtype=bool)
        mask[np.array(arg.split(","), dtype=int)] = True
    else:
        mask = default_mask
    r = client.evaluate_mask(mask, mode="full")
    sys.stdout.write(json.dumps({"tp": [int(v) for v in r["tp"]],
                                 "fp": [int(v) for v in r["fp"]],
                                 "fn": [int(v) for v in r["fn"]],
                                 "n_val": int(r["n_val"])}) + "\n")
    sys.stdout.flush()
