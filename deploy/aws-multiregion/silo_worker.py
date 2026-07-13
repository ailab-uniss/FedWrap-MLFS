#!/usr/bin/env python3
"""Resident FedWrap-MLFS silo evaluator (models a persistent Flower client).

Loads its LOCAL shard and the broadcast mask ONCE (Python/sklearn import + data load happen a single
time), then on every ``eval`` line from stdin re-runs the client evaluation and prints the label-wise
TP/FP/FN counters as one JSON line. Keeping import/load out of the per-round loop means a timed round
measures the real network round-trip + inference -- the steady-state cost in a deployed federation,
which is where the resource-aware scheduler's latency disparity actually shows.
"""
import json
import sys

import numpy as np

sys.path.insert(0, ".")
from fedwrap.federated.client import ClientEvalConfig, FederatedClient
from workflow.shard_io import load_shard

x_tr, y_tr, x_va, y_va = load_shard("shard.npz")
mask = np.asarray(np.load("mask.npy"), dtype=bool)
cfg = ClientEvalConfig(kind="mlknn", k=5, s=1.0, mlknn_backend="sklearn", mlknn_device="cpu")
client = FederatedClient(0, x_tr, y_tr, x_va, y_va, n_labels=y_tr.shape[1], cfg=cfg)

sys.stdout.write("READY\n")
sys.stdout.flush()
for line in sys.stdin:
    cmd = line.strip()
    if cmd == "eval":
        r = client.evaluate_mask(mask, mode="full")
        sys.stdout.write(json.dumps({"tp": [int(v) for v in r["tp"]],
                                     "fp": [int(v) for v in r["fp"]],
                                     "fn": [int(v) for v in r["fn"]]}) + "\n")
        sys.stdout.flush()
    elif cmd == "quit":
        break
