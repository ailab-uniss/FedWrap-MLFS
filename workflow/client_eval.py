"""Client step of the FedWrap-MLFS workflow (one CWL task per client).

Loads a client's local shard and a broadcast feature mask, trains the client's local ML-kNN on the
selected features, and writes ONLY the label-wise sufficient statistics (TP/FP/FN) and the local
validation size. This is the exact per-client computation of the federated evaluation step; the
server aggregates these counters (see ``aggregate.py``). No raw features, predictions, or model
parameters are emitted---only the integer counters.

Usage: python client_eval.py --shard client_k.npz --mask mask.npy --k 5 --s 1.0 --out counters_k.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fedwrap.federated.client import ClientEvalConfig, FederatedClient
from workflow.shard_io import load_shard


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--mask", required=True)
    ap.add_argument("--client-id", type=int, default=0)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--s", type=float, default=1.0)
    ap.add_argument("--out", default="counters.json")
    a = ap.parse_args()

    x_tr, y_tr, x_va, y_va = load_shard(a.shard)
    mask = np.asarray(np.load(a.mask), dtype=bool)
    cfg = ClientEvalConfig(kind="mlknn", k=a.k, s=a.s, mlknn_backend="sklearn", mlknn_device="cpu")
    client = FederatedClient(a.client_id, x_tr, y_tr, x_va, y_va, n_labels=y_tr.shape[1], cfg=cfg)
    r = client.evaluate_mask(mask, mode="full")
    payload = {"client_id": a.client_id,
               "tp": np.asarray(r["tp"], dtype=np.int64).tolist(),
               "fp": np.asarray(r["fp"], dtype=np.int64).tolist(),
               "fn": np.asarray(r["fn"], dtype=np.int64).tolist(),
               "n_val": int(r["n_val"])}
    Path(a.out).write_text(json.dumps(payload))
    print(f"client {a.client_id}: n_val={payload['n_val']} selected={int(mask.sum())}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
