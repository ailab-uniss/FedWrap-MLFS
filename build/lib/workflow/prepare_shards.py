"""Materialize per-client data shards for the FedWrap-MLFS workflow.

Splits a natural-silo federation into one shard file per client (silo): each client's rows are split
into a local train/val partition and written as a single ``.npz`` (see ``shard_io``). These shards are
the only data inputs the CWL workflow needs; in deployment each lives behind its client's boundary.
A ``manifest.json`` records the feature dimension, label dimension, and client list.

NOTE: eICU shards derive from PhysioNet credentialed data; they may be used on the cluster provided
every person with access is individually PhysioNet-credentialed (CITI + DUA) and the files are
restricted to credentialed users (not world-readable). ECG/ExtraSensory/synthetic carry no such limit.

Usage: python prepare_shards.py --dataset ECG_cinc2021 --root data/fed_real --out workflow/shards --val 0.25
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fedwrap.datasets import _load_npz_any
from workflow.shard_io import save_shard


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--root", default="data/fed_real")
    ap.add_argument("--out", default="workflow/shards")
    ap.add_argument("--val", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    fold = Path(a.root) / a.dataset / "fold0"
    x, y = _load_npz_any(fold / "trainval.npz")
    x, y = x.tocsr(), y.tocsr()
    groups = np.load(fold / "trainval_groups.npy", allow_pickle=True)
    out_dir = Path(a.out) / a.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    client_files = []
    for cid, g in enumerate(sorted(set(groups.tolist()))):
        rows = np.flatnonzero(groups == g)
        perm = rng.permutation(rows)
        n_val = max(1, int(a.val * len(perm)))
        va, tr = perm[:n_val], perm[n_val:]
        if len(tr) == 0:
            continue
        fpath = out_dir / f"client_{cid}.npz"
        save_shard(str(fpath), x[tr], y[tr], x[va], y[va])
        client_files.append({"client_id": cid, "group": str(g), "n_train": int(len(tr)),
                             "n_val": int(len(va)), "file": fpath.name})

    manifest = {"dataset": a.dataset, "n_features": int(x.shape[1]), "n_labels": int(y.shape[1]),
                "n_clients": len(client_files), "val_fraction": a.val, "clients": client_files}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(client_files)} shards for {a.dataset} (D={x.shape[1]}, L={y.shape[1]}) -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
