"""Server aggregation step of the FedWrap-MLFS workflow.

Sums the per-client label-wise counters returned by the client steps and reconstructs the EXACT
global micro/macro-F1 (identical to scoring the concatenation of the clients' predictions). Because
the objective is a sum of counters, aggregating over whichever clients responded is exact for that
subset---the property the resource-aware scheduler relies on.

Usage: python aggregate.py --counters c0.json c1.json ... --out global.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fedwrap.federated.metrics import compute_macro_f1, compute_micro_f1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--counters", nargs="+", required=True)
    ap.add_argument("--out", default="global.json")
    a = ap.parse_args()

    tp = fp = fn = None; n_val = 0; ids = []
    for c in a.counters:
        d = json.loads(Path(c).read_text())
        t, f, n = np.array(d["tp"]), np.array(d["fp"]), np.array(d["fn"])
        tp = t if tp is None else tp + t
        fp = f if fp is None else fp + f
        fn = n if fn is None else fn + n
        n_val += int(d["n_val"]); ids.append(int(d["client_id"]))
    out = {"macro_f1": float(compute_macro_f1(tp, fp, fn)),
           "micro_f1": float(compute_micro_f1(tp, fp, fn)),
           "n_val": int(n_val), "n_clients": len(ids), "client_ids": sorted(ids)}
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"aggregated {len(ids)} clients: macro={out['macro_f1']:.4f} micro={out['micro_f1']:.4f}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
