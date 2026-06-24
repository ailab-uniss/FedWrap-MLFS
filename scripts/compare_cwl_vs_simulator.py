"""Verify the CWL/StreamFlow workflow reproduces the in-process simulator exactly (cluster deliverable ii).

Runs the SAME per-client evaluation IN-PROCESS on the shards (no containers, no JSON serialization) and
aggregates the label-wise counts, then compares the global macro/micro-F1 against the workflow's
``global.json`` (produced by running the CWL workflow on the same shards + mask). Because the counters
are integers, an exact match (max |delta| < 1e-9) confirms the distributed cross-node execution is
numerically identical to the fast simulator -- the claim the HPC-cluster section needs to substantiate.

Usage:
  # 1. run the workflow to get global.json (locally with cwltool, or on the cluster with StreamFlow):
  python workflow/make_job.py workflow/shards/<ds> <mask.npy> > job.yml
  cwltool --outdir out workflow/cwl/fed_eval.cwl job.yml
  # 2. confirm it matches the in-process simulator on the same shards:
  python scripts/compare_cwl_vs_simulator.py --shards workflow/shards/<ds> --mask <mask.npy> --cwl-output out/global.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from fedwrap.federated.client import ClientEvalConfig, FederatedClient
from fedwrap.federated.metrics import compute_macro_f1, compute_micro_f1
from workflow.shard_io import load_shard


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True, help="dir with client_*.npz (built by prepare_shards.py)")
    ap.add_argument("--mask", required=True, help="boolean feature mask .npy (length D)")
    ap.add_argument("--cwl-output", required=True, help="global.json produced by the CWL workflow")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--s", type=float, default=1.0)
    a = ap.parse_args()

    mask = np.asarray(np.load(a.mask), dtype=bool)
    cfg = ClientEvalConfig(kind="mlknn", k=a.k, s=a.s, mlknn_backend="sklearn", mlknn_device="cpu")
    shards = sorted(Path(a.shards).glob("client_*.npz"))
    if not shards:
        print(f"no client_*.npz under {a.shards}", file=sys.stderr); return 2

    tp = fp = fn = None
    for i, sh in enumerate(shards):
        x_tr, y_tr, x_va, y_va = load_shard(str(sh))
        c = FederatedClient(i, x_tr, y_tr, x_va, y_va, n_labels=y_tr.shape[1], cfg=cfg)
        r = c.evaluate_mask(mask, mode="full")
        t, f, n = np.asarray(r["tp"]), np.asarray(r["fp"]), np.asarray(r["fn"])
        tp = t if tp is None else tp + t
        fp = f if fp is None else fp + f
        fn = n if fn is None else fn + n
    ip_macro = float(compute_macro_f1(tp, fp, fn))
    ip_micro = float(compute_micro_f1(tp, fp, fn))

    cwl = json.loads(Path(a.cwl_output).read_text())
    dmac = abs(ip_macro - float(cwl["macro_f1"]))
    dmic = abs(ip_micro - float(cwl["micro_f1"]))
    ok = max(dmac, dmic) < 1e-9
    print(f"in-process simulator   : macro={ip_macro:.10f}  micro={ip_micro:.10f}  ({len(shards)} shards)")
    print(f"CWL workflow (cross-node): macro={float(cwl['macro_f1']):.10f}  micro={float(cwl['micro_f1']):.10f}")
    print(f"max |delta|            : macro={dmac:.2e}  micro={dmic:.2e}  -> "
          f"{'EXACT MATCH (<1e-9)' if ok else 'MISMATCH'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
