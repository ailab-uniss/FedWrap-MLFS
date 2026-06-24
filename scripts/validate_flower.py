"""Does the Flower backend actually reproduce the in-process FederatedEvaluator?

Build the ECG natural-silo federation in-process, extract each client's exact shards, run ONE Flower
round on the deployed faport mask through run_federated_eval_flower, and compare the aggregated
TP/FP/FN and global macro/micro-F1 against the in-process evaluator on the identical mask and data.
"""
import sys
import numpy as np
sys.path.insert(0, ".")
from fedwrap.datasets import _load_npz_any
from fedwrap.ml_eval import EvalConfig
from fedwrap.federated import make_evaluator
from fedwrap.federated.flower_backend import run_federated_eval_flower

DS, NC = "ECG_cinc2021", 8
objn = ["one_minus_macro_f1", "one_minus_micro_f1", "feature_ratio"]


def valbest(z):
    D = int(z["n_features"]); M = np.unpackbits(z["masks_packed"], axis=1)[:, :D].astype(bool)
    vo, on = z["val_objs"], z["pareto_val_mask"].astype(bool); idx = np.flatnonzero(on)
    cap = M.sum(1) <= 0.25 * D + 1; cand = idx[cap[idx]] if cap[idx].any() else idx
    return M[cand[np.argmin(vo[cand, 0])]]


def main():
    import glob
    xtr, ytr = _load_npz_any(f"data/fed_real/{DS}/fold0/trainval.npz")
    xte, yte = _load_npz_any(f"data/fed_real/{DS}/fold0/test.npz")
    gtr = np.load(f"data/fed_real/{DS}/fold0/trainval_groups.npy", allow_pickle=True)
    gte = np.load(f"data/fed_real/{DS}/fold0/test_groups.npy", allow_pickle=True)
    L = ytr.shape[1]
    ec = EvalConfig(kind="mlknn", primary_objective=objn[0], objective_names=objn, random_state=0,
                    k=5, s=1.0, mlknn_backend="sklearn", mlknn_device="cpu")
    cfg = {"federated": {"enabled": True, "n_clients": NC, "partition": "natural_silo",
           "min_samples_per_client": 32, "client_fraction_full": 1.0, "final_eval_all_clients": True},
           "objectives": {"names": objn}}
    ev = make_evaluator(xtr, ytr, xte, yte, ec, cfg, 0, groups=(gtr, gte))
    mask = valbest(np.load(sorted(glob.glob(f"runs/plain_fed/{DS}_faport_s*/population_masks.npz"))[0]))

    # in-process aggregated counts
    tp = np.zeros(L); fp = np.zeros(L); fn = np.zeros(L)
    for c in ev.clients:
        r = c.evaluate_mask(np.asarray(mask, bool), mode="full")
        tp += r["tp"]; fp += r["fp"]; fn += r["fn"]
    from fedwrap.federated.metrics import compute_macro_f1, compute_micro_f1
    ip = {"tp": tp, "fp": fp, "fn": fn, "macro": float(compute_macro_f1(tp, fp, fn)),
          "micro": float(compute_micro_f1(tp, fp, fn))}

    # same data through Flower (one round)
    shards = [(c.x_train, c.y_train, c.x_val, c.y_val) for c in ev.clients]
    fl = run_federated_eval_flower(shards, np.asarray(mask, bool), n_labels=L, k=5, s=1.0, backend="sklearn")

    d_tp = float(np.max(np.abs(fl["tp"] - ip["tp"])))
    d_fp = float(np.max(np.abs(fl["fp"] - ip["fp"])))
    d_fn = float(np.max(np.abs(fl["fn"] - ip["fn"])))
    print(f"\n=== Flower vs in-process ({DS}, K={len(ev.clients)}, L={L}, |mask|={int(mask.sum())}) ===")
    print(f"  in-process : macro={ip['macro']:.6f} micro={ip['micro']:.6f}")
    print(f"  Flower     : macro={fl['macro_f1']:.6f} micro={fl['micro_f1']:.6f}")
    print(f"  max|dTP|={d_tp:g} max|dFP|={d_fp:g} max|dFN|={d_fn:g}")
    print(f"  |dmacro|={abs(fl['macro_f1']-ip['macro']):.2e} |dmicro|={abs(fl['micro_f1']-ip['micro']):.2e}")
    ok = max(d_tp, d_fp, d_fn) == 0 and abs(fl["macro_f1"] - ip["macro"]) < 1e-9
    print(f"  RESULT: {'EXACT MATCH' if ok else 'MISMATCH'}")


if __name__ == "__main__":
    raise SystemExit(main())
