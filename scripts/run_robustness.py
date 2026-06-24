"""Resilience of the count-aggregation protocol to faulty/adversarial clients, on the deployed
FedAware-NSGA-II mask. A corrupted client reports flipped predictions (all-positive), inflating its
FP/FN. We compare the plain sum against a drop-$f$ robust aggregator that discards the $f$ clients
whose per-label count rates deviate most from the coordinate-wise median and rescales the rest.
Writes reports/r3_robustness.csv. Datasets/seeds from argv (default ECG, eICU, seeds 0-4).
"""
import sys, glob, csv
import numpy as np
sys.path.insert(0, ".")
from fedwrap.datasets import _load_npz_any
from fedwrap.ml_eval import EvalConfig
from fedwrap.federated import make_evaluator
from fedwrap.federated.metrics import compute_macro_f1

objn = ["one_minus_macro_f1", "one_minus_micro_f1", "feature_ratio"]
DATASETS = [("ECG_cinc2021", 8), ("eICU_expl_k12", 12)]


def best_mask(npz):
    z = np.load(npz); D = int(z["n_features"])
    M = np.unpackbits(z["masks_packed"], axis=1)[:, :D].astype(bool)
    v, on = z["val_objs"], z["pareto_val_mask"].astype(bool); idx = np.flatnonzero(on)
    cap = M.sum(1) <= 0.25 * D + 1; cand = idx[cap[idx]] if cap[idx].any() else idx
    return M[cand[np.argmin(v[cand, 0])]]


def client_counts(ev, mask):
    """per-client (tp,fp,fn) arrays for a mask."""
    out = []
    for c in ev.clients:
        r = c.evaluate_mask(np.asarray(mask, bool), mode="full")
        out.append((r["tp"].astype(float), r["fp"].astype(float), r["fn"].astype(float)))
    return out


def global_macro(counts):
    tp = sum(c[0] for c in counts); fp = sum(c[1] for c in counts); fn = sum(c[2] for c in counts)
    return compute_macro_f1(tp, fp, fn)


def corrupt(counts, f, rng):
    """f Byzantine clients report adversarial counts that minimize the global score: no true
    positives, and a large false-positive/false-negative flood on every label, sized to dominate
    the clean aggregate so global precision and recall collapse."""
    L = counts[0][0].size
    clean_mass = sum((c[0] + c[1] + c[2]).sum() for c in counts)
    big = float(max(1.0, clean_mass / L))          # per-label flood comparable to total clean mass
    counts = [list(c) for c in counts]
    vic = rng.choice(len(counts), size=f, replace=False)
    for i in vic:
        counts[i] = (np.zeros(L), np.full(L, big), np.full(L, big))
    return counts, set(int(v) for v in vic)


def drop_f(counts, f):
    """drop the f clients whose reported count mass deviates most from the coordinate-wise median
    (a Byzantine flooder reports anomalously large counts), then sum the rest. A genuine client's
    total mass is bounded by its data size, so the flood is an outlier the median exposes."""
    K = len(counts)
    mass = np.array([float((c[0] + c[1] + c[2]).sum()) for c in counts])
    dev = np.abs(mass - np.median(mass))
    keep = np.argsort(dev)[: K - f]
    kept = [counts[i] for i in keep]
    tp = sum(c[0] for c in kept); fp = sum(c[1] for c in kept); fn = sum(c[2] for c in kept)
    return compute_macro_f1(tp, fp, fn)


def main():
    rows = []
    for ds, nc in DATASETS:
        xtr, ytr = _load_npz_any(f"data/fed_real/{ds}/fold0/trainval.npz")
        xte, yte = _load_npz_any(f"data/fed_real/{ds}/fold0/test.npz")
        gtr = np.load(f"data/fed_real/{ds}/fold0/trainval_groups.npy", allow_pickle=True)
        gte = np.load(f"data/fed_real/{ds}/fold0/test_groups.npy", allow_pickle=True)
        ec = EvalConfig(kind="mlknn", primary_objective=objn[0], objective_names=objn,
                        random_state=0, k=5, s=1.0, mlknn_backend="sklearn", mlknn_device="cpu")
        cfg = {"federated": {"enabled": True, "n_clients": nc, "partition": "natural_silo",
                "min_samples_per_client": 32, "client_fraction_full": 1.0, "final_eval_all_clients": True},
               "objectives": {"names": objn}}
        ev = make_evaluator(xtr, ytr, xte, yte, ec, cfg, 0, groups=(gtr, gte))
        agg = {"clean": [], "f1_sum": [], "f1_drop": [], "f2_sum": [], "f2_drop": []}
        for npz in sorted(glob.glob(f"runs/plain_fed/{ds}_faport_s*/population_masks.npz")):
            counts = client_counts(ev, best_mask(npz))
            rng = np.random.default_rng(0)
            agg["clean"].append(global_macro(counts))
            for f, tagsum, tagdrop in [(1, "f1_sum", "f1_drop"), (2, "f2_sum", "f2_drop")]:
                cc, _ = corrupt(counts, f, np.random.default_rng(1))
                agg[tagsum].append(global_macro(cc))
                agg[tagdrop].append(drop_f(cc, f))
        row = {"dataset": ds, "K": nc, **{k: round(float(np.mean(v)), 3) for k, v in agg.items()}}
        rows.append(row)
        print(f"{ds} (K={nc}): clean={row['clean']} | f1 sum={row['f1_sum']} drop={row['f1_drop']} "
              f"| f2 sum={row['f2_sum']} drop={row['f2_drop']}", flush=True)
    with open("reports/r3_robustness.csv", "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=["dataset", "K", "clean", "f1_sum", "f1_drop", "f2_sum", "f2_drop"])
        w.writeheader(); w.writerows(rows)
    print("wrote reports/r3_robustness.csv", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
