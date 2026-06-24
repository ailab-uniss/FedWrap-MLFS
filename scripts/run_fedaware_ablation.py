"""FedAware-NSGA-II component ablation (graph-free, flat-mask wrapper).

Cumulative arms on a natural-silo federation:
  base        : plain federated NSGA-II wrapper (no federation-aware operators)
  +stability  : + client-stability tie-break (prefer low inter-client dispersion)
  full        : + disagreement/label-error-guided mutation  (= full FedAware-NSGA-II)

Per arm/seed we record val-selected test macro-F1, micro-F1, feature ratio, the number of FULL
federated evaluations, and the worst-client macro-F1. Reads each run's summary.json + masks.

Usage: python scripts/run_fedaware_ablation.py <dataset> <seeds csv>
"""
import sys, copy, glob, csv, json
import numpy as np
from pathlib import Path
sys.path.insert(0, ".")
import yaml
from fedwrap.experiment import run_experiment_from_config
sys.path.insert(0, str(Path(__file__).resolve().parent))
from synth_scaling import materialize

# Heterogeneous synthetic regime for the ablation (where the federation-aware components
# differentiate): moderate dim, many clients, strong non-IID label skew.
SYNTH_HET = dict(N=4000, D=300, L=20, K=16, informative_ratio=0.10, noise=0.3, alpha=0.1, n_informative=30)

BASE = yaml.safe_load(open("configs/main_bench.yaml"))
# Cumulative arms. The federation-aware OPERATORS are added at full evaluation budget first
# 
ARMS = {
    "base":          dict(fa=False, stability=False, disagreement=False),
    "+stability":    dict(fa=True,  stability=True,  disagreement=False),
    "+disagreement": dict(fa=True,  stability=True,  disagreement=True),   # = headline FedAware
}


def setk(d, dot, v):
    p = dot.split("."); c = d
    for k in p[:-1]:
        c = c.setdefault(k, {})
    c[p[-1]] = v


def run_arm(ds, root, name, nclients, arm, a, seed):
    out = f"runs/fedaware_ablation/{ds}_{arm}_s{seed}"
    if Path(out + "_fold0", "summary.json").exists():
        return out + "_fold0"
    cfg = copy.deepcopy(BASE); setk(cfg, "seed", seed)
    setk(cfg, "dataset.kind", "prefold"); setk(cfg, "dataset.root", root); setk(cfg, "dataset.name", name)
    setk(cfg, "cross_validation.enabled", True); setk(cfg, "cross_validation.n_folds", 1)
    setk(cfg, "model.kind", "mlknn"); setk(cfg, "model.mlknn_backend", "sklearn"); setk(cfg, "model.mlknn_device", "cpu")
    setk(cfg, "model.k", 5); setk(cfg, "model.cv_folds", 1)
    setk(cfg, "evolution.genotype", "bitstring")
    cfg.get("evolution", {}).pop("max_evals_per_feature", None)
    setk(cfg, "evolution.max_evals", 300000)
    setk(cfg, "evolution.early_stopping", {"enabled": True, "mode": "window", "window": 10, "rel_tol": 0.002, "patience": 3})
    setk(cfg, "fedaware.enabled", a["fa"])
    setk(cfg, "fedaware.stability_tiebreak", a["stability"])
    setk(cfg, "fedaware.disagreement_mutation", a["disagreement"])
    setk(cfg, "federated", {"enabled": True, "n_clients": nclients, "partition": "natural_silo",
          "min_samples_per_client": 32, "client_fraction_full": 1.0, "final_eval_all_clients": True,
          "return_client_metrics": True, "estimate_communication": True,
          "n_jobs": 8})
    setk(cfg, "reporting.max_feature_ratio", 0.25)
    setk(cfg, "logging.out_dir", out)
    try:
        run_experiment_from_config(cfg, fold_idx=0)
    except Exception as e:
        print(f"ERR {ds} {arm} s{seed}: {repr(e)[:200]}", flush=True); return None
    return out + "_fold0"


def best_ratio(run_dir):
    z = np.load(Path(run_dir) / "population_masks.npz"); D2 = int(z["n_features"])
    M = np.unpackbits(z["masks_packed"], axis=1)[:, :D2].astype(bool)
    v, on = z["val_objs"], z["pareto_val_mask"].astype(bool); idx = np.flatnonzero(on)
    cap = M.sum(1) <= 0.25 * D2 + 1; cand = idx[cap[idx]] if cap[idx].any() else idx
    bi = cand[np.argmin(v[cand, 0])]
    return float(M[bi].sum() / D2)


def main():
    ds = sys.argv[1] if len(sys.argv) > 1 else "eICU_expl_k12"
    seeds = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0]
    is_synth = ds == "synth_het"
    if not is_synth:
        nclients = len(set(np.load(f"data/fed_real/{ds}/fold0/trainval_groups.npy", allow_pickle=True).tolist()))
    rows = []
    for arm, a in ARMS.items():
        macro, micro, worst, ratio, fevals = [], [], [], [], []
        for seed in seeds:
            if is_synth:
                p = SYNTH_HET; name = f"synth_het_s{seed}"; root = "data/synth_scaling"
                if not Path(f"{root}/{name}/fold0/trainval.npz").exists():
                    materialize(f"{root}/{name}", N=p["N"], D=p["D"], L=p["L"], K=p["K"],
                        informative_ratio=p["informative_ratio"], noise=p["noise"], alpha=p["alpha"],
                        n_informative=p["n_informative"], seed=seed)
                nclients = p["K"]
            else:
                root, name = "data/fed_real", ds
            rd = run_arm(ds, root, name, nclients, arm, a, seed)
            if rd is None or not Path(rd, "summary.json").exists():
                continue
            s = json.loads(Path(rd, "summary.json").read_text())
            fin = s.get("final", {})
            macro.append(fin.get("macro_f1_best")); micro.append(fin.get("micro_f1_best"))
            worst.append(fin.get("worst_client_macro_f1")); ratio.append(best_ratio(rd))
            fe = s.get("federated", {}).get("counters", {}).get("full_evals")
            fevals.append(fe if fe is not None else fin.get("total_evals"))
        macro = [m for m in macro if m is not None]
        if not macro:
            continue
        row = {"dataset": ds, "arm": arm, "seeds": len(macro),
               "macro": round(float(np.mean(macro)), 4), "macro_std": round(float(np.std(macro)), 4),
               "micro": round(float(np.mean([m for m in micro if m is not None])), 4),
               "feat_ratio": round(float(np.mean(ratio)), 4),
               "full_evals": int(np.mean([f for f in fevals if f is not None])),
               "worst_client": round(float(np.mean([w for w in worst if w is not None])), 4)}
        rows.append(row)
        print(f"  [{arm:11s}] macro={row['macro']}±{row['macro_std']} micro={row['micro']} "
              f"ratio={row['feat_ratio']} full_evals={row['full_evals']} worst={row['worst_client']}", flush=True)
    out = f"reports/fedaware_ablation_{ds}.csv"
    if rows:
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
