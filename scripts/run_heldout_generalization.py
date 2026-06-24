"""Held-out-client generalization (the strongest federated check).

Question: does a feature subset selected using only some silos still work on silos that
never participated in the optimization?

Protocol per federation:
  1. FedWrap-MLFS (faport) runs its full federated selection on the OPTIMIZATION silos
     only (the {DS}__opt datasets built by build_heldout_silo_splits.py), for seeds 0,1,2.
  2. The val-selected mask is then scored, under the deployment protocol, on
       - the SEEN (optimization) silos, and
       - the HELD-OUT silos, which never influenced the search.
     Each silo trains its own ML-kNN on the selected features and tests on its own split;
     global macro-F1 is the count-aggregated exact objective over the silo group.
A small seen->held-out gap means the selected subset is genuinely informative for the
federation, not overfitted to the participating silos.

Usage: python scripts/run_heldout_generalization.py [select|eval|both] [seeds csv]
"""
import sys, json, copy, glob, csv
import numpy as np
from pathlib import Path
sys.path.insert(0, ".")
import yaml
from fedwrap.datasets import _load_npz_any
from fedwrap.ml_eval import EvalConfig
from fedwrap.federated import make_evaluator
from fedwrap.experiment import run_experiment_from_config

MANIFEST = json.load(open("reports/heldout_silo_manifest.json"))
objn = ["one_minus_macro_f1", "one_minus_micro_f1", "feature_ratio"]
BASE = yaml.safe_load(open("configs/main_bench.yaml"))


def setk(d, dot, v):
    p = dot.split("."); c = d
    for k in p[:-1]:
        c = c.setdefault(k, {})
    c[p[-1]] = v


def run_selection(seeds):
    for ds, info in MANIFEST.items():
        optds = info["opt_dataset"]
        nclients = info["n_opt"]
        for seed in seeds:
            out = f"runs/heldout/{optds}_faport_s{seed}"
            if Path(out + "_fold0", "population_masks.npz").exists():
                print(f"skip (exists) {out}", flush=True); continue
            cfg = copy.deepcopy(BASE); setk(cfg, "seed", seed)
            setk(cfg, "dataset.kind", "prefold"); setk(cfg, "dataset.root", "data/fed_real")
            setk(cfg, "dataset.name", optds)
            setk(cfg, "cross_validation.enabled", True); setk(cfg, "cross_validation.n_folds", 1)
            setk(cfg, "model.mlknn_backend", "sklearn"); setk(cfg, "model.mlknn_device", "cpu")
            setk(cfg, "model.k", 5); setk(cfg, "model.cv_folds", 1)
            setk(cfg, "evolution.genotype", "bitstring")
            setk(cfg, "fedaware.enabled", True); setk(cfg, "fedaware.stability_tiebreak", True)
            setk(cfg, "fedaware.disagreement_mutation", True); setk(cfg, "fedaware.relevance_warmstart", True)
            setk(cfg, "fedaware.warmstart_frac", 0.3); setk(cfg, "fedaware.filter_seed", True)
            setk(cfg, "fedaware.swap_prob", 0.4)
            cfg.get("evolution", {}).pop("max_evals_per_feature", None)
            setk(cfg, "evolution.max_evals", 300000)
            setk(cfg, "evolution.early_stopping", {"enabled": True, "mode": "window",
                  "window": 10, "rel_tol": 0.002, "patience": 3})
            setk(cfg, "federated", {"enabled": True, "n_clients": nclients, "partition": "natural_silo",
                  "min_samples_per_client": 32, "client_fraction_full": 1.0,
                  "final_eval_all_clients": True, "return_client_metrics": True, "n_jobs": 8})
            setk(cfg, "reporting.max_feature_ratio", 0.25)
            setk(cfg, "logging.out_dir", out)
            try:
                run_experiment_from_config(cfg, fold_idx=0)
                print(f"DONE selection {optds} s{seed}", flush=True)
            except Exception as e:
                print(f"ERR {optds} s{seed}: {repr(e)[:200]}", flush=True)


def best_mask(run_npz):
    z = np.load(run_npz); D2 = int(z["n_features"])
    M = np.unpackbits(z["masks_packed"], axis=1)[:, :D2].astype(bool)
    v, on = z["val_objs"], z["pareto_val_mask"].astype(bool); idx = np.flatnonzero(on)
    cap = M.sum(1) <= 0.25 * D2 + 1; cand = idx[cap[idx]] if cap[idx].any() else idx
    bi = cand[np.argmin(v[cand, 0])]
    return M[bi], float(M[bi].sum() / D2)


def macro_on_silos(ds, silo_set, mask, nclients_hint):
    """Global (count-aggregated) macro-F1 of a fixed mask over the given silos, under the
    deployment protocol (each silo trains its own ML-kNN locally and tests on its split)."""
    base = Path(f"data/fed_real/{ds}/fold0")
    xtr, ytr = _load_npz_any(base / "trainval.npz")
    xte, yte = _load_npz_any(base / "test.npz")
    gtr = np.load(base / "trainval_groups.npy", allow_pickle=True)
    gte = np.load(base / "test_groups.npy", allow_pickle=True)
    keep_tr = np.array([str(g) in silo_set for g in gtr.tolist()])
    keep_te = np.array([str(g) in silo_set for g in gte.tolist()])
    ec = EvalConfig(kind="mlknn", primary_objective=objn[0], objective_names=objn,
                    random_state=0, k=5, s=1.0, mlknn_backend="sklearn", mlknn_device="cpu")
    cfg = {"federated": {"enabled": True, "n_clients": len(silo_set), "partition": "natural_silo",
            "min_samples_per_client": 32, "client_fraction_full": 1.0, "final_eval_all_clients": True},
           "objectives": {"names": objn}}
    ev = make_evaluator(xtr[keep_tr], ytr[keep_tr], xte[keep_te], yte[keep_te], ec, cfg, 0,
                        groups=(gtr[keep_tr], gte[keep_te]))
    _, r = ev.evaluate_mask(mask)
    return float(r.f1_macro)


def run_eval(seeds):
    rows = []
    for ds, info in MANIFEST.items():
        optds = info["opt_dataset"]
        seen = set(info["optimization_silos"]); held = set(info["heldout_silos"])
        seen_l, held_l, ratios = [], [], []
        for seed in seeds:
            rn = f"runs/heldout/{optds}_faport_s{seed}_fold0/population_masks.npz"
            if not Path(rn).exists():
                print(f"  missing {rn}", flush=True); continue
            mask, rt = best_mask(rn); ratios.append(rt)
            s = macro_on_silos(ds, seen, mask, info["n_opt"])
            h = macro_on_silos(ds, held, mask, info["n_held"])
            seen_l.append(s); held_l.append(h)
            print(f"  {ds} s{seed}: ratio={rt:.3f} seen={s:.3f} held-out={h:.3f} gap={s-h:+.3f}", flush=True)
        if not seen_l:
            continue
        sm, ss = np.mean(seen_l), np.std(seen_l)
        hm, hs = np.mean(held_l), np.std(held_l)
        # control: all-features on the SAME held-out silos -- does a subset chosen without
        # those silos still beat using every feature on them.
        D_full = np.load(f"data/fed_real/{ds}/fold0/trainval.npz")["X_shape"][1]
        af_held = macro_on_silos(ds, held, np.ones(int(D_full), bool), info["n_held"])
        print(f"== {ds}: SEEN {sm:.3f}+/-{ss:.3f}  HELD-OUT {hm:.3f}+/-{hs:.3f}  gap {sm-hm:+.3f} "
              f"| all-feat held-out {af_held:.3f} (n={len(seen_l)} seeds) ==", flush=True)
        rows.append({"dataset": ds, "opt_silos": info["n_opt"], "heldout_silos": info["n_held"],
                     "ratio": round(float(np.mean(ratios)), 3),
                     "seen_macro": round(float(sm), 4), "seen_std": round(float(ss), 4),
                     "heldout_macro": round(float(hm), 4), "heldout_std": round(float(hs), 4),
                     "heldout_allfeat": round(float(af_held), 4), "gap": round(float(sm - hm), 4)})
    with open("reports/heldout_generalization.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "opt_silos", "heldout_silos", "ratio",
                            "seen_macro", "seen_std", "heldout_macro", "heldout_std",
                            "heldout_allfeat", "gap"])
        w.writeheader(); w.writerows(rows)
    print("wrote reports/heldout_generalization.csv", flush=True)


def main():
    phase = sys.argv[1] if len(sys.argv) > 1 else "both"
    seeds = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0, 1, 2]
    if phase in ("select", "both"):
        run_selection(seeds)
    if phase in ("eval", "both"):
        run_eval(seeds)


if __name__ == "__main__":
    raise SystemExit(main())
