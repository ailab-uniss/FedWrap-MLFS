"""Interaction campaign: as the interaction content of the labels grows, only a subset-evaluating
wrapper recovers the interaction features that per-feature filters miss. Sweeps interaction_frac at
fixed D, comparing the FedWrap search (faport) and base NSGA-II against fed-rank, FMLFS, all-features,
and the true-subset ceiling, over several seeds. Writes reports/interaction_sweep.csv (resumable).

Usage: python scripts/run_interaction_campaign.py <interaction_frac> <seeds csv>
"""
import sys, csv
import numpy as np
from pathlib import Path
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import yaml, io, contextlib
from synth_scaling import materialize
from fedwrap.datasets import _load_npz_any
from fedwrap.ml_eval import EvalConfig
from fedwrap.federated import make_evaluator
from fedwrap.federated.baselines import fed_rank_relevance, ranking_to_mask, fmlfs
from fedwrap.experiment import run_experiment_from_config

objn = ["one_minus_macro_f1", "one_minus_micro_f1", "feature_ratio"]
D, L, K, NINF, SIG, NOISE = 300, 12, 8, 16, 8.0, 0.1
FINT = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
SEEDS = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0, 1, 2, 3, 4]
OUT = "reports/interaction_sweep.csv"


def setk(d, dot, v):
    p = dot.split("."); c = d
    for k in p[:-1]:
        c = c.setdefault(k, {})
    c[p[-1]] = v


def valbest(rundir):
    z = np.load(Path(rundir) / "population_masks.npz"); Dz = int(z["n_features"])
    M = np.unpackbits(z["masks_packed"], axis=1)[:, :Dz].astype(bool)
    vo, on = z["val_objs"], z["pareto_val_mask"].astype(bool); idx = np.flatnonzero(on)
    cap = M.sum(1) <= 0.25 * Dz + 1; cand = idx[cap[idx]] if cap[idx].any() else idx
    return 1.0 - float(z["test_objs"][cand[np.argmin(vo[cand, 0])], 0])


def search(name, method, seed):
    out = f"runs/synth_int/{name}_{method}"
    if Path(out + "_fold0", "population_masks.npz").exists():
        return valbest(out + "_fold0")
    cfg = yaml.safe_load(open("configs/main_bench.yaml")); setk(cfg, "seed", seed)
    setk(cfg, "dataset.kind", "prefold"); setk(cfg, "dataset.root", "data/synth_int"); setk(cfg, "dataset.name", name)
    setk(cfg, "cross_validation.enabled", True); setk(cfg, "cross_validation.n_folds", 1)
    setk(cfg, "model.kind", "mlknn"); setk(cfg, "model.mlknn_backend", "sklearn"); setk(cfg, "model.mlknn_device", "cpu")
    setk(cfg, "model.k", 5); setk(cfg, "model.cv_folds", 1); setk(cfg, "evolution.genotype", "bitstring")
    cfg.get("evolution", {}).pop("max_evals_per_feature", None)
    setk(cfg, "evolution.max_evals", 300000)
    setk(cfg, "evolution.early_stopping", {"enabled": True, "mode": "window", "window": 10, "rel_tol": 0.002, "patience": 3})
    fp = method == "faport"
    setk(cfg, "fedaware.enabled", fp); setk(cfg, "fedaware.relevance_warmstart", fp)
    setk(cfg, "fedaware.warmstart_frac", 0.3); setk(cfg, "fedaware.filter_seed", fp); setk(cfg, "fedaware.swap_prob", 0.4)
    setk(cfg, "federated", {"enabled": True, "n_clients": K, "partition": "natural_silo",
         "min_samples_per_client": 16, "client_fraction_full": 1.0, "final_eval_all_clients": True, "n_jobs": 4})
    setk(cfg, "reporting.max_feature_ratio", 0.25); setk(cfg, "logging.out_dir", out)
    run_experiment_from_config(cfg, fold_idx=0)
    return valbest(out + "_fold0")


def main():
    done = set()
    if Path(OUT).exists():
        for r in csv.DictReader(open(OUT)):
            done.add((r["interaction_frac"], r["seed"]))
    f = open(OUT, "a", newline="")
    w = csv.DictWriter(f, fieldnames=["interaction_frac", "seed", "D", "true_subset", "faport", "base",
                                      "fed_rank", "fmlfs", "all_features"])
    if f.tell() == 0:
        w.writeheader(); f.flush()
    for seed in SEEDS:
        if (str(FINT), str(seed)) in done:
            print(f"skip frac={FINT} s{seed}", flush=True); continue
        name = f"int_D{D}_f{FINT}_s{seed}"
        materialize(f"data/synth_int/{name}", N=4000, D=D, L=L, K=K, informative_ratio=0.1, noise=NOISE,
                    alpha=1.0, seed=seed, n_informative=NINF, interaction_frac=FINT, signal_strength=SIG)
        xtr, ytr = _load_npz_any(f"data/synth_int/{name}/fold0/trainval.npz")
        xte, yte = _load_npz_any(f"data/synth_int/{name}/fold0/test.npz")
        gtr = np.load(f"data/synth_int/{name}/fold0/trainval_groups.npy", allow_pickle=True)
        gte = np.load(f"data/synth_int/{name}/fold0/test_groups.npy", allow_pickle=True)
        inf = np.load(f"data/synth_int/{name}/true_informative.npy")
        ec = EvalConfig(kind="mlknn", primary_objective=objn[0], objective_names=objn, random_state=0, k=5, s=1.0,
                        mlknn_backend="sklearn", mlknn_device="cpu")
        ev = make_evaluator(xtr, ytr, xte, yte, ec, {"federated": {"enabled": True, "n_clients": K,
             "partition": "natural_silo", "min_samples_per_client": 16, "client_fraction_full": 1.0,
             "final_eval_all_clients": True}, "objectives": {"names": objn}}, 0, groups=(gtr, gte))
        r = len(inf) / D
        def m(mask): return ev.evaluate_mask(mask)[1].f1_macro
        tmask = np.zeros(D, bool); tmask[inf] = True
        with contextlib.redirect_stdout(io.StringIO()):
            fm = fmlfs(ev.clients, r)
        row = {"interaction_frac": FINT, "seed": seed, "D": D,
               "true_subset": round(m(tmask), 4),
               "faport": round(search(name, "faport", seed), 4),
               "base": round(search(name, "base", seed), 4),
               "fed_rank": round(m(ranking_to_mask(fed_rank_relevance(ev.clients), r)), 4),
               "fmlfs": round(m(fm), 4), "all_features": round(m(np.ones(D, bool)), 4)}
        w.writerow(row); f.flush()
        print(f"[frac={FINT} s{seed}] true={row['true_subset']} faport={row['faport']} "
              f"base={row['base']} fed-rank={row['fed_rank']} fmlfs={row['fmlfs']} all={row['all_features']}", flush=True)
    f.close()


if __name__ == "__main__":
    raise SystemExit(main())
