"""Validate that a real wrapper SEARCH (not just the true-subset ceiling) beats per-feature filters
on the interaction synthetic. Materializes one interaction federation, runs FedAware (faport: filter
seed + swap) and base NSGA-II, and compares to fed-rank/FMLFS/all-features/true-subset.

Usage: python scripts/validate_interaction_search.py [D] [interaction_frac] [seed]
"""
import sys, copy
import numpy as np
from pathlib import Path
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import yaml
from synth_scaling import materialize
from fedwrap.datasets import _load_npz_any
from fedwrap.ml_eval import EvalConfig
from fedwrap.federated import make_evaluator
from fedwrap.federated.baselines import fed_rank_relevance, ranking_to_mask, topk_frequency_scores, fmlfs
from fedwrap.experiment import run_experiment_from_config

objn = ["one_minus_macro_f1", "one_minus_micro_f1", "feature_ratio"]
D = int(sys.argv[1]) if len(sys.argv) > 1 else 300
FINT = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 0
L, K, NINF, SIG, NOISE = 12, 8, 16, 8.0, 0.1


def setk(d, dot, v):
    p = dot.split("."); c = d
    for k in p[:-1]:
        c = c.setdefault(k, {})
    c[p[-1]] = v


def valbest_macro(rundir):
    z = np.load(Path(rundir) / "population_masks.npz"); Dz = int(z["n_features"])
    M = np.unpackbits(z["masks_packed"], axis=1)[:, :Dz].astype(bool)
    vo, on = z["val_objs"], z["pareto_val_mask"].astype(bool); idx = np.flatnonzero(on)
    cap = M.sum(1) <= 0.25 * Dz + 1; cand = idx[cap[idx]] if cap[idx].any() else idx
    return 1.0 - float(z["test_objs"][cand[np.argmin(vo[cand, 0])], 0])


def search(name, method):
    cfg = yaml.safe_load(open("configs/main_bench.yaml")); setk(cfg, "seed", SEED)
    setk(cfg, "dataset.kind", "prefold"); setk(cfg, "dataset.root", "data/synth_int"); setk(cfg, "dataset.name", name)
    setk(cfg, "cross_validation.enabled", True); setk(cfg, "cross_validation.n_folds", 1)
    setk(cfg, "model.kind", "mlknn"); setk(cfg, "model.mlknn_backend", "sklearn"); setk(cfg, "model.mlknn_device", "cpu")
    setk(cfg, "model.k", 5); setk(cfg, "model.cv_folds", 1); setk(cfg, "evolution.genotype", "bitstring")
    cfg.get("evolution", {}).pop("max_evals_per_feature", None)
    setk(cfg, "evolution.max_evals", 12000)   # short budget for a quick go/no-go
    setk(cfg, "evolution.early_stopping", {"enabled": True, "mode": "window", "window": 10, "rel_tol": 0.002, "patience": 3})
    fp = method == "faport"
    setk(cfg, "fedaware.enabled", fp); setk(cfg, "fedaware.relevance_warmstart", fp)
    setk(cfg, "fedaware.warmstart_frac", 0.3); setk(cfg, "fedaware.filter_seed", fp); setk(cfg, "fedaware.swap_prob", 0.4)
    setk(cfg, "federated", {"enabled": True, "n_clients": K, "partition": "natural_silo",
         "min_samples_per_client": 16, "client_fraction_full": 1.0, "final_eval_all_clients": True, "n_jobs": 6})
    setk(cfg, "reporting.max_feature_ratio", 0.25); setk(cfg, "logging.out_dir", f"runs/synth_int/{name}_{method}")
    run_experiment_from_config(cfg, fold_idx=0)
    return valbest_macro(f"runs/synth_int/{name}_{method}_fold0")


def main():
    name = f"int_D{D}_f{FINT}_s{SEED}"
    materialize(f"data/synth_int/{name}", N=4000, D=D, L=L, K=K, informative_ratio=0.1, noise=NOISE,
                alpha=1.0, seed=SEED, n_informative=NINF, interaction_frac=FINT, signal_strength=SIG)
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
    true_mask = np.zeros(D, bool); true_mask[inf] = True
    fr = ranking_to_mask(fed_rank_relevance(ev.clients), r)
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        fm = fmlfs(ev.clients, r)
    print(f"\n=== interaction synthetic D={D} frac={FINT} (true n_inf={len(inf)}) ===", flush=True)
    print(f"  true-subset (ceiling) : {m(true_mask):.4f}", flush=True)
    print(f"  fed-rank filter       : {m(fr):.4f}", flush=True)
    print(f"  FMLFS filter          : {m(fm):.4f}", flush=True)
    print(f"  all-features          : {m(np.ones(D, bool)):.4f}", flush=True)
    print(f"  base NSGA-II (search) : {search(name, 'base'):.4f}", flush=True)
    print(f"  FedAware faport (search): {search(name, 'faport'):.4f}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
