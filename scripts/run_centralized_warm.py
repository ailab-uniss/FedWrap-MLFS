"""Centralized-pooling counterpart of the warmstart selector, for the cost-of-locality table.

Runs the SAME FedAware warm-start wrapper but with the federation pooled into a single client
(n_clients=1, iid), so the relevance sketch and search operate on the pooled data and the objective
is the centralized global F1. The deployed mask's test macro-F1 is read from the saved population
(val-best at matched sparsity), exactly as for the federated runs, giving an apples-to-apples
federated-vs-centralized comparison with one selector.

Usage: python scripts/run_centralized_warm.py <dataset> <seeds csv>
"""
import sys, copy
import numpy as np
from pathlib import Path
sys.path.insert(0, ".")
import yaml
from fedwrap.experiment import run_experiment_from_config

BASE = yaml.safe_load(open("configs/main_bench.yaml"))


def setk(d, dot, v):
    p = dot.split("."); c = d
    for k in p[:-1]:
        c = c.setdefault(k, {})
    c[p[-1]] = v


DS = sys.argv[1] if len(sys.argv) > 1 else "eICU_expl_k12"
SEEDS = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0, 1, 2]

for seed in SEEDS:
    out = f"runs/centralized_warm/{DS}_s{seed}"
    if Path(out + "_fold0", "population_masks.npz").exists():
        print(f"skip {out}", flush=True); continue
    cfg = copy.deepcopy(BASE); setk(cfg, "seed", seed)
    setk(cfg, "dataset.kind", "prefold"); setk(cfg, "dataset.root", "data/fed_real"); setk(cfg, "dataset.name", DS)
    setk(cfg, "cross_validation.enabled", True); setk(cfg, "cross_validation.n_folds", 1)
    setk(cfg, "model.kind", "mlknn"); setk(cfg, "model.mlknn_backend", "sklearn"); setk(cfg, "model.mlknn_device", "cpu")
    setk(cfg, "model.k", 5); setk(cfg, "model.cv_folds", 1)
    setk(cfg, "evolution.genotype", "bitstring")
    cfg.get("evolution", {}).pop("max_evals_per_feature", None)
    setk(cfg, "evolution.max_evals", 300000)
    setk(cfg, "evolution.early_stopping", {"enabled": True, "mode": "window", "window": 10, "rel_tol": 0.002, "patience": 3})
    setk(cfg, "fedaware.enabled", True); setk(cfg, "fedaware.stability_tiebreak", True)
    setk(cfg, "fedaware.disagreement_mutation", True); setk(cfg, "fedaware.relevance_warmstart", True)
    setk(cfg, "fedaware.warmstart_frac", 0.3); setk(cfg, "bites.enabled", False)
    # pooled = federation with a single client (centralized objective, same selector machinery)
    setk(cfg, "federated", {"enabled": True, "n_clients": 1, "partition": "iid",
         "min_samples_per_client": 32, "client_fraction_full": 1.0, "final_eval_all_clients": True,
         "return_client_metrics": True, "estimate_communication": False, "n_jobs": 1})
    setk(cfg, "reporting.max_feature_ratio", 0.25); setk(cfg, "logging.out_dir", out)
    try:
        run_experiment_from_config(cfg, fold_idx=0); print(f"DONE centralized {DS} s{seed}", flush=True)
    except Exception as e:
        print(f"ERR centralized {DS} s{seed}: {repr(e)[:200]}", flush=True)
