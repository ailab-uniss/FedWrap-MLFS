"""Controlled non-IID stress test: run the federated FedWrap-MLFS (faport) search under a Dirichlet
label-skew sweep (alpha in {1.0, 0.3, 0.1}) on a credible dataset, K=10 clients. Tests how
the method behaves as heterogeneity worsens. Uses the fixed pipeline (run_experiment_from_config,
FedAware-NSGA-II, exact TP/FP/FN aggregation)."""
import sys, copy, yaml, glob
import numpy as np
sys.path.insert(0, ".")
from fedwrap.experiment import run_experiment_from_config

BASE = yaml.safe_load(open("configs/main_bench.yaml"))
def setk(d, dot, v):
    p = dot.split("."); c = d
    for k in p[:-1]: c = c.setdefault(k, {})
    c[p[-1]] = v

DS = sys.argv[1] if len(sys.argv) > 1 else "eICU_expl_k12"
ALPHAS = [float(a) for a in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["1.0", "0.3", "0.1"])]
SEEDS = [int(s) for s in (sys.argv[3].split(",") if len(sys.argv) > 3 else ["0", "1"])]

for alpha in ALPHAS:
    for seed in SEEDS:
        cfg = copy.deepcopy(BASE); setk(cfg, "seed", seed)
        setk(cfg, "dataset.kind", "prefold"); setk(cfg, "dataset.root", "data/fed_real"); setk(cfg, "dataset.name", DS)
        setk(cfg, "cross_validation.enabled", True); setk(cfg, "cross_validation.n_folds", 1)
        setk(cfg, "model.mlknn_backend", "sklearn"); setk(cfg, "model.mlknn_device", "cpu"); setk(cfg, "model.k", 5)
        setk(cfg, "model.cv_folds", 1)
        setk(cfg, "evolution.genotype", "bitstring")
        setk(cfg, "fedaware.enabled", True); setk(cfg, "fedaware.relevance_warmstart", True)
        setk(cfg, "fedaware.warmstart_frac", 0.3); setk(cfg, "fedaware.filter_seed", True)
        setk(cfg, "fedaware.swap_prob", 0.4)
        cfg.get("evolution", {}).pop("max_evals_per_feature", None)
        setk(cfg, "evolution.max_evals_per_feature", None); setk(cfg, "evolution.max_evals", 300000)
        setk(cfg, "evolution.early_stopping", {"enabled": True, "mode": "window", "window": 10, "rel_tol": 0.002, "patience": 3})
        # controlled Dirichlet label-skew partition (NOT natural silo)
        setk(cfg, "federated", {"enabled": True, "n_clients": 10, "partition": "label_skew_dirichlet",
                                "dirichlet_alpha": alpha, "min_samples_per_client": 8, "client_fraction_full": 1.0,
                                "final_eval_all_clients": True, "return_client_metrics": True, "n_jobs": 8})
        setk(cfg, "reporting.max_feature_ratio", 0.25)
        setk(cfg, "logging.out_dir", f"runs/skew/{DS}_a{alpha}_s{seed}")
        try:
            run_experiment_from_config(cfg, fold_idx=0); print(f"DONE {DS} a={alpha} s{seed}", flush=True)
        except Exception as e:
            print(f"ERR {DS} a={alpha} s{seed}: {repr(e)[:160]}", flush=True)
