"""Run one selector variant on a real natural-silo federation, writing val-selected Pareto
populations to runs/plain_fed/{ds}_{method}_s{seed} for the 10-seed main table.

  method = fedaware : FedAware-NSGA-II (relevance sketch + hard-label mutation + client stability)
  method = base     : plain NSGA-II wrapper over flat masks, exact global objective
  method = localavg : same plain NSGA-II but the search optimizes the (biased) mean of per-client F1

All three optimize binary masks under the same evaluator/budget/stopping rule; the difference is the
search (fedaware) or the objective aggregation (localavg).

Usage: python scripts/run_reals_method.py <dataset> <method> <seeds csv>
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
METHOD = sys.argv[2] if len(sys.argv) > 2 else "fedaware"
SEEDS = [int(s) for s in sys.argv[3].split(",")] if len(sys.argv) > 3 else list(range(10))
nclients = len(set(np.load(f"data/fed_real/{DS}/fold0/trainval_groups.npy", allow_pickle=True).tolist()))

for seed in SEEDS:
    out = f"runs/plain_fed/{DS}_{METHOD}_s{seed}"
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
    is_warm = METHOD.startswith("fawarm")
    is_port = METHOD == "faport"   # portfolio: relevance warmstart + filter-seeded init + swap operator
    setk(cfg, "fedaware.enabled", METHOD == "fedaware" or is_warm or is_port)
    setk(cfg, "fedaware.stability_tiebreak", True); setk(cfg, "fedaware.disagreement_mutation", True)
    # 'fedaware' = original operators only; 'fawarm[NN]' adds relevance warmstart (frac NN/100);
    # 'faport' = warmstart(0.3) + filter-seeded initialization + sparsity-preserving swap operator.
    setk(cfg, "fedaware.relevance_warmstart", is_warm or is_port)
    if is_warm and METHOD[len("fawarm"):].isdigit():
        setk(cfg, "fedaware.warmstart_frac", int(METHOD[len("fawarm"):]) / 100.0)
    if is_port:
        setk(cfg, "fedaware.warmstart_frac", 0.3)
        setk(cfg, "fedaware.filter_seed", True)
        setk(cfg, "fedaware.swap_prob", 0.4)      # 0.4 swap / 0.4 guided / 0.2 bit-flip
    fed = {"enabled": True, "n_clients": nclients, "partition": "natural_silo",
           "min_samples_per_client": 32, "client_fraction_full": 1.0, "final_eval_all_clients": True,
           "return_client_metrics": True, "estimate_communication": True, "n_jobs": 8}
    if METHOD == "localavg":
        fed["objective_aggregation"] = "local_avg"
    setk(cfg, "federated", fed)
    setk(cfg, "reporting.max_feature_ratio", 0.25)
    setk(cfg, "logging.out_dir", out)
    try:
        run_experiment_from_config(cfg, fold_idx=0); print(f"DONE {METHOD} {DS} s{seed}", flush=True)
    except Exception as e:
        print(f"ERR {METHOD} {DS} s{seed}: {repr(e)[:200]}", flush=True)
