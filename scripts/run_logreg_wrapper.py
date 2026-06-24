"""Genuine BR-LogReg wrapper: run the FedWrap-MLFS (faport) search with binary-relevance logistic
regression as the INNER evaluator (model.kind=logreg), so features are selected *for* the classifier
that will be deployed. This is the methodologically correct classifier-agnostic test: a wrapper selects
features for its downstream model; you then deploy that same model. We compare FedWrap-LogReg against
the filters under LogReg (matched ratio) -- the analogue of the ML-kNN comparison, with the classifier
matched to the wrapper.

Usage: python scripts/run_logreg_wrapper.py <dataset> <seeds csv>   (default: all datasets, seeds 0,1,2)
"""
import sys, copy, yaml
from pathlib import Path
sys.path.insert(0, ".")
from fedwrap.experiment import run_experiment_from_config

BASE = yaml.safe_load(open("configs/main_bench.yaml"))
DATASETS = {"ECG_cinc2021": 8, "eICU_expl_k12": 12, "ExtraSensory": 16}


def setk(d, dot, v):
    p = dot.split("."); c = d
    for k in p[:-1]:
        c = c.setdefault(k, {})
    c[p[-1]] = v


def main():
    only = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in DATASETS else None
    seeds = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0, 1, 2]
    items = [(only, DATASETS[only])] if only else list(DATASETS.items())
    for ds, nc in items:
        for seed in seeds:
            out = f"runs/plain_fed_logreg/{ds}_faport_s{seed}"
            if Path(out + "_fold0", "population_masks.npz").exists():
                print(f"skip {out}", flush=True); continue
            cfg = copy.deepcopy(BASE); setk(cfg, "seed", seed)
            setk(cfg, "dataset.kind", "prefold"); setk(cfg, "dataset.root", "data/fed_real"); setk(cfg, "dataset.name", ds)
            setk(cfg, "cross_validation.enabled", True); setk(cfg, "cross_validation.n_folds", 1)
            setk(cfg, "model.kind", "logreg"); setk(cfg, "model.cv_folds", 1)   # BR-LogReg inner evaluator
            setk(cfg, "evolution.genotype", "bitstring")
            cfg.get("evolution", {}).pop("max_evals_per_feature", None)
            setk(cfg, "evolution.max_evals", 300000)
            setk(cfg, "evolution.early_stopping", {"enabled": True, "mode": "window",
                  "window": 10, "rel_tol": 0.002, "patience": 3})
            # faport: relevance warmstart + filter-seeded init + sparsity-preserving swap
            setk(cfg, "fedaware.enabled", True); setk(cfg, "fedaware.relevance_warmstart", True)
            setk(cfg, "fedaware.warmstart_frac", 0.3); setk(cfg, "fedaware.filter_seed", True)
            setk(cfg, "fedaware.swap_prob", 0.4)
            setk(cfg, "federated", {"enabled": True, "n_clients": nc, "partition": "natural_silo",
                  "min_samples_per_client": 32, "client_fraction_full": 1.0,
                  "final_eval_all_clients": True, "return_client_metrics": True, "n_jobs": 8})
            setk(cfg, "reporting.max_feature_ratio", 0.25); setk(cfg, "logging.out_dir", out)
            try:
                run_experiment_from_config(cfg, fold_idx=0)
                print(f"DONE logreg-wrapper {ds} s{seed}", flush=True)
            except Exception as e:
                print(f"ERR {ds} s{seed}: {repr(e)[:200]}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
