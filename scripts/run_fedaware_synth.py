"""Headline FedAware evidence on the synthetic federated benchmark: does the federation-aware
search beat a federation-naive NSGA-II, and does its advantage grow with heterogeneity?

For each sweep point and seed we run TWO selectors at full evaluation budget:
  base    : plain federated NSGA-II wrapper (federation-naive search)
  fedaware: FedAware-NSGA-II (stability tie-break + disagreement-
            guided mutation on) -- the headline configuration
and record global macro-F1 and worst-client macro-F1 for each, plus fed-rank / all-features
references at matched sparsity. Sweeps that stress the federation (K, non-IID alpha) are where the
federation-aware operators are expected to help.

Usage: python scripts/run_fedaware_synth.py <sweep in {alpha,K,D,L,ninf,noise}> <seeds csv>
"""
import sys, copy, csv, json
import numpy as np
from pathlib import Path
sys.path.insert(0, ".")
import yaml
from fedwrap.datasets import _load_npz_any
from fedwrap.ml_eval import EvalConfig
from fedwrap.federated import make_evaluator
from fedwrap.federated.baselines import build_baseline_masks
from fedwrap.experiment import run_experiment_from_config
sys.path.insert(0, str(Path(__file__).resolve().parent))
from synth_scaling import materialize
from run_synth_scaling import DEFAULT, GRIDS, setk, per_client_and_global, best_mask

BASE = yaml.safe_load(open("configs/main_bench.yaml"))
objn = ["one_minus_macro_f1", "one_minus_micro_f1", "feature_ratio"]


def run_selector(name, nclients, seed, fedaware):
    cfg = copy.deepcopy(BASE); setk(cfg, "seed", seed)
    setk(cfg, "dataset.kind", "prefold"); setk(cfg, "dataset.root", "data/synth_scaling"); setk(cfg, "dataset.name", name)
    setk(cfg, "cross_validation.enabled", True); setk(cfg, "cross_validation.n_folds", 1)
    setk(cfg, "model.kind", "mlknn"); setk(cfg, "model.mlknn_backend", "sklearn"); setk(cfg, "model.mlknn_device", "cpu")
    setk(cfg, "model.k", 5); setk(cfg, "model.cv_folds", 1)
    setk(cfg, "evolution.genotype", "bitstring")
    cfg.get("evolution", {}).pop("max_evals_per_feature", None)
    setk(cfg, "evolution.max_evals", 300000)
    setk(cfg, "evolution.early_stopping", {"enabled": True, "mode": "window", "window": 10, "rel_tol": 0.002, "patience": 3})
    setk(cfg, "fedaware.enabled", bool(fedaware))
    setk(cfg, "fedaware.stability_tiebreak", True); setk(cfg, "fedaware.disagreement_mutation", True)
    setk(cfg, "federated", {"enabled": True, "n_clients": nclients, "partition": "natural_silo",
          "min_samples_per_client": 16, "client_fraction_full": 1.0, "final_eval_all_clients": True,
          "return_client_metrics": True, "n_jobs": 8})
    setk(cfg, "reporting.max_feature_ratio", 0.25)
    out = f"runs/fedaware_synth/{name}_{'fa' if fedaware else 'base'}_s{seed}"
    setk(cfg, "logging.out_dir", out)
    run_experiment_from_config(cfg, fold_idx=0)
    return out + "_fold0"


def main():
    sweep = sys.argv[1] if len(sys.argv) > 1 else "alpha"
    seeds = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0]
    Path("data/synth_scaling").mkdir(parents=True, exist_ok=True)
    Path("runs/fedaware_synth").mkdir(parents=True, exist_ok=True)
    rows = []
    for (param, val) in GRIDS[sweep]:
        for seed in seeds:
            p = dict(DEFAULT); p[param] = val
            name = f"fas_{sweep}_{val}_s{seed}"
            dpath = f"data/synth_scaling/{name}"
            _, _, K = materialize(dpath, N=p["N"], D=p["D"], L=p["L"], K=p["K"],
                informative_ratio=p["informative_ratio"], noise=p["noise"], alpha=p["alpha"],
                n_informative=p["n_informative"], seed=seed)
            D, L = p["D"], p["L"]
            xtr, ytr = _load_npz_any(f"{dpath}/fold0/trainval.npz")
            xte, yte = _load_npz_any(f"{dpath}/fold0/test.npz")
            gtr = np.load(f"{dpath}/fold0/trainval_groups.npy", allow_pickle=True)
            gte = np.load(f"{dpath}/fold0/test_groups.npy", allow_pickle=True)
            ec = EvalConfig(kind="mlknn", primary_objective=objn[0], objective_names=objn,
                            random_state=0, k=5, s=1.0, mlknn_backend="sklearn", mlknn_device="cpu")
            fc = {"federated": {"enabled": True, "n_clients": K, "partition": "natural_silo",
                  "min_samples_per_client": 16, "client_fraction_full": 1.0, "final_eval_all_clients": True},
                  "objectives": {"names": objn}}
            ev = make_evaluator(xtr, ytr, xte, yte, ec, fc, 0, groups=(gtr, gte))
            rec = {"sweep": sweep, param: val, "seed": seed, "D": D, "L": L, "K": K}
            ratio_for_baseline = None
            for tag, fa in [("base", False), ("fedaware", True)]:
                try:
                    rd = run_selector(name, K, seed, fa)
                    mask, _, _, ratio, _ = best_mask(rd)
                except Exception as e:
                    print(f"ERR {name} {tag}: {repr(e)[:160]}", flush=True); continue
                gmacro, mean_c, worst_c = per_client_and_global(ev, mask)
                rec[f"{tag}_macro"] = round(gmacro, 4)
                rec[f"{tag}_worst"] = round(worst_c, 4)
                rec[f"{tag}_ratio"] = round(ratio, 4)
                if tag == "fedaware":
                    ratio_for_baseline = ratio
            # references at fedaware's sparsity
            import io, contextlib
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    bm = build_baseline_masks(ev.clients, D, [ratio_for_baseline], seed=0)
                fr = bm.get("fed_rank_relevance", {}).get(ratio_for_baseline)
                rec["fedrank_macro"] = round(ev.evaluate_mask(fr)[1].f1_macro, 4) if fr is not None else None
                rec["all_macro"] = round(ev.evaluate_mask(np.ones(D, bool))[1].f1_macro, 4)
            except Exception:
                pass
            rows.append(rec)
            print(f"[{sweep}={val} s{seed}] K={K} base={rec.get('base_macro')}/{rec.get('base_worst')} "
                  f"fedaware={rec.get('fedaware_macro')}/{rec.get('fedaware_worst')} (macro/worst)", flush=True)
            import shutil; shutil.rmtree(dpath, ignore_errors=True)
    outcsv = f"reports/fedaware_synth_{sweep}.csv"
    if rows:
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("sweep",), k))
        with open(outcsv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) + [k for k in keys if k not in rows[0]])
            w.writeheader(); w.writerows(rows)
        print(f"wrote {outcsv}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
