"""Controlled federated scaling benchmark runner (graph-free FedWrap-MLFS = flat-bitstring wrapper).

For each sweep point and seed: generate a synthetic federated multi-label dataset, run the flat
federated wrapper search, and record how the protocol behaves as D, L, K, non-IID severity, and
signal sparsity change. Metrics per point:
  - FedWrap macro/micro-F1 (val-selected at matched sparsity)
  - fed-rank / all-features / random baselines at matched sparsity
  - FMLFS feasibility/time (O(D^2); flags when it would exhaust memory)
  - local-F1 averaging bias = global macro - mean-of-local macro for the SAME mask
  - worst-client macro
  - search wall-time, #full evaluations, uplink communication (3L counters)

Usage: python scripts/run_synth_scaling.py <sweep> <seeds csv>
  sweep in {D, L, K, alpha, noise}
"""
import sys, os, copy, time, csv, shutil, json
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

BASE = yaml.safe_load(open("configs/main_bench.yaml"))
objn = ["one_minus_macro_f1", "one_minus_micro_f1", "feature_ratio"]

# default (fixed) parameters and per-sweep grids
DEFAULT = dict(N=4000, D=300, L=20, K=8, informative_ratio=0.10, noise=0.3, alpha=0.5, n_informative=30)
GRIDS = {
    "D":     [("D", v) for v in [100, 500, 1000, 2000, 5000, 10000]],
    "L":     [("L", v) for v in [5, 10, 20, 50, 100]],
    "K":     [("K", v) for v in [2, 4, 8, 16, 32]],
    "alpha": [("alpha", v) for v in [10.0, 1.0, 0.3, 0.1]],
    "noise": [("noise", v) for v in [0.0, 0.1, 0.2, 0.4]],
    "ninf":  [("n_informative", v) for v in [10, 30, 100, 200]],
}


def setk(d, dot, v):
    p = dot.split("."); c = d
    for k in p[:-1]:
        c = c.setdefault(k, {})
    c[p[-1]] = v


def per_client_and_global(ev, mask):
    """global macro and mean/worst per-client local-support macro for a mask."""
    TP = FP = FN = None; per = []
    for c in ev.clients:
        r = c.evaluate_mask(np.asarray(mask, bool), mode="full")
        tp, fp, fn = r["tp"].astype(float), r["fp"].astype(float), r["fn"].astype(float)
        den = 2 * tp + fp + fn
        f1 = np.divide(2 * tp, den, out=np.zeros_like(den), where=den > 0)
        sup = (tp + fn) > 0
        per.append(float(f1[sup].mean()) if sup.any() else 0.0)
        TP = tp.copy() if TP is None else TP + tp
        FP = fp.copy() if FP is None else FP + fp
        FN = fn.copy() if FN is None else FN + fn
    den = 2 * TP + FP + FN
    gmacro = float(np.divide(2 * TP, den, out=np.zeros_like(den), where=den > 0).mean())
    return gmacro, float(np.mean(per)), float(np.min(per))


def run_fedwrap(name, nclients, seed, method="fedaware"):
    cfg = copy.deepcopy(BASE); setk(cfg, "seed", seed)
    setk(cfg, "dataset.kind", "prefold"); setk(cfg, "dataset.root", "data/synth_scaling"); setk(cfg, "dataset.name", name)
    setk(cfg, "cross_validation.enabled", True); setk(cfg, "cross_validation.n_folds", 1)
    setk(cfg, "model.kind", "mlknn"); setk(cfg, "model.mlknn_backend", "sklearn"); setk(cfg, "model.mlknn_device", "cpu")
    setk(cfg, "model.k", 5); setk(cfg, "model.cv_folds", 1)
    setk(cfg, "evolution.genotype", "bitstring")
    cfg.get("evolution", {}).pop("max_evals_per_feature", None)
    setk(cfg, "evolution.max_evals", 300000)
    setk(cfg, "evolution.early_stopping", {"enabled": True, "mode": "window", "window": 10, "rel_tol": 0.002, "patience": 3})
    # FedAware = relevance warmstart (frac 0.3) + relevance-guided mutation + client-stability tie-break.
    setk(cfg, "fedaware.enabled", method == "fedaware")
    setk(cfg, "fedaware.stability_tiebreak", True); setk(cfg, "fedaware.disagreement_mutation", True)
    setk(cfg, "fedaware.relevance_warmstart", method == "fedaware"); setk(cfg, "fedaware.warmstart_frac", 0.3)
    setk(cfg, "bites.enabled", False)
    setk(cfg, "federated", {"enabled": True, "n_clients": nclients, "partition": "natural_silo",
          "min_samples_per_client": 16, "client_fraction_full": 1.0, "final_eval_all_clients": True,
          "return_client_metrics": True, "estimate_communication": True, "n_jobs": 8})
    setk(cfg, "reporting.max_feature_ratio", 0.25)
    out = f"runs/synth_scaling/{name}_{method}_s{seed}"
    setk(cfg, "logging.out_dir", out)
    if Path(out + "_fold0", "population_masks.npz").exists():
        return out + "_fold0", 0.0  # reuse a search that survived a previous (interrupted) run
    setk(cfg, "logging.out_dir", out)
    t0 = time.perf_counter()
    run_experiment_from_config(cfg, fold_idx=0)
    walltime = time.perf_counter() - t0
    return out + "_fold0", walltime


def best_mask(run_dir):
    z = np.load(Path(run_dir) / "population_masks.npz"); D2 = int(z["n_features"])
    M = np.unpackbits(z["masks_packed"], axis=1)[:, :D2].astype(bool)
    v, t, on = z["val_objs"], z["test_objs"], z["pareto_val_mask"].astype(bool)
    idx = np.flatnonzero(on); cap = M.sum(1) <= 0.25 * D2 + 1
    cand = idx[cap[idx]] if cap[idx].any() else idx
    bi = cand[np.argmin(v[cand, 0])]
    n_evals = int(z["full_evals"]) if "full_evals" in z.files else None
    # test_objs store (1 - F1); convert back to macro/micro F1
    return M[bi], float(1 - t[bi, 0]), float(1 - t[bi, 1]), float(M[bi].sum() / D2), n_evals


def main():
    sweep = sys.argv[1] if len(sys.argv) > 1 else "D"
    seeds = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0]
    method = sys.argv[3] if len(sys.argv) > 3 else "fedaware"
    Path("data/synth_scaling").mkdir(parents=True, exist_ok=True)
    Path("runs/synth_scaling").mkdir(parents=True, exist_ok=True)
    grid = GRIDS[sweep]
    if os.environ.get("SC_GRID"):  # e.g. SC_GRID="100,500,1000" to restrict the sweep values
        keep = {float(v) for v in os.environ["SC_GRID"].split(",")}
        grid = [(p, v) for (p, v) in grid if float(v) in keep]

    # ---- crash-safe / resumable CSV: write each row immediately, skip points already done ----
    outcsv = f"reports/synth_scaling_{sweep}_{method}.csv"
    fields = ["sweep", "method", "sweep_val", "seed", "D", "L", "K", "ratio", "fedwrap_macro",
              "fedwrap_micro", "test_objs_macro", "mean_client_macro", "worst_client_macro",
              "localavg_bias", "all_macro", "fedrank_macro", "fmlfs_macro", "random_macro",
              "walltime_s", "n_full_evals", "uplink_counters_per_round", "baseline_build_s"]
    done = set()
    if Path(outcsv).exists():
        for r in csv.DictReader(open(outcsv)):
            done.add((r["sweep_val"], r["seed"]))
    cf = open(outcsv, "a", newline="")
    w = csv.DictWriter(cf, fieldnames=fields)
    if cf.tell() == 0:
        w.writeheader(); cf.flush()

    for (param, val) in grid:
        for seed in seeds:
            if (str(val), str(seed)) in done:
                print(f"skip {sweep}={val} s{seed} (already in CSV)", flush=True); continue
            p = dict(DEFAULT); p[param] = val
            name = f"sc_{sweep}_{val}_s{seed}"
            dpath = f"data/synth_scaling/{name}"
            shape, _, K = materialize(dpath, N=p["N"], D=p["D"], L=p["L"], K=p["K"],
                informative_ratio=p["informative_ratio"], noise=p["noise"], alpha=p["alpha"],
                n_informative=p["n_informative"], seed=seed)
            D = p["D"]; L = p["L"]
            try:
                rundir, walltime = run_fedwrap(name, K, seed, method=method)
                mask, fw_macro, fw_micro, ratio, n_evals = best_mask(rundir)
            except Exception as e:
                print(f"ERR fedwrap {name}: {repr(e)[:200]}", flush=True)
                shutil.rmtree(dpath, ignore_errors=True); continue
            # independent federated evaluator for baselines + diagnostics
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
            gmacro, mean_client, worst_client = per_client_and_global(ev, mask)
            allm = np.ones(D, bool)
            all_macro = ev.evaluate_mask(allm)[1].f1_macro
            import io, contextlib
            t_fmlfs = None; fmlfs_macro = None; fed_macro = None; rnd_macro = None
            try:
                tf = time.perf_counter()
                with contextlib.redirect_stdout(io.StringIO()):
                    bm = build_baseline_masks(ev.clients, D, [ratio], seed=0, fmlfs_max_features=1000)
                t_fmlfs = time.perf_counter() - tf
                fr = bm.get("fed_rank_relevance", {}).get(ratio)
                fed_macro = ev.evaluate_mask(fr)[1].f1_macro if fr is not None else None
                fm = bm.get("fmlfs", {}).get(ratio)
                fmlfs_macro = ev.evaluate_mask(fm)[1].f1_macro if fm is not None else None
                rnd = bm.get("random_subset", {}).get(ratio)
                rnd_macro = ev.evaluate_mask(rnd)[1].f1_macro if rnd is not None else None
            except Exception as e:
                print(f"  baselines warn {name}: {repr(e)[:120]}", flush=True)
            uplink_per_round = 3 * L  # integer counters per client per round
            row = {"sweep": sweep, "method": method, "sweep_val": val, "seed": seed, "D": D, "L": L, "K": K,
                   "ratio": round(ratio, 4), "fedwrap_macro": round(gmacro, 4),  # independent re-eval (trustworthy)
                   "fedwrap_micro": round(fw_micro, 4), "test_objs_macro": round(fw_macro, 4),
                   "mean_client_macro": round(mean_client, 4), "worst_client_macro": round(worst_client, 4),
                   "localavg_bias": round(gmacro - mean_client, 4),
                   "all_macro": round(all_macro, 4),
                   "fedrank_macro": round(fed_macro, 4) if fed_macro is not None else None,
                   "fmlfs_macro": round(fmlfs_macro, 4) if fmlfs_macro is not None else None,
                   "random_macro": round(rnd_macro, 4) if rnd_macro is not None else None,
                   "walltime_s": round(walltime, 1), "n_full_evals": n_evals,
                   "uplink_counters_per_round": uplink_per_round, "baseline_build_s": round(t_fmlfs, 2) if t_fmlfs else None}
            w.writerow(row); cf.flush()  # persist immediately so a power loss cannot wipe progress
            print(f"[{sweep}={val} s{seed}] D={D} L={L} K={K} | FedWrap macro={gmacro:.3f} "
                  f"fed-rank={fed_macro} all={all_macro:.3f} fmlfs={fmlfs_macro} | bias={gmacro-mean_client:+.3f} "
                  f"worst={worst_client:.3f} | t={walltime:.0f}s", flush=True)
            shutil.rmtree(dpath, ignore_errors=True)  # cleanup heavy data, keep masks + the CSV row
    cf.close()
    print(f"wrote {outcsv}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
