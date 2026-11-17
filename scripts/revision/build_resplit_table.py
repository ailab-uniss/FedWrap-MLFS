"""Revision analysis (Reviewer 2, comment 3; Reviewer 1, comment 1): main comparison over independent
train/test resplits of the natural federations.

For each dataset and split r (data/fed_real/{DS}__r{r}, search seed r):
  * FedWrap-MLFS (faport) and base NSGA-II (bitstring): validation-selected mask, re-evaluated on the test
    partition of that split with the federated test evaluator (train = non-test records, counts summed);
  * fed-rank and top-frequency at FedWrap's feature count; local-top-k union with that per-client budget;
    FMLFS at FedWrap's feature ratio; all features;
  * fed-rank and top-frequency at the base wrapper's feature count (for the wrapper-vs-filter effect);
  * evaluations to convergence, generations and hypervolume of both searches.
Per-split results are cached in reports/revision/resplit_cache. The summary reports mean and standard
deviation over splits, paired two-sided Wilcoxon signed-rank tests, and Holm-adjusted p-values within
each dataset's family of comparisons against FedWrap-MLFS.

Usage: python scripts/revision/build_resplit_table.py [n_splits=10]
"""
import os, sys, json, io, contextlib, csv, time
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
from scipy.stats import wilcoxon
from fedwrap.datasets import _load_npz_any
from fedwrap.ml_eval import EvalConfig
from fedwrap.federated import make_evaluator
from fedwrap.federated.baselines import (fed_rank_relevance, ranking_to_mask, local_topk_union,
                                         topk_frequency_scores, fmlfs)

ONE = sys.argv[1] == "--one" if len(sys.argv) > 1 else False     # --one <dataset> <split>: cache one split and exit
R = int(sys.argv[1]) if (len(sys.argv) > 1 and not ONE) else 10
DATASETS = [("ECG_cinc2021", 8), ("eICU_expl_k12", 12), ("ExtraSensory", 16)]
RUNS = Path("runs/revision/resplit")
CACHE = Path("reports/revision/resplit_cache"); CACHE.mkdir(parents=True, exist_ok=True)
objn = ["one_minus_macro_f1", "one_minus_micro_f1", "feature_ratio"]


def valbest(z):
    D = int(z["n_features"]); M = np.unpackbits(z["masks_packed"], axis=1)[:, :D].astype(bool)
    vo, on = z["val_objs"], z["pareto_val_mask"].astype(bool); idx = np.flatnonzero(on)
    cap = M.sum(1) <= 0.25 * D + 1; cand = idx[cap[idx]] if cap[idx].any() else idx
    return M[cand[np.argmin(vo[cand, 0])]]


def one_split(ds, nc, r):
    out_p = CACHE / f"{ds}_r{r}.json"
    if out_p.exists():
        return json.loads(out_p.read_text())
    if os.environ.get("CACHED_ONLY"):          # summary of the rows cached so far, no computation
        return None
    runs = {g: RUNS / f"{ds}__r{r}_{g}_s{r}_fold0" for g in ("faport", "bitstring")}
    if not all((p / "population_masks.npz").exists() and (p / "summary.json").exists() for p in runs.values()):
        return None
    fold = Path(f"data/fed_real/{ds}__r{r}/fold0")
    xtr, ytr = _load_npz_any(fold / "trainval.npz"); xte, yte = _load_npz_any(fold / "test.npz")
    gtr = np.load(fold / "trainval_groups.npy", allow_pickle=True)
    gte = np.load(fold / "test_groups.npy", allow_pickle=True)
    D = xtr.shape[1]
    ec = EvalConfig(kind="mlknn", primary_objective=objn[0], objective_names=objn,
                    random_state=0, k=5, s=1.0, mlknn_backend="sklearn", mlknn_device="cpu")
    cfg = {"federated": {"enabled": True, "n_clients": nc, "partition": "natural_silo",
           "min_samples_per_client": 32, "client_fraction_full": 1.0, "final_eval_all_clients": True},
           "objectives": {"names": objn}}
    ev = make_evaluator(xtr, ytr, xte, yte, ec, cfg, 0, groups=(gtr, gte))

    def score(mask):
        res = ev.evaluate_mask(np.asarray(mask, bool))[1]
        return {"macro": float(res.f1_macro), "micro": float(res.f1_micro), "nfeat": int(np.sum(mask))}

    rec = {"dataset": ds, "split": r, "D": int(D)}
    masks = {}
    for g, p in runs.items():
        m = valbest(np.load(p / "population_masks.npz")); masks[g] = m
        rec[g] = score(m)
        sm = json.loads((p / "summary.json").read_text())
        rec[g].update(total_evals=int(sm["final"]["total_evals"]), gens=int(sm["final"]["gen"]),
                      unique_evals=int(sm["federated"]["counters"]["full_evals"]),
                      hv_val=float(sm["final"]["hv_val"]), hv_test=float(sm["final"]["hv_test"]))
    with contextlib.redirect_stdout(io.StringIO()):
        fr = fed_rank_relevance(ev.clients)
        for tag, g in (("", "faport"), ("@base", "bitstring")):
            ratio = float(masks[g].sum() / D)
            rec["fed-rank" + tag] = score(ranking_to_mask(fr, ratio))
            rec["top-frequency" + tag] = score(ranking_to_mask(topk_frequency_scores(ev.clients, ratio), ratio))
        ratio = float(masks["faport"].sum() / D)
        rec["local-top-k union"] = score(local_topk_union(ev.clients, ratio))
        t0 = time.perf_counter()
        rec["FMLFS"] = score(fmlfs(ev.clients, ratio)); rec["fmlfs_seconds"] = time.perf_counter() - t0
        rec["all features"] = score(np.ones(D, dtype=bool))
    out_p.write_text(json.dumps(rec, indent=1))
    return rec


def p_wilcoxon(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(wilcoxon(a, b).pvalue) if np.any(a != b) else 1.0


def holm(pvals):
    order = np.argsort(pvals); m = len(pvals); adj = np.zeros(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * pvals[i]); adj[i] = min(1.0, run)
    return adj


def main():
    if ONE:
        ds, r = sys.argv[2], int(sys.argv[3]); nc = dict(DATASETS)[ds]
        print(ds, r, "cached" if one_split(ds, nc, r) is not None else "runs not ready", flush=True)
        return 0
    rows, deco = [], []
    for ds, nc in DATASETS:
        recs = [x for x in (one_split(ds, nc, r) for r in range(1, R + 1)) if x is not None]
        if not recs:
            continue
        n = len(recs); print(f"\n===== {ds}: {n} splits =====", flush=True)
        fw = np.array([x["faport"]["macro"] for x in recs])
        methods = [("FedWrap-MLFS", "faport"), ("base NSGA-II", "bitstring"), ("local-top-k union", "local-top-k union"),
                   ("fed-rank", "fed-rank"), ("top-frequency", "top-frequency"), ("FMLFS", "FMLFS"),
                   ("all features", "all features")]
        ps = [p_wilcoxon(fw, [x[k]["macro"] for x in recs]) for _, k in methods[1:]]
        padj = holm(np.array(ps))
        for i, (lab, k) in enumerate(methods):
            ma = np.array([x[k]["macro"] for x in recs]); mi = np.array([x[k]["micro"] for x in recs])
            nf = np.array([x[k]["nfeat"] for x in recs]); D = recs[0]["D"]
            row = dict(dataset=ds, method=lab, n=n, macro=ma.mean(), macro_std=ma.std(), micro=mi.mean(),
                       nfeat=nf.mean(), ratio=nf.mean() / D,
                       diff=(fw - ma).mean() if i else 0.0, p=ps[i - 1] if i else np.nan,
                       p_holm=padj[i - 1] if i else np.nan, wins=int(np.sum(fw > ma)) if i else 0)
            rows.append(row)
            print(f"  {lab:20s} macro={row['macro']:.4f}±{row['macro_std']:.4f} micro={row['micro']:.4f} "
                  f"#feat={row['nfeat']:.0f} ratio={row['ratio']:.3f}"
                  + (f"  FedWrap-diff={row['diff']:+.4f} p={row['p']:.4f} holm={row['p_holm']:.4f} wins={row['wins']}/{n}" if i else ""))
        base = np.array([x["bitstring"]["macro"] for x in recs])
        cand = {k: np.array([x[k + "@base"]["macro"] for x in recs]) for k in ("fed-rank", "top-frequency")}
        sf = max(cand, key=lambda k: cand[k].mean()); fb = cand[sf]
        d = dict(dataset=ds, n=n, filter_at_base=sf, filter_at_base_macro=fb.mean(), base=base.mean(), fedwrap=fw.mean(),
                 wrapper_effect=(base - fb).mean(), wrapper_p=p_wilcoxon(base, fb), wrapper_wins=int(np.sum(base > fb)),
                 fedaware_effect=(fw - base).mean(), fedaware_p=p_wilcoxon(fw, base), fedaware_wins=int(np.sum(fw > base)))
        for key in ("total_evals", "unique_evals", "gens", "hv_val", "hv_test"):
            a = np.array([x["faport"][key] for x in recs], float); b = np.array([x["bitstring"][key] for x in recs], float)
            d[f"{key}_fedaware"] = a.mean(); d[f"{key}_base"] = b.mean(); d[f"{key}_p"] = p_wilcoxon(a, b)
            d[f"{key}_fedaware_lower"] = int(np.sum(a < b))
        deco.append(d)
        print("  decomposition:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()})
        print(f"  FMLFS build time (mean s): {np.mean([x['fmlfs_seconds'] for x in recs]):.1f}")
    for name, data in (("resplit_table.csv", rows), ("resplit_decomposition.csv", deco)):
        if data:
            with open(Path("reports/revision") / name, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(data[0].keys())); w.writeheader(); w.writerows(data)
    print("\nwrote reports/revision/resplit_table.csv and resplit_decomposition.csv")


if __name__ == "__main__":
    raise SystemExit(main())
