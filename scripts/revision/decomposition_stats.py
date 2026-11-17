"""Revision analysis (Reviewer 1, comment 1): separate the two sources of gain on the natural federations.

  wrapper effect   = base NSGA-II wrapper  - strongest filter matched to the base wrapper's feature count
  FedAware effect  = FedWrap-MLFS          - base NSGA-II wrapper (same evaluator, budget rule, stopping rule)
  total            = FedWrap-MLFS          - strongest filter matched to FedWrap's feature count

Uses the saved ten-seed runs of the submitted manuscript (runs/plain_fed) and the same test evaluator as
scripts/build_maintable_warm.py. Paired two-sided Wilcoxon signed-rank tests over seeds.
Writes reports/revision/decomposition.csv.
"""
import sys, glob, csv, io, contextlib
import numpy as np
sys.path.insert(0, ".")
from scipy.stats import wilcoxon
from fedwrap.datasets import _load_npz_any
from fedwrap.ml_eval import EvalConfig
from fedwrap.federated import make_evaluator
from fedwrap.federated.baselines import build_baseline_masks

RUNS = sys.argv[1] if len(sys.argv) > 1 else "runs/plain_fed"
SUFFIX = sys.argv[2] if len(sys.argv) > 2 else ""
DATASETS = [("ECG_cinc2021", 8), ("eICU_expl_k12", 12), ("ExtraSensory", 16)]
objn = ["one_minus_macro_f1", "one_minus_micro_f1", "feature_ratio"]
FILTERS = {"fed_rank_relevance": "fed-rank", "topk_frequency": "top-frequency"}


def valbest(z):
    D = int(z["n_features"]); M = np.unpackbits(z["masks_packed"], axis=1)[:, :D].astype(bool)
    vo, on = z["val_objs"], z["pareto_val_mask"].astype(bool); idx = np.flatnonzero(on)
    cap = M.sum(1) <= 0.25 * D + 1; cand = idx[cap[idx]] if cap[idx].any() else idx
    return M[cand[np.argmin(vo[cand, 0])]]


def pval(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(wilcoxon(a, b).pvalue) if np.any(a != b) else 1.0


rows = []
for ds, nc in DATASETS:
    xtr, ytr = _load_npz_any(f"data/fed_real/{ds}/fold0/trainval.npz")
    xte, yte = _load_npz_any(f"data/fed_real/{ds}/fold0/test.npz")
    gtr = np.load(f"data/fed_real/{ds}/fold0/trainval_groups.npy", allow_pickle=True)
    gte = np.load(f"data/fed_real/{ds}/fold0/test_groups.npy", allow_pickle=True)
    D = xtr.shape[1]
    ec = EvalConfig(kind="mlknn", primary_objective=objn[0], objective_names=objn,
                    random_state=0, k=5, s=1.0, mlknn_backend="sklearn", mlknn_device="cpu")
    cfg = {"federated": {"enabled": True, "n_clients": nc, "partition": "natural_silo",
           "min_samples_per_client": 32, "client_fraction_full": 1.0, "final_eval_all_clients": True},
           "objectives": {"names": objn}}
    ev = make_evaluator(xtr, ytr, xte, yte, ec, cfg, 0, groups=(gtr, gte))
    res = {}
    for geno in ("faport", "bitstring"):
        ma, rt = [], []
        for d in sorted(glob.glob(f"{RUNS}/{ds}_{geno}_s*/population_masks.npz")):
            m = valbest(np.load(d)); ma.append(ev.evaluate_mask(m)[1].f1_macro); rt.append(float(m.sum() / D))
        res[geno] = (np.array(ma), rt)
    filt = {}
    for geno in ("faport", "bitstring"):
        ratios = res[geno][1]
        with contextlib.redirect_stdout(io.StringIO()):
            bm = build_baseline_masks(ev.clients, D, ratios, seed=0, fmlfs_max_features=0)
        for key, lab in FILTERS.items():
            filt[(geno, lab)] = np.array([ev.evaluate_mask(bm[key][r])[1].f1_macro for r in ratios])
    fw, base = res["faport"][0], res["bitstring"][0]
    sf_fw = max(FILTERS.values(), key=lambda l: filt[("faport", l)].mean())
    sf_b = max(FILTERS.values(), key=lambda l: filt[("bitstring", l)].mean())
    f_fw, f_b = filt[("faport", sf_fw)], filt[("bitstring", sf_b)]
    row = dict(dataset=ds, n=len(fw),
               fedwrap=fw.mean(), base=base.mean(),
               ratio_fedwrap=np.mean(res["faport"][1]), ratio_base=np.mean(res["bitstring"][1]),
               filter_at_base=sf_b, filter_at_base_macro=f_b.mean(),
               filter_at_fedwrap=sf_fw, filter_at_fedwrap_macro=f_fw.mean(),
               wrapper_effect=(base - f_b).mean(), wrapper_p=pval(base, f_b), wrapper_wins=int(np.sum(base > f_b)),
               fedaware_effect=(fw - base).mean(), fedaware_p=pval(fw, base), fedaware_wins=int(np.sum(fw > base)),
               total_effect=(fw - f_fw).mean(), total_p=pval(fw, f_fw), total_wins=int(np.sum(fw > f_fw)))
    rows.append(row)
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
with open(f"reports/revision/decomposition{SUFFIX}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
