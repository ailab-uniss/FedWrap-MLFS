"""Refresh the client-dispersion table with the canonical 10-seed FedAware (warmstart) masks.

For each dataset and each of the 10 fawarm30 seeds, take the val-selected deployed mask and the
matched fed-rank mask, and compute the per-client local-support macro-F1 distribution (F1 averaged
over the labels each silo actually observes). Report the seed-averaged global macro and the
mean/std/worst/tenth-percentile of the client distribution.
"""
import sys, glob
import numpy as np
sys.path.insert(0, ".")
from fedwrap.datasets import _load_npz_any
from fedwrap.ml_eval import EvalConfig
from fedwrap.federated import make_evaluator
from fedwrap.federated.baselines import fed_rank_relevance, ranking_to_mask

DATASETS = [("ECG_cinc2021", 8), ("eICU_expl_k12", 12), ("ExtraSensory", 16)]
objn = ["one_minus_macro_f1", "one_minus_micro_f1", "feature_ratio"]


def valbest(z):
    D = int(z["n_features"]); M = np.unpackbits(z["masks_packed"], axis=1)[:, :D].astype(bool)
    vo, on = z["val_objs"], z["pareto_val_mask"].astype(bool); idx = np.flatnonzero(on)
    cap = M.sum(1) <= 0.25 * D + 1; cand = idx[cap[idx]] if cap[idx].any() else idx
    return M[cand[np.argmin(vo[cand, 0])]]


def dist(ev, mask):
    """global macro + per-client local-support macro-F1 list for a mask."""
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
    per = np.array(per)
    return gmacro, per.mean(), per.std(), per.min(), float(np.percentile(per, 10))


def main():
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
        fed_scores = fed_rank_relevance(ev.clients)
        rows = {"FedWrap": [], "fed-rank": []}
        for d in sorted(glob.glob(f"runs/plain_fed/{ds}_fawarm30_s*/population_masks.npz")):
            m = valbest(np.load(d)); r = float(m.sum() / D)
            rows["FedWrap"].append(dist(ev, m))
            rows["fed-rank"].append(dist(ev, ranking_to_mask(fed_scores, r)))
        print(f"\n== {ds} (n={len(rows['FedWrap'])}) ==")
        for meth, vals in rows.items():
            a = np.array(vals)  # cols: gmacro, mean, std, worst, p10
            mu = a.mean(0)
            print(f"  {meth:9s} global={mu[0]:.3f} mean={mu[1]:.3f} std={mu[2]:.3f} "
                  f"worst={mu[3]:.3f} P10={mu[4]:.3f}")


if __name__ == "__main__":
    raise SystemExit(main())
