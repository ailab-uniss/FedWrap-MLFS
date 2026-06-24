"""Classifier-MATCHED comparison (the methodologically correct wrapper test).

For each downstream classifier C in {ML-kNN, BR-LogReg}, the wrapper is run WITH C as its inner
evaluator (features selected for the model that will be deployed), and we compare FedWrap-C against the
filters evaluated under C at the matched feature ratio. The wrapper is never asked to transfer a subset
selected for one classifier to a different one. Reads runs/plain_fed/{ds}_faport_s* (ML-kNN wrapper) and
runs/plain_fed_logreg/{ds}_faport_s* (LogReg wrapper). Writes reports/classifier_matched.csv.
"""
import sys, glob, csv
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


def evaluator(ds, nc, kind):
    xtr, ytr = _load_npz_any(f"data/fed_real/{ds}/fold0/trainval.npz")
    xte, yte = _load_npz_any(f"data/fed_real/{ds}/fold0/test.npz")
    gtr = np.load(f"data/fed_real/{ds}/fold0/trainval_groups.npy", allow_pickle=True)
    gte = np.load(f"data/fed_real/{ds}/fold0/test_groups.npy", allow_pickle=True)
    ec = EvalConfig(kind=kind, primary_objective=objn[0], objective_names=objn, random_state=0,
                    k=5, s=1.0, mlknn_backend="sklearn", mlknn_device="cpu")
    cfg = {"federated": {"enabled": True, "n_clients": nc, "partition": "natural_silo",
           "min_samples_per_client": 32, "client_fraction_full": 1.0, "final_eval_all_clients": True},
           "objectives": {"names": objn}}
    return make_evaluator(xtr, ytr, xte, yte, ec, cfg, 0, groups=(gtr, gte)), xtr.shape[1]


def regime(ds, nc, kind, run_glob):
    runs = sorted(glob.glob(run_glob))
    if not runs:
        return None
    ev, D = evaluator(ds, nc, kind)
    fr = fed_rank_relevance(ev.clients)
    fw, ratios = [], []
    for rp in runs:
        m = valbest(np.load(rp)); ratios.append(m.sum() / D)
        fw.append(ev.evaluate_mask(m)[1].f1_macro)
    r = float(np.mean(ratios))
    fed = ev.evaluate_mask(ranking_to_mask(fr, r))[1].f1_macro
    allf = ev.evaluate_mask(np.ones(D, bool))[1].f1_macro
    return dict(fedwrap=float(np.mean(fw)), fed_rank=fed, all=allf, ratio=r, n=len(fw))


def main():
    rows = []
    for ds, nc in DATASETS:
        kn = regime(ds, nc, "mlknn", f"runs/plain_fed/{ds}_faport_s*/population_masks.npz")
        lr = regime(ds, nc, "logreg", f"runs/plain_fed_logreg/{ds}_faport_s*/population_masks.npz")
        print(f"\n== {ds} ==")
        for cls, reg in [("ML-kNN", kn), ("BR-LogReg", lr)]:
            if reg is None:
                print(f"  {cls:10s}: (no runs yet)"); continue
            print(f"  {cls:10s}: FedWrap={reg['fedwrap']:.4f}  fed-rank={reg['fed_rank']:.4f}  "
                  f"all={reg['all']:.4f}  (ratio {reg['ratio']:.3f}, n={reg['n']})")
            rows.append(dict(dataset=ds, classifier=cls, ratio=round(reg["ratio"], 3),
                             all_features=round(reg["all"], 4), fed_rank=round(reg["fed_rank"], 4),
                             fedwrap=round(reg["fedwrap"], 4), n_seeds=reg["n"]))
    if rows:
        with open("reports/classifier_matched.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["dataset", "classifier", "ratio", "all_features",
                                              "fed_rank", "fedwrap", "n_seeds"])
            w.writeheader(); w.writerows(rows)
        print("\nwrote reports/classifier_matched.csv")


if __name__ == "__main__":
    raise SystemExit(main())
