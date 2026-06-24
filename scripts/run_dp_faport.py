"""Differential privacy of a single count release, on the deployed faport mask.

The only quantity a client releases per evaluation is its 3L count vector (TP,FP,FN), so the Gaussian
mechanism applies with L2 sensitivity sqrt(L). We take the val-selected faport mask, get the exact
summed clean counts, add calibrated noise (clipped to >=0, since counts are non-negative), and trace
the mean ABSOLUTE DEVIATION of macro/micro-F1 from the clean release as eps tightens. Deviation is the
honest quantity: it is monotone in eps and free of the sign confusion of plotting the F1 itself, whose
clip-induced bias pulls it toward the 0.5 fixed point. Writes reports/dp_faport_<ds>.csv.
"""
import sys, csv
import numpy as np
from pathlib import Path
sys.path.insert(0, ".")
from fedwrap.datasets import _load_npz_any
from fedwrap.ml_eval import EvalConfig
from fedwrap.federated import make_evaluator
from fedwrap.federated.metrics import compute_macro_f1, compute_micro_f1

objn = ["one_minus_macro_f1", "one_minus_micro_f1", "feature_ratio"]
DATASETS = [("ECG_cinc2021", 8), ("eICU_expl_k12", 12)]
EPS = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
DELTA = 1e-5
N_TRIALS = 2000


def valbest(z):
    D = int(z["n_features"]); M = np.unpackbits(z["masks_packed"], axis=1)[:, :D].astype(bool)
    vo, on = z["val_objs"], z["pareto_val_mask"].astype(bool); idx = np.flatnonzero(on)
    cap = M.sum(1) <= 0.25 * D + 1; cand = idx[cap[idx]] if cap[idx].any() else idx
    return M[cand[np.argmin(vo[cand, 0])]]


def main():
    for ds, nc in DATASETS:
        xtr, ytr = _load_npz_any(f"data/fed_real/{ds}/fold0/trainval.npz")
        xte, yte = _load_npz_any(f"data/fed_real/{ds}/fold0/test.npz")
        gtr = np.load(f"data/fed_real/{ds}/fold0/trainval_groups.npy", allow_pickle=True)
        gte = np.load(f"data/fed_real/{ds}/fold0/test_groups.npy", allow_pickle=True)
        ec = EvalConfig(kind="mlknn", primary_objective=objn[0], objective_names=objn, random_state=0,
                        k=5, s=1.0, mlknn_backend="sklearn", mlknn_device="cpu")
        cfg = {"federated": {"enabled": True, "n_clients": nc, "partition": "natural_silo",
               "min_samples_per_client": 32, "client_fraction_full": 1.0, "final_eval_all_clients": True},
               "objectives": {"names": objn}}
        ev = make_evaluator(xtr, ytr, xte, yte, ec, cfg, 0, groups=(gtr, gte))
        z = np.load(sorted(Path(".").glob(f"runs/plain_fed/{ds}_faport_s*/population_masks.npz"))[0])
        mask = valbest(z)
        # exact summed clean counts on the deployed mask
        L = ytr.shape[1]; TP = np.zeros(L); FP = np.zeros(L); FN = np.zeros(L)
        for c in ev.clients:
            r = c.evaluate_mask(np.asarray(mask, bool), mode="full")
            TP += r["tp"]; FP += r["fp"]; FN += r["fn"]
        macro0 = float(compute_macro_f1(TP, FP, FN)); micro0 = float(compute_micro_f1(TP, FP, FN))
        sens = np.sqrt(L); rng = np.random.default_rng(0)
        rows = [{"eps": "inf", "macro": round(macro0, 4), "micro": round(micro0, 4),
                 "dmacro": 0.0, "dmicro": 0.0}]
        print(f"\n== {ds}: L={L}, clean macro={macro0:.4f} micro={micro0:.4f}, "
              f"clean counts TP[min/med/max]={TP.min():.0f}/{np.median(TP):.0f}/{TP.max():.0f} ==")
        for eps in EPS:
            sigma = sens * np.sqrt(2.0 * np.log(1.25 / DELTA)) / eps
            ma, mi = [], []
            for _ in range(N_TRIALS):
                tpn = np.clip(TP + rng.normal(0, sigma, L), 0, None)
                fpn = np.clip(FP + rng.normal(0, sigma, L), 0, None)
                fnn = np.clip(FN + rng.normal(0, sigma, L), 0, None)
                ma.append(compute_macro_f1(tpn, fpn, fnn)); mi.append(compute_micro_f1(tpn, fpn, fnn))
            ma = np.array(ma); mi = np.array(mi)
            dmac = float(np.mean(np.abs(ma - macro0))); dmic = float(np.mean(np.abs(mi - micro0)))
            rows.append({"eps": eps, "macro": round(float(ma.mean()), 4), "micro": round(float(mi.mean()), 4),
                         "dmacro": round(dmac, 4), "dmicro": round(dmic, 4)})
            print(f"  eps={eps:5.1f} sigma={sigma:6.1f}  |dmacro|={dmac:.4f}  |dmicro|={dmic:.4f}  "
                  f"(macro {ma.mean():.3f}, micro {mi.mean():.3f})")
        with open(f"reports/dp_faport_{ds}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
