"""Refresh the main real-data table with the canonical 10-seed FedAware (warmstart frac=0.3) runs.

FedWrap (fawarm30), base NSGA-II (bitstring), and local-avg are read from saved runs and re-evaluated
with one independent federated evaluator; the federated filters are matched to FedWrap's per-seed
ratio (fast ones per seed; FMLFS once at the mean ratio, as its O(D^2) build is the bottleneck).
Paired Wilcoxon: FedWrap vs base and vs each fast filter. Prints table-ready rows.
"""
import sys, glob, csv
import numpy as np
sys.path.insert(0, ".")
from scipy.stats import wilcoxon
from fedwrap.datasets import _load_npz_any
from fedwrap.ml_eval import EvalConfig
from fedwrap.federated import make_evaluator
from fedwrap.federated.baselines import build_baseline_masks, fmlfs
from fedwrap.metrics import hypervolume_3d, pareto_nondominated

DATASETS = [("ECG_cinc2021", 8), ("eICU_expl_k12", 12), ("ExtraSensory", 16)]
OURS = {"fawarm30": "FedWrap-MLFS", "bitstring": "base NSGA-II", "localavg": "local-avg wrapper"}
objn = ["one_minus_macro_f1", "one_minus_micro_f1", "feature_ratio"]
REF = (1.0, 1.0, 1.0)


def valbest(z):
    D = int(z["n_features"]); M = np.unpackbits(z["masks_packed"], axis=1)[:, :D].astype(bool)
    vo, on = z["val_objs"], z["pareto_val_mask"].astype(bool); idx = np.flatnonzero(on)
    cap = M.sum(1) <= 0.25 * D + 1; cand = idx[cap[idx]] if cap[idx].any() else idx
    return M, cand[np.argmin(vo[cand, 0])]


def main():
    import io, contextlib
    rows_csv = []
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

        ours = {}
        ratios = []
        for geno, lab in OURS.items():
            ma, mi, rt, hv = [], [], [], []
            for d in sorted(glob.glob(f"runs/plain_fed/{ds}_{geno}_s*/population_masks.npz")):
                z = np.load(d); M, bi = valbest(z)
                _, r = ev.evaluate_mask(M[bi]); ma.append(r.f1_macro); mi.append(r.f1_micro)
                rt.append(float(M[bi].sum() / D))
                hv.append(hypervolume_3d(pareto_nondominated(z["test_objs"]), ref=REF))
            ours[geno] = dict(macro=np.array(ma), micro=np.array(mi), ratio=np.array(rt), hv=np.array(hv))
            if geno == "fawarm30":
                ratios = rt
        print(f"\n===== {ds} (D={D}, {len(ratios)} seeds) =====")
        for geno, lab in OURS.items():
            o = ours[geno]
            print(f"  {lab:20s} macro={o['macro'].mean():.4f}±{o['macro'].std():.4f} "
                  f"micro={o['micro'].mean():.4f} ratio={o['ratio'].mean():.3f} "
                  f"#feat={o['ratio'].mean()*D:.0f} HV={o['hv'].mean():.3f}")
            rows_csv.append(dict(dataset=ds, method=lab, macro=round(o['macro'].mean(), 4),
                                 macro_std=round(o['macro'].std(), 4), micro=round(o['micro'].mean(), 4),
                                 nfeat=int(round(o['ratio'].mean()*D)), ratio=round(o['ratio'].mean(), 3)))

        # fast filters matched per-seed; FMLFS once at the mean ratio
        with contextlib.redirect_stdout(io.StringIO()):
            bm = build_baseline_masks(ev.clients, D, ratios, seed=0, fmlfs_max_features=0)
        names = {"fed_rank_relevance": "fed-rank", "topk_frequency": "top-frequency",
                 "local_topk_union": "local-top-k union", "all_features": "all features"}
        filt = {}
        for key, lab in names.items():
            vals, mics = [], []
            for rt in ratios:
                mk = bm.get(key, {}).get(1.0 if key == "all_features" else rt)
                if mk is not None:
                    r = ev.evaluate_mask(mk)[1]; vals.append(r.f1_macro); mics.append(r.f1_micro)
                    nf = int(mk.sum())
            filt[lab] = np.array(vals)
            print(f"  {lab:20s} macro={np.mean(vals):.4f}±{np.std(vals):.4f} micro={np.mean(mics):.4f}")
            rows_csv.append(dict(dataset=ds, method=lab, macro=round(np.mean(vals), 4),
                                 macro_std=round(np.std(vals), 4), micro=round(np.mean(mics), 4),
                                 nfeat=(D if lab == "all features" else int(round(np.mean(ratios)*D))),
                                 ratio=(1.0 if lab == "all features" else round(np.mean(ratios), 3))))
        mean_r = float(np.mean(ratios))
        with contextlib.redirect_stdout(io.StringIO()):
            fm = fmlfs(ev.clients, mean_r)
        rfm = ev.evaluate_mask(fm)[1]
        print(f"  {'FMLFS (@mean ratio)':20s} macro={rfm.f1_macro:.4f} micro={rfm.f1_micro:.4f}")
        rows_csv.append(dict(dataset=ds, method="FMLFS", macro=round(rfm.f1_macro, 4), macro_std=0.0,
                             micro=round(rfm.f1_micro, 4), nfeat=int(round(mean_r*D)), ratio=round(mean_r, 3)))

        # Wilcoxon: FedWrap vs base, vs strongest fast filter
        fw = ours["fawarm30"]["macro"]
        for ref_lab, arr in [("base NSGA-II", ours["bitstring"]["macro"])] + list(filt.items()):
            if len(arr) == len(fw) and np.any(fw != arr):
                p = wilcoxon(fw, arr).pvalue
                tag = "SIG" if p < 0.05 else ("~" if p < 0.10 else "ns")
                print(f"    Wilcoxon FedWrap vs {ref_lab:18s} d={fw.mean()-arr.mean():+.4f} p={p:.4f} {tag}")

    with open("reports/maintable_warm.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "method", "macro", "macro_std", "micro", "nfeat", "ratio"])
        w.writeheader(); w.writerows(rows_csv)
    print("\nwrote reports/maintable_warm.csv")


if __name__ == "__main__":
    raise SystemExit(main())
