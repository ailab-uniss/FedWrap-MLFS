"""Paired Wilcoxon: relevance-warmstart FedAware vs base NSGA-II and vs no-warmstart FedAware.

Pairs per-seed val-selected macro-F1 (cap 0.25) and final test-front hypervolume over the seeds
present for all three variants, then runs scipy's Wilcoxon signed-rank test. Also reports the mean
number of full evaluations (federated rounds) each variant spends.
"""
import sys, glob, json
import numpy as np
sys.path.insert(0, ".")
from scipy.stats import wilcoxon
from fedwrap.metrics import hypervolume_3d, pareto_nondominated

REF = (1.0, 1.0, 1.0)
DATASETS = ["ECG_cinc2021", "eICU_expl_k12", "ExtraSensory"]
TMPL = {"base": "runs/plain_fed/{ds}_bitstring_s{s}_fold0",
        "fedaware": "runs/plain_fed/{ds}_fedaware_s{s}_fold0",
        "fa+warm": "runs/plain_fed/{ds}_fawarm30_s{s}_fold0"}


def metrics(rundir):
    z = np.load(f"{rundir}/population_masks.npz")
    D = int(z["n_features"]); M = np.unpackbits(z["masks_packed"], axis=1)[:, :D].astype(bool)
    vo, on = z["val_objs"], z["pareto_val_mask"].astype(bool); idx = np.flatnonzero(on)
    cap = M.sum(1) <= 0.25 * D + 1; cand = idx[cap[idx]] if cap[idx].any() else idx
    bi = cand[np.argmin(vo[cand, 0])]
    macro = 1.0 - float(z["test_objs"][bi, 0])
    hv = hypervolume_3d(pareto_nondominated(z["test_objs"]), ref=REF)
    try:
        ev = [json.loads(l) for l in open(f"{rundir}/history.jsonl")][-1]["evals"]
    except Exception:
        ev = np.nan
    return macro, hv, ev


def collect(ds, variant):
    out = {}
    for s in range(10):
        d = TMPL[variant].format(ds=ds, s=s)
        if glob.glob(f"{d}/population_masks.npz"):
            out[s] = metrics(d)
    return out


def main():
    for ds in DATASETS:
        cols = {v: collect(ds, v) for v in TMPL}
        seeds = sorted(set(cols["base"]) & set(cols["fa+warm"]) & set(cols["fedaware"]))
        print(f"\n=== {ds}  (paired seeds: {len(seeds)}) ===")
        arr = {v: np.array([[cols[v][s][0], cols[v][s][1], cols[v][s][2]] for s in seeds]) for v in TMPL}
        for v in TMPL:
            a = arr[v]
            print(f"  {v:9s}: macro={a[:,0].mean():.4f}±{a[:,0].std():.4f}  HV={a[:,1].mean():.4f}±{a[:,1].std():.4f}  evals={np.nanmean(a[:,2]):.0f}")
        for ref in ("base", "fedaware"):
            for j, name in [(0, "macro"), (1, "HV")]:
                x, y = arr["fa+warm"][:, j], arr[ref][:, j]
                if len(seeds) >= 6 and np.any(x != y):
                    p = float(wilcoxon(x, y).pvalue)
                    tag = "SIG" if p < 0.05 else ("~" if p < 0.10 else "ns")
                    print(f"    fa+warm vs {ref:8s} {name:5s}: d={x.mean()-y.mean():+.4f}  p={p:.4f} {tag}")


if __name__ == "__main__":
    main()
