"""Compare search variants over the seeds available in runs/plain_fed and runs/loop1.

For each dataset and variant we report, over seeds, the val-selected macro-F1 (cap 0.25),
the final test-front hypervolume, and the number of full evaluations spent. Uses the stored
test_objs (computed by the same evaluator at run time) so the variants are directly comparable
without re-evaluating; the final winning config is re-checked with an independent re-eval.
"""
import sys, glob, json
import numpy as np
sys.path.insert(0, ".")
from fedwrap.metrics import hypervolume_3d, pareto_nondominated

REF = (1.0, 1.0, 1.0)
DATASETS = ["ECG_cinc2021", "eICU_expl_k12", "ExtraSensory"]
# (label, glob template)
VARIANTS = [
    ("base",     "runs/plain_fed/{ds}_bitstring_s{s}_fold0/population_masks.npz"),
    ("fedaware", "runs/plain_fed/{ds}_fedaware_s{s}_fold0/population_masks.npz"),
    ("fa+warm",  "runs/plain_fed/{ds}_fawarm_s{s}_fold0/population_masks.npz"),
]


def val_best_macro(z):
    D = int(z["n_features"]); M = np.unpackbits(z["masks_packed"], axis=1)[:, :D].astype(bool)
    vo, on = z["val_objs"], z["pareto_val_mask"].astype(bool); idx = np.flatnonzero(on)
    cap = M.sum(1) <= 0.25 * D + 1; cand = idx[cap[idx]] if cap[idx].any() else idx
    bi = cand[np.argmin(vo[cand, 0])]
    return 1.0 - float(z["test_objs"][bi, 0]), float(M[bi].sum() / D)


def main():
    for ds in DATASETS:
        print(f"\n=== {ds} ===")
        for label, tmpl in VARIANTS:
            ma, hv, ev, rt = [], [], [], []
            for s in range(0, 10):
                p = tmpl.format(ds=ds, s=s)
                fs = glob.glob(p)
                if not fs:
                    continue
                z = np.load(fs[0])
                m, r = val_best_macro(z); ma.append(m); rt.append(r)
                nd = pareto_nondominated(z["test_objs"]); hv.append(hypervolume_3d(nd, ref=REF))
                hp = fs[0].replace("population_masks.npz", "history.jsonl")
                try:
                    rows = [json.loads(l) for l in open(hp)]; ev.append(rows[-1]["evals"])
                except Exception:
                    pass
            if not ma:
                print(f"  {label:9s}: (no runs)"); continue
            print(f"  {label:9s}: macro={np.mean(ma):.4f}±{np.std(ma):.4f}  HV={np.mean(hv):.4f}"
                  f"  evals={np.mean(ev):.0f}  ratio={np.mean(rt):.3f}  (n={len(ma)})")


if __name__ == "__main__":
    main()
