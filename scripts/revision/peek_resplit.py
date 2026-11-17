"""Summary of the per-split rows cached so far (no computation)."""
import json, glob, numpy as np
from scipy.stats import wilcoxon
for ds in ["ECG_cinc2021", "eICU_expl_k12", "ExtraSensory"]:
    recs = [json.load(open(f)) for f in sorted(glob.glob(f"reports/revision/resplit_cache/{ds}_r*.json"))]
    if not recs: continue
    print(f"== {ds}: {len(recs)} splits {[r['split'] for r in recs]}")
    fw = np.array([r["faport"]["macro"] for r in recs])
    for k in ["faport", "bitstring", "local-top-k union", "fed-rank", "top-frequency", "FMLFS", "all features", "fed-rank@base", "top-frequency@base"]:
        a = np.array([r[k]["macro"] for r in recs]); nf = np.mean([r[k]["nfeat"] for r in recs])
        p = wilcoxon(fw, a).pvalue if (len(recs) > 5 and np.any(fw != a)) else float("nan")
        print(f"   {k:20s} {a.mean():.4f} ± {a.std():.4f}  nfeat {nf:6.1f}  FedWrap-diff {np.mean(fw-a):+.4f}  wins {int(np.sum(fw>a))}/{len(recs)}  p={p:.4f}")
    for key in ["total_evals", "unique_evals", "gens", "hv_val", "hv_test"]:
        a = np.array([r["faport"][key] for r in recs], float); b = np.array([r["bitstring"][key] for r in recs], float)
        print(f"   {key:14s} FedAware {a.mean():9.3f}  base {b.mean():9.3f}  ratio {a.mean()/b.mean():.3f}  FedAware lower in {int(np.sum(a<b))}/{len(recs)}")
