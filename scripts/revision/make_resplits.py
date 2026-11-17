"""Revision experiment (Reviewer 2, comment 3): independent train/test resplits of the natural federations.

The submitted protocol used one fixed per-silo 75/25 train/test partition (fold0) and varied, per seed,
the train/validation partition and the search. Here the non-test and test records of each silo are
pooled and re-partitioned per silo (75/25) with an independent split seed, so that the held-out test
set also changes across repetitions. Output: data/fed_real/{DS}__r{r}/fold0 in the prefold convention.

Usage: python scripts/revision/make_resplits.py [n_splits=10]
"""
import sys
from pathlib import Path
import numpy as np
from scipy import sparse
sys.path.insert(0, ".")
from fedwrap.datasets import _load_npz_any
from fedwrap.npz_format import save_csr_pair

FR = Path("data/fed_real")
DATASETS = ["ECG_cinc2021", "eICU_expl_k12", "ExtraSensory"]
R = int(sys.argv[1]) if len(sys.argv) > 1 else 10

for ds in DATASETS:
    src = FR / ds / "fold0"
    xtv, ytv = _load_npz_any(src / "trainval.npz"); xte, yte = _load_npz_any(src / "test.npz")
    gtv = np.load(src / "trainval_groups.npy", allow_pickle=True)
    gte = np.load(src / "test_groups.npy", allow_pickle=True)
    X = sparse.vstack([xtv, xte]).tocsr(); Y = sparse.vstack([ytv, yte]).tocsr()
    G = np.concatenate([gtv, gte])
    for r in range(1, R + 1):
        rng = np.random.default_rng(100_000 + r)
        te = np.zeros(X.shape[0], dtype=bool)
        for g in sorted(set(G.tolist())):
            idx = np.flatnonzero(G == g); rng.shuffle(idx)
            te[idx[: max(1, int(0.25 * len(idx)))]] = True
        tv_i = np.flatnonzero(~te); rng.shuffle(tv_i)
        te_i = np.flatnonzero(te)
        out = FR / f"{ds}__r{r}" / "fold0"; out.mkdir(parents=True, exist_ok=True)
        save_csr_pair(out / "trainval.npz", X[tv_i], Y[tv_i]); save_csr_pair(out / "test.npz", X[te_i], Y[te_i])
        np.save(out / "trainval_groups.npy", G[tv_i]); np.save(out / "test_groups.npy", G[te_i])
        ov = len(set(te_i.tolist()) & set(range(xtv.shape[0], X.shape[0]))) / max(1, len(te_i))
        print(f"{ds} r{r}: trainval={len(tv_i)} test={len(te_i)} silos={len(set(G.tolist()))} "
              f"test-overlap-with-original={ov:.2f}", flush=True)
