"""Build the optimization-only datasets for the held-out-client generalization study.

For each federation we deterministically hold out the last ~25% of silos (sorted by id)
and write a prefold dataset containing ONLY the optimization silos, so that FedWrap-MLFS
can run its selection without ever seeing the held-out clients. The held-out silos are
recorded in a json so the evaluation step can score the selected mask on unseen clients.

  ECG (8 silos)          -> 6 optimization / 2 held-out
  eICU (12 silos)        -> 9 optimization / 3 held-out
  ExtraSensory (16 silos)-> 12 optimization / 4 held-out
"""
import sys, json
import numpy as np
from pathlib import Path
from scipy import sparse
sys.path.insert(0, ".")
from fedwrap.datasets import _load_npz_any

DATASETS = ["ECG_cinc2021", "eICU_expl_k12", "ExtraSensory"]
HELDOUT_FRAC = 0.25


def save_csr_pair(path, x, y):
    x = x.tocsr(); y = y.tocsr()
    np.savez(path, X_data=x.data, X_indices=x.indices, X_indptr=x.indptr, X_shape=np.array(x.shape),
             Y_data=y.data, Y_indices=y.indices, Y_indptr=y.indptr, Y_shape=np.array(y.shape))


def main():
    manifest = {}
    for ds in DATASETS:
        base = Path(f"data/fed_real/{ds}/fold0")
        xtr, ytr = _load_npz_any(base / "trainval.npz")
        xte, yte = _load_npz_any(base / "test.npz")
        gtr = np.load(base / "trainval_groups.npy", allow_pickle=True)
        gte = np.load(base / "test_groups.npy", allow_pickle=True)
        silos = sorted(set(gtr.tolist()))
        n_held = max(1, int(round(HELDOUT_FRAC * len(silos))))
        held = silos[-n_held:]
        opt = silos[:-n_held]
        tr_keep = np.array([g in opt for g in gtr.tolist()])
        te_keep = np.array([g in opt for g in gte.tolist()])
        out = Path(f"data/fed_real/{ds}__opt/fold0"); out.mkdir(parents=True, exist_ok=True)
        save_csr_pair(out / "trainval.npz", xtr[tr_keep], ytr[tr_keep])
        save_csr_pair(out / "test.npz", xte[te_keep], yte[te_keep])
        np.save(out / "trainval_groups.npy", gtr[tr_keep])
        np.save(out / "test_groups.npy", gte[te_keep])
        manifest[ds] = {"all_silos": [str(s) for s in silos],
                        "optimization_silos": [str(s) for s in opt],
                        "heldout_silos": [str(s) for s in held],
                        "n_opt": len(opt), "n_held": len(held),
                        "opt_dataset": f"{ds}__opt"}
        print(f"{ds}: {len(opt)} opt silos, {len(held)} held-out {held} "
              f"| opt rows tr/te={tr_keep.sum()}/{te_keep.sum()}", flush=True)
    Path("reports").mkdir(exist_ok=True)
    json.dump(manifest, open("reports/heldout_silo_manifest.json", "w"), indent=2)
    print("wrote reports/heldout_silo_manifest.json", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
