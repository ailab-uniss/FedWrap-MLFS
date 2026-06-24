"""Resource-aware scheduling replay study on real evaluated masks (fills tab:scheduling).

For a federation, precompute each client's TP/FP/FN for every mask on a run's Pareto front (once),
then replay scheduling policies analytically under an emulated three-tier latency model: full
participation, a data-mass quorum, a latency deadline (straggler drop), and their combination.
Reports critical-path speed-up vs the synchronous baseline against the mean absolute deviation from
full-participation macro-F1 (exact on the responding subset).
"""
import sys, glob
import numpy as np
sys.path.insert(0, ".")
from fedwrap.datasets import _load_npz_any
from fedwrap.ml_eval import EvalConfig
from fedwrap.federated import make_evaluator
from fedwrap.federated.scheduling import TierModel, replay_policy

objn = ["one_minus_macro_f1", "one_minus_micro_f1", "feature_ratio"]
DS, NC = "eICU_expl_k12", 12          # the heterogeneous clinical federation
N_SEEDS, MAX_MASKS = 5, 40            # masks sampled from each run's Pareto front


def main():
    xtr, ytr = _load_npz_any(f"data/fed_real/{DS}/fold0/trainval.npz")
    xte, yte = _load_npz_any(f"data/fed_real/{DS}/fold0/test.npz")
    gtr = np.load(f"data/fed_real/{DS}/fold0/trainval_groups.npy", allow_pickle=True)
    gte = np.load(f"data/fed_real/{DS}/fold0/test_groups.npy", allow_pickle=True)
    D = xtr.shape[1]
    ec = EvalConfig(kind="mlknn", primary_objective=objn[0], objective_names=objn,
                    random_state=0, k=5, s=1.0, mlknn_backend="sklearn", mlknn_device="cpu")
    cfg = {"federated": {"enabled": True, "n_clients": NC, "partition": "natural_silo",
           "min_samples_per_client": 32, "client_fraction_full": 1.0, "final_eval_all_clients": True},
           "objectives": {"names": objn}}
    ev = make_evaluator(xtr, ytr, xte, yte, ec, cfg, 0, groups=(gtr, gte))
    K = len(ev.clients); L = ytr.shape[1]
    sizes = np.array([c.n_val for c in ev.clients], dtype=float)
    # Realistic continuum: edge nodes ~20x slower per unit compute than a cloud/HPC node, on-prem
    # servers ~5x; a small fixed communication term that does not dominate (so the tier factor, not a
    # floor, sets the disparity). Round-robin tier assignment over the K clients.
    tier = TierModel(tiers={"edge": 20.0, "server": 5.0, "cloud": 1.0}, comm_fixed=0.01)
    lat = tier.latencies(sizes)
    fac = np.array([tier.tiers[n] for n in tier.assign(K)])

    # collect masks from a few runs' Pareto fronts
    masks = []
    for d in sorted(glob.glob(f"runs/plain_fed/{DS}_faport_s*/population_masks.npz"))[:N_SEEDS]:
        z = np.load(d); Dz = int(z["n_features"])
        M = np.unpackbits(z["masks_packed"], axis=1)[:, :Dz].astype(bool)
        on = np.flatnonzero(z["pareto_val_mask"].astype(bool))
        for i in on[:MAX_MASKS]:
            masks.append(M[i])
    print(f"{DS}: K={K} L={L}, {len(masks)} masks; tier factor={fac.max()/fac.min():.0f}x, "
          f"realized latency ratio={lat.max()/lat.min():.1f}x (min {lat.min():.3f}, max {lat.max():.3f})",
          flush=True)

    # precompute per-client counts for each mask (the only compute)
    tp = np.zeros((len(masks), K, L)); fp = np.zeros_like(tp); fn = np.zeros_like(tp)
    for mi, m in enumerate(masks):
        for ci, c in enumerate(ev.clients):
            r = c.evaluate_mask(np.asarray(m, bool), mode="full")
            tp[mi, ci] = r["tp"]; fp[mi, ci] = r["fp"]; fn[mi, ci] = r["fn"]

    full = replay_policy(tp, fp, fn, sizes, lat, "full", 1.0)
    print(f"\n{'policy':30s}{'particip.':>10}{'speedup':>9}{'dmacro':>9}")
    print(f"{'full (synchronous)':30s}{full['mean_participation']:>10.2f}{1.0:>8.2f}x{0.0:>9.4f}")
    for lab, pol, q, dq in [("deadline p80 (full mass)", "full", 1.0, 0.80),
                            ("deadline p60 (full mass)", "full", 1.0, 0.60),
                            ("resource-aware quorum 0.8", "resource_aware", 0.8, 1.0),
                            ("resource-aware quorum 0.6", "resource_aware", 0.6, 1.0),
                            ("resource-aware quorum 0.3", "resource_aware", 0.3, 1.0),
                            ("uniform quorum 0.6", "uniform", 0.6, 1.0),
                            ("quorum 0.6 + deadline p60", "resource_aware", 0.6, 0.60)]:
        r = replay_policy(tp, fp, fn, sizes, lat, pol, q, deadline_q=dq, seed=1)
        print(f"{lab:30s}{r['mean_participation']:>10.2f}{full['walltime']/r['walltime']:>8.2f}x"
              f"{r['mae_vs_full']:>9.4f}")


if __name__ == "__main__":
    raise SystemExit(main())
