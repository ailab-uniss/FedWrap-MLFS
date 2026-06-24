"""Empirical corroboration of the resource-aware scheduler (no root / no netem needed).

The analytic study (run_scheduling_study.py) used each client's data mass as a proxy for compute time.
Here we instead MEASURE each client's real ML-kNN evaluation time, running the K clients as real
parallel OS processes, and report (a) the natural spread of those measured times (real heterogeneity on
the compute axis, before any injection) and (b) the critical-path speed-ups of the scheduling policies
when those measured times are combined with an injected per-tier slowdown and WAN delay. If the
measured-compute speed-ups match the analytic projection, the model is corroborated on the axis we can
actually measure; the per-tier factor and the WAN delay remain injected, since homogeneous hardware
cannot exhibit them (stated honestly in the paper).
"""
import sys, time, csv
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, ".")
from fedwrap.datasets import _load_npz_any
from fedwrap.ml_eval import EvalConfig
from fedwrap.federated import make_evaluator
from fedwrap.federated.metrics import compute_macro_f1

DS, NC = "eICU_expl_k12", 12
objn = ["one_minus_macro_f1", "one_minus_micro_f1", "feature_ratio"]
TIERS = {"edge": 20.0, "server": 5.0, "cloud": 1.0}      # injected compute slowdown
NET = {"edge": 0.060, "server": 0.030, "cloud": 0.010}   # injected per-tier WAN delay (s/round)

_EV = None  # per-process evaluator handle


def _init():
    global _EV
    xtr, ytr = _load_npz_any(f"data/fed_real/{DS}/fold0/trainval.npz")
    xte, yte = _load_npz_any(f"data/fed_real/{DS}/fold0/test.npz")
    gtr = np.load(f"data/fed_real/{DS}/fold0/trainval_groups.npy", allow_pickle=True)
    gte = np.load(f"data/fed_real/{DS}/fold0/test_groups.npy", allow_pickle=True)
    ec = EvalConfig(kind="mlknn", primary_objective=objn[0], objective_names=objn, random_state=0,
                    k=5, s=1.0, mlknn_backend="sklearn", mlknn_device="cpu")
    cfg = {"federated": {"enabled": True, "n_clients": NC, "partition": "natural_silo",
           "min_samples_per_client": 32, "client_fraction_full": 1.0, "final_eval_all_clients": True},
           "objectives": {"names": objn}}
    _EV = make_evaluator(xtr, ytr, xte, yte, ec, cfg, 0, groups=(gtr, gte))


def _time_client(args):
    """Measure one client's REAL ML-kNN evaluation time for a mask (runs in its own process)."""
    ci, mask_bytes, D = args
    mask = np.frombuffer(mask_bytes, dtype=bool)
    c = _EV.clients[ci]
    t0 = time.perf_counter()
    r = c.evaluate_mask(mask, mode="full")
    dt = time.perf_counter() - t0
    return ci, dt, r["tp"].astype(np.int64), r["fp"].astype(np.int64), r["fn"].astype(np.int64)


def main():
    _init()
    K = len(_EV.clients); L = _EV.clients[0].n_labels
    sizes = np.array([c.n_val for c in _EV.clients], dtype=float)
    tier_names = [list(TIERS)[i % 3] for i in range(K)]
    factor = np.array([TIERS[t] for t in tier_names]); net = np.array([NET[t] for t in tier_names])

    z = np.load(sorted(Path(".").glob(f"runs/plain_fed/{DS}_faport_s*/population_masks.npz"))[0])
    D = int(z["n_features"]); M = np.unpackbits(z["masks_packed"], axis=1)[:, :D].astype(bool)
    masks = [M[i] for i in np.flatnonzero(z["pareto_val_mask"].astype(bool))[:12]]

    # measure real per-client compute, running clients as real parallel processes.
    # pcc[mi] = (K,L) per-client TP/FP/FN for mask mi; comp[mi,ci] = measured compute seconds.
    comp = np.zeros((len(masks), K))
    pcc = [(np.zeros((K, L)), np.zeros((K, L)), np.zeros((K, L))) for _ in masks]
    wall_full = []
    with ProcessPoolExecutor(max_workers=min(K, 12), initializer=_init) as pool:
        for mi, m in enumerate(masks):
            args = [(ci, np.asarray(m, bool).tobytes(), D) for ci in range(K)]
            t0 = time.perf_counter()
            res = list(pool.map(_time_client, args))
            wall_full.append(time.perf_counter() - t0)        # real parallel round wall-clock
            for ci, dt, t, f, n in res:
                comp[mi, ci] = dt; pcc[mi][0][ci] = t; pcc[mi][1][ci] = f; pcc[mi][2][ci] = n
    tcomp = comp.mean(0)                                       # measured per-client compute (s)
    print(f"== {DS}: K={K}, L={L}, {len(masks)} masks ==")
    print(f"measured per-client ML-kNN compute: min={tcomp.min()*1e3:.1f}ms median={np.median(tcomp)*1e3:.1f}ms "
          f"max={tcomp.max()*1e3:.1f}ms -> natural compute spread {tcomp.max()/tcomp.min():.1f}x (real, no injection)")
    print(f"real parallel round wall-clock (full, no injection): {np.mean(wall_full)*1e3:.0f} ms")

    # realized per-client latency with measured compute + injected tier slowdown + WAN delay
    lat = tcomp * factor + net
    print(f"injected tiers {TIERS}, WAN {NET}; realized latency spread {lat.max()/lat.min():.1f}x")

    def sel(policy, q, dq):
        if policy == "full":
            idx = np.arange(K)
        else:
            order = np.argsort(lat / np.maximum(sizes, 1.0)) if policy == "ra" else np.argsort(lat)
            acc, chosen = 0.0, []
            for i in order:
                chosen.append(i); acc += sizes[i]
                if acc >= q * sizes.sum():
                    break
            idx = np.array(chosen)
        if dq < 1.0:
            keep = idx[lat[idx] <= np.quantile(lat, dq)]
            idx = keep if keep.size else idx
        return idx

    def dmacro_vs_full(idx):
        ds = []
        for tp, fp, fn in pcc:
            full = compute_macro_f1(tp.sum(0), fp.sum(0), fn.sum(0))
            sub = compute_macro_f1(tp[idx].sum(0), fp[idx].sum(0), fn[idx].sum(0))
            ds.append(abs(sub - full))
        return float(np.mean(ds))

    full_cp = lat.max()
    rows = []
    for lab, pol, q, dq in [("full", "full", 1.0, 1.0), ("RA quorum 0.6", "ra", 0.6, 1.0),
                            ("RA quorum 0.3 (cloud)", "ra", 0.3, 1.0), ("deadline p60", "full", 1.0, 0.60)]:
        idx = sel(pol, q, dq)
        speed = full_cp / lat[idx].max(); dmac = dmacro_vs_full(idx)
        print(f"  {lab:24s} particip={len(idx)/K:.2f}  measured-compute speedup={speed:5.2f}x  dmacro={dmac:.4f}")
        rows.append({"policy": lab, "participation": round(len(idx)/K, 2),
                     "speedup": round(speed, 2), "dmacro": round(dmac, 4)})
    with open("reports/scheduling_empirical.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["policy", "participation", "speedup", "dmacro"])
        w.writeheader(); w.writerows(rows)
    print("wrote reports/scheduling_empirical.csv")


if __name__ == "__main__":
    raise SystemExit(main())
