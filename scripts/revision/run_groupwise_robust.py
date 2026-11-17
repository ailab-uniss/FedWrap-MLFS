"""Revision experiment (Reviewer 2, comment 4b): outlier removal under secure aggregation.

Per-client outlier removal needs individual count vectors, which federation-wide secure aggregation
hides. Two compositions are evaluated on the count-flooding attack of scripts/run_robustness.py:

(a) Aggregate consistency check under federation-wide secure aggregation. For every mask and label,
    TP+FN equals the number of validation positives, a mask-independent quantity that the server
    obtains once as a secure sum. A violation of sum_i(TP_i+FN_i)[l] = P_val[l] (or of
    TP+FP <= N_val) is therefore detectable from the aggregate alone. Reports detection rate and
    false-alarm rate.

(b) Group-wise secure aggregation. Clients are partitioned at random into groups of at least g
    members; pairwise masks cancel within a group, so the server observes only group sums. The
    outlier rule of the paper (largest deviation of count mass from the median) is applied to group
    sums and the f most deviating groups are excluded. Honest members of an excluded group are lost.

Writes reports/revision/groupwise_robust.csv.
"""
import sys, glob, csv
import numpy as np
sys.path.insert(0, ".")
from fedwrap.datasets import _load_npz_any
from fedwrap.ml_eval import EvalConfig
from fedwrap.federated import make_evaluator
from fedwrap.federated.metrics import compute_macro_f1
from fedwrap.federated.secure_agg import secure_sum

objn = ["one_minus_macro_f1", "one_minus_micro_f1", "feature_ratio"]
DATASETS = [("ECG_cinc2021", 8), ("eICU_expl_k12", 12)]
N_TRIALS = 20     # random (victim set, grouping) draws per mask


def best_mask(npz):
    z = np.load(npz); D = int(z["n_features"])
    M = np.unpackbits(z["masks_packed"], axis=1)[:, :D].astype(bool)
    v, on = z["val_objs"], z["pareto_val_mask"].astype(bool); idx = np.flatnonzero(on)
    cap = M.sum(1) <= 0.25 * D + 1; cand = idx[cap[idx]] if cap[idx].any() else idx
    return M[cand[np.argmin(v[cand, 0])]]


def macro(vecs, L):
    s = np.sum(vecs, axis=0)
    return compute_macro_f1(s[:L], s[L:2 * L], s[2 * L:])


def flood(vecs, f, rng, L):
    """Same attack as scripts/run_robustness.py: TP=0 and FP=FN=clean_mass/L on every label."""
    big = int(max(1.0, float(np.sum(vecs)) / L))
    out = [v.copy() for v in vecs]
    vic = rng.choice(len(vecs), size=f, replace=False)
    for i in vic:
        out[i] = np.concatenate([np.zeros(L, np.int64), np.full(L, big, np.int64), np.full(L, big, np.int64)])
    return out, set(int(i) for i in vic)


def drop_clients(vecs, f, L):
    mass = np.array([float(v.sum()) for v in vecs]); dev = np.abs(mass - np.median(mass))
    keep = np.argsort(dev)[: len(vecs) - f]
    return macro([vecs[i] for i in keep], L)


def groupwise(vecs, f, g, rng, L):
    """Random groups of >= g clients; secure sum within each group; drop the f most deviating groups."""
    K = len(vecs); perm = rng.permutation(K); G = max(1, K // g)
    if G < 2 * f + 1:            # the median rule needs a majority of unaffected groups
        return float("nan"), [], 0
    groups = [perm[j::G] for j in range(G)]                       # sizes differ by at most one
    gsum = [secure_sum([vecs[i] for i in grp], base_seed=int(rng.integers(1 << 30))) for grp in groups]
    # count mass per member, so that groups of unequal size are comparable
    mass = np.array([float(s.sum()) / len(grp) for s, grp in zip(gsum, groups)])
    dev = np.abs(mass - np.median(mass)); order = np.argsort(dev)
    keep = order[: G - f]; dropped = order[G - f:]
    kept_clients = [int(i) for gi in keep for i in groups[gi]]
    lost = sum(len(groups[gi]) for gi in dropped)
    return macro([gsum[gi] for gi in keep], L), kept_clients, lost


def main():
    rows = []
    for ds, nc in DATASETS:
        xtr, ytr = _load_npz_any(f"data/fed_real/{ds}/fold0/trainval.npz")
        xte, yte = _load_npz_any(f"data/fed_real/{ds}/fold0/test.npz")
        gtr = np.load(f"data/fed_real/{ds}/fold0/trainval_groups.npy", allow_pickle=True)
        gte = np.load(f"data/fed_real/{ds}/fold0/test_groups.npy", allow_pickle=True)
        L = ytr.shape[1]
        ec = EvalConfig(kind="mlknn", primary_objective=objn[0], objective_names=objn,
                        random_state=0, k=5, s=1.0, mlknn_backend="sklearn", mlknn_device="cpu")
        cfg = {"federated": {"enabled": True, "n_clients": nc, "partition": "natural_silo",
                "min_samples_per_client": 32, "client_fraction_full": 1.0, "final_eval_all_clients": True},
               "objectives": {"names": objn}}
        ev = make_evaluator(xtr, ytr, xte, yte, ec, cfg, 0, groups=(gtr, gte))
        K = len(ev.clients)
        # mask-independent reference obtained once by secure sum: positives per label and N_val
        pos_ref = secure_sum([c._y_val_dense.sum(axis=0).astype(np.int64) for c in ev.clients], base_seed=1)
        n_ref = int(sum(c.n_val for c in ev.clients))
        npzs = sorted(glob.glob(f"runs/plain_fed/{ds}_faport_s*/population_masks.npz"))
        res = {}
        det = {"alarm_clean": [], "alarm_f1": [], "alarm_f2": []}
        for npz in npzs:
            m = best_mask(npz)
            vecs = []
            for c in ev.clients:
                r = c.evaluate_mask(np.asarray(m, bool), mode="full")
                vecs.append(np.concatenate([r["tp"], r["fp"], r["fn"]]).astype(np.int64))
            res.setdefault("clean", []).append(macro(vecs, L))

            def alarm(vv):
                s = secure_sum(vv, base_seed=3)
                tp, fp, fn = s[:L], s[L:2 * L], s[2 * L:]
                return bool(np.any(tp + fn != pos_ref) or np.any(tp + fp > n_ref))
            det["alarm_clean"].append(alarm(vecs))
            rng = np.random.default_rng(1)
            for f in (1, 2):
                for t in range(N_TRIALS):
                    cc, vic = flood(vecs, f, rng, L)
                    det[f"alarm_f{f}"].append(alarm(cc))
                    res.setdefault(f"f{f}_sum", []).append(macro(cc, L))
                    res.setdefault(f"f{f}_client", []).append(drop_clients(cc, f, L))
                    for g in (2, 3):
                        val, kept, lost = groupwise(cc, f, g, rng, L)
                        if not np.isfinite(val):
                            continue
                        res.setdefault(f"f{f}_group{g}", []).append(val)
                        res.setdefault(f"f{f}_group{g}_lost", []).append(lost)
                        res.setdefault(f"f{f}_group{g}_clean", []).append(float(len(vic & set(kept)) == 0))
        row = {"dataset": ds, "K": K, "n_masks": len(npzs)}
        row.update({k: round(float(np.mean(v)), 4) for k, v in res.items()})
        row.update({k: round(float(np.mean(v)), 4) for k, v in det.items()})
        rows.append(row)
        print(row, flush=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("dataset", "K", "n_masks"), k))
    with open("reports/revision/groupwise_robust.csv", "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print("wrote reports/revision/groupwise_robust.csv")


if __name__ == "__main__":
    raise SystemExit(main())
