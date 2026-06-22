"""Workflow-level performance: federated evaluation throughput on the real federations.

Times the federated wrapper-evaluation round (broadcast mask -> per-client ML-kNN -> exact count
aggregation), with client-level thread parallelism as deployed, and reports evaluations/second and
the implied wall-clock for a full search. Combined with the communication accounting (Table 9) this
gives the workflow's systems profile.
"""
import sys, time
import numpy as np
sys.path.insert(0, ".")
from fedwrap.datasets import _load_npz_any
from fedwrap.ml_eval import EvalConfig
from fedwrap.federated import make_evaluator

objn = ["one_minus_macro_f1", "one_minus_micro_f1", "feature_ratio"]
DATASETS = [("ECG_cinc2021", 8, 4900), ("eICU_expl_k12", 12, 5850), ("ExtraSensory", 16, 4000)]
N_WARM, N_TIME = 5, 40


def main():
    print(f"{'dataset':14s}{'K':>4}{'D':>6}{'eval/s':>9}{'evals':>8}{'runtime(s)':>12}")
    for ds, nc, n_evals in DATASETS:
        xtr, ytr = _load_npz_any(f"data/fed_real/{ds}/fold0/trainval.npz")
        xte, yte = _load_npz_any(f"data/fed_real/{ds}/fold0/test.npz")
        gtr = np.load(f"data/fed_real/{ds}/fold0/trainval_groups.npy", allow_pickle=True)
        gte = np.load(f"data/fed_real/{ds}/fold0/test_groups.npy", allow_pickle=True)
        D = xtr.shape[1]
        ec = EvalConfig(kind="mlknn", primary_objective=objn[0], objective_names=objn,
                        random_state=0, k=5, s=1.0, mlknn_backend="sklearn", mlknn_device="cpu")
        cfg = {"federated": {"enabled": True, "n_clients": nc, "partition": "natural_silo",
               "min_samples_per_client": 32, "client_fraction_full": 1.0,
               "final_eval_all_clients": True, "n_jobs": 8}, "objectives": {"names": objn}}
        ev = make_evaluator(xtr, ytr, xte, yte, ec, cfg, 0, groups=(gtr, gte))
        rng = np.random.default_rng(0)
        masks = [rng.random(D) < 0.1 for _ in range(N_WARM + N_TIME)]
        for m in masks[:N_WARM]:
            ev.evaluate_mask(m)
        t0 = time.perf_counter()
        for m in masks[N_WARM:]:
            ev.evaluate_mask(m)
        dt = time.perf_counter() - t0
        eps = N_TIME / dt
        print(f"{ds:14s}{nc:>4}{D:>6}{eps:>9.1f}{n_evals:>8}{n_evals/eps:>12.0f}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
