"""Secure-aggregation prototype: verify exactness and measure overhead on the real eICU federation.

(1) Exactness: aggregate the per-client TP/FP/FN of the deployed faport masks both plainly and through
    additive pairwise masking; the reconstructed global macro/micro-F1 must be bitwise identical.
(2) Overhead: time one secure-aggregation round per generation (the whole population's count vectors
    batched into a single masked exchange) against plain summation, over a realistic number of
    generations, and report the added latency and the communication footprint.
"""
import sys, time, csv
import numpy as np
from pathlib import Path
sys.path.insert(0, ".")
from fedwrap.datasets import _load_npz_any
from fedwrap.ml_eval import EvalConfig
from fedwrap.federated import make_evaluator
from fedwrap.federated.metrics import compute_macro_f1, compute_micro_f1
from fedwrap.federated.secure_agg import secure_sum

DS, NC = "eICU_expl_k12", 12
objn = ["one_minus_macro_f1", "one_minus_micro_f1", "feature_ratio"]
POP, GENS = 50, 100   # one secure-agg round per generation aggregates the whole population's counts


def valbest_masks(z, n=12):
    D = int(z["n_features"]); M = np.unpackbits(z["masks_packed"], axis=1)[:, :D].astype(bool)
    on = np.flatnonzero(z["pareto_val_mask"].astype(bool))
    return [M[i] for i in on[:n]]


def main():
    xtr, ytr = _load_npz_any(f"data/fed_real/{DS}/fold0/trainval.npz")
    xte, yte = _load_npz_any(f"data/fed_real/{DS}/fold0/test.npz")
    gtr = np.load(f"data/fed_real/{DS}/fold0/trainval_groups.npy", allow_pickle=True)
    gte = np.load(f"data/fed_real/{DS}/fold0/test_groups.npy", allow_pickle=True)
    ec = EvalConfig(kind="mlknn", primary_objective=objn[0], objective_names=objn, random_state=0,
                    k=5, s=1.0, mlknn_backend="sklearn", mlknn_device="cpu")
    cfg = {"federated": {"enabled": True, "n_clients": NC, "partition": "natural_silo",
           "min_samples_per_client": 32, "client_fraction_full": 1.0, "final_eval_all_clients": True},
           "objectives": {"names": objn}}
    ev = make_evaluator(xtr, ytr, xte, yte, ec, cfg, 0, groups=(gtr, gte))
    K = len(ev.clients); L = ytr.shape[1]

    # (1) EXACTNESS on real deployed masks
    z = np.load(sorted(Path(".").glob(f"runs/plain_fed/{DS}_faport_s*/population_masks.npz"))[0])
    max_diff = 0.0
    for mask in valbest_masks(z):
        per = [c.evaluate_mask(np.asarray(mask, bool), mode="full") for c in ev.clients]
        # concatenate [TP|FP|FN] per client (the count vector a client would secret-share)
        vecs = [np.concatenate([r["tp"], r["fp"], r["fn"]]).astype(np.int64) for r in per]
        plain = np.sum(vecs, axis=0)
        sec = secure_sum(vecs, base_seed=12345)
        assert np.array_equal(plain, sec), "secure sum != plain sum"
        tp, fp, fn = plain[:L], plain[L:2*L], plain[2*L:]
        tps, fps, fns = sec[:L], sec[L:2*L], sec[2*L:]
        max_diff = max(max_diff,
                       abs(compute_macro_f1(tp, fp, fn) - compute_macro_f1(tps, fps, fns)),
                       abs(compute_micro_f1(tp, fp, fn) - compute_micro_f1(tps, fps, fns)))
    print(f"[exactness] K={K} L={L}: max |F1_secure - F1_plain| over deployed masks = {max_diff:.2e}")

    # (2) OVERHEAD: one secure-agg round per generation over POP candidates (dim = POP*3L per client)
    dim = POP * 3 * L
    rng = np.random.default_rng(0)
    batches = [[rng.integers(0, 5000, size=dim).astype(np.int64) for _ in range(K)] for _ in range(GENS)]
    t0 = time.perf_counter()
    for b in batches:
        _ = np.sum(b, axis=0)
    t_plain = time.perf_counter() - t0
    t0 = time.perf_counter()
    for b in batches:
        _ = secure_sum(b, base_seed=7)
    t_secure = time.perf_counter() - t0

    payload_kb = K * dim * 8 / 1024.0                 # masked vectors are the SAME size as plaintext
    setup_kb = (K * (K - 1) // 2) * 32 / 1024.0       # one-time pairwise 256-bit seeds (DH key agreement)
    added_ms = (t_secure - t_plain) / GENS * 1e3
    # contextualise against the actual federated compute of one generation (POP evaluations); eICU
    # runs at ~12.4 evaluations/s (Table tab:comm), so a generation already costs ~POP/eval_per_s s.
    eval_per_s = 12.4
    gen_compute_ms = POP / eval_per_s * 1e3
    pct_of_compute = 100 * added_ms / gen_compute_ms
    print(f"[overhead] dim/round/client={dim} ({POP} cand x 3L), {GENS} rounds")
    print(f"  masking adds +{added_ms:.2f} ms per generation-round (server-side, all {K} clients serial)")
    print(f"  one generation already computes {POP} evaluations ~= {gen_compute_ms/1e3:.1f} s at {eval_per_s} eval/s")
    print(f"  -> secure-agg overhead = {pct_of_compute:.2f}% of per-generation compute")
    print(f"  per-round payload identical: {payload_kb:.0f} KB ({K} clients); one-time setup {setup_kb:.1f} KB")

    Path("reports").mkdir(exist_ok=True)
    with open("reports/secure_agg_overhead.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["metric", "value"])
        w.writerows([("max_f1_diff", f"{max_diff:.2e}"), ("added_ms_per_gen_round", round(added_ms, 3)),
                     ("gen_compute_s", round(gen_compute_ms/1e3, 2)),
                     ("overhead_pct_of_compute", round(pct_of_compute, 3)),
                     ("payload_kb_per_round", round(payload_kb, 1)), ("setup_kb_once", round(setup_kb, 1))])


if __name__ == "__main__":
    raise SystemExit(main())
