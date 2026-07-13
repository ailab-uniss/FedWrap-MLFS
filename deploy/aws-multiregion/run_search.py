#!/usr/bin/env python3
"""Real FedAware-NSGA-II wrapper feature selection executed ACROSS the 3 multi-region AWS silos.

This runs the framework's actual method, not a simplified one:
  * one-time federated relevance sketch R, aggregated from each silo's sufficient statistics
    (Y^T X, sum X, sum X^2, label counts, n) -- no raw features leave a silo;
  * filter-seeded + relevance warm-start initialization;
  * relevance-guided / sparsity-preserving mutation (fedwrap.fedaware.FedAwareVariation);
  * client-stability tie-break in selection;
  * every candidate evaluated by summing the silos' integer counters into the exact global objective.

The search engine is the real fedwrap.nsga2; only the evaluator is remote (dispatch over SSH to the
silos). Run deploy_shards.sh first.
"""
import argparse
import json
import statistics as st
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

_here = Path(__file__).resolve()
for _cand in [_here.parents[1] / "release", *_here.parents]:
    if (_cand / "fedwrap").is_dir():
        sys.path.insert(0, str(_cand)); break
from fedwrap.nsga2 import nsga2                                       # noqa: E402
from fedwrap.genotypes import BitstringConfig                        # noqa: E402
from fedwrap.fedaware import FedAwareConfig, FedAwareVariation       # noqa: E402
from fedwrap.federated.baselines import ranking_to_mask             # noqa: E402
from fedwrap.federated.metrics import per_label_f1                   # noqa: E402

KEY = str(Path.home() / ".ssh/fedwrap-configb")
SSH = ["ssh", "-i", KEY, "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes",
       "-o", "ConnectTimeout=12", "-o", "ServerAliveInterval=20"]
STATE = _here.parent / "state.tsv"


def silos():
    rows = []
    for ln in STATE.read_text().splitlines():
        if ln.strip():
            tier, region, iid, ip = ln.split("\t")
            rows.append((tier, region, ip))
    return rows


class Worker:
    def __init__(self, tier, ip):
        self.tier, self.ip = tier, ip
        self.p = subprocess.Popen([*SSH, f"ubuntu@{ip}", "cd ~/fed && python3 silo_worker.py"],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, bufsize=1)
        hdr = self.p.stdout.readline().split()
        self.ready = bool(hdr) and hdr[0] == "READY"
        self.D = int(hdr[1]) if self.ready and len(hdr) > 1 else None

    def cmd(self, line):
        self.p.stdin.write(line + "\n"); self.p.stdin.flush()
        return json.loads(self.p.stdout.readline())

    def close(self):
        try:
            self.p.stdin.write("quit\n"); self.p.stdin.flush()
        except Exception:
            pass
        self.p.terminate()


def _macro(tp, fp, fn):
    d = tp + 0.5 * (fp + fn)
    return float(np.divide(tp, d, out=np.zeros(len(tp)), where=d > 0).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=16)
    ap.add_argument("--evals", type=int, default=160)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    ws = [Worker(t, ip) for t, r, ip in silos()]
    for w in ws:
        print(f"  silo {w.tier}: {'READY' if w.ready else 'FAIL'} (D={w.D})")
    if not all(w.ready for w in ws):
        for w in ws:
            w.close()
        print("worker startup failed"); return
    D = ws[0].D

    # ── one-time FEDERATED relevance sketch + filter stats (sufficient statistics only) ──
    print("aggregating federated relevance sketch from the silos...")
    preps = [w.cmd("prep") for w in ws]
    L = len(preps[0]["pos"])
    YtX = sum(np.asarray(p["YtX"], dtype=float) for p in preps)
    sumX = sum(np.asarray(p["sumX"], dtype=float) for p in preps)
    sumX2 = sum(np.asarray(p["sumX2"], dtype=float) for p in preps)
    pos = sum(np.asarray(p["pos"], dtype=float) for p in preps)
    n_tot = max(sum(p["n"] for p in preps), 1.0)
    neg = np.maximum(n_tot - pos, 1.0); pos_safe = np.maximum(pos, 1.0)
    mean = sumX / n_tot; sd = np.sqrt(np.maximum(sumX2 / n_tot - mean ** 2, 1e-9))
    R = np.abs(YtX / pos_safe[:, None] - (sumX[None, :] - YtX) / neg[:, None]) / sd[None, :]
    R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    global_rel = R.mean(axis=0).astype(np.float32)

    lrels = [np.asarray(p["lrel"], dtype=float) for p in preps]
    nvec = [p["n"] for p in preps]
    fed_rank = sum(nv * lr for nv, lr in zip(nvec, lrels))

    def local_topk_union(ratio):
        k = max(1, int(round(ratio * D))); m = np.zeros(D, dtype=bool)
        for lr in lrels:
            m[np.argsort(-lr)[:k]] = True
        return m

    def topk_freq(ratio):
        k = max(1, int(round(ratio * D))); v = np.zeros(D)
        for lr in lrels:
            v[np.argsort(-lr)[:k]] += 1.0
        return v

    mr = 0.25
    seeds = []
    for r in sorted({0.05, 0.10, 0.20, mr}):
        seeds += [ranking_to_mask(fed_rank, r), local_topk_union(r), ranking_to_mask(topk_freq(r), r)]
    seeds = [np.asarray(s, dtype=bool) for s in seeds]

    # ── the real FedAware operators ──
    bcfg = BitstringConfig(init_prob=0.1, bitflip_prob=1.0 / max(1, D))
    facfg = FedAwareConfig(enabled=True, stability_tiebreak=True, disagreement_mutation=True,
                           disagreement_prob=0.5, relevance_pool=20, hardness_temperature=0.5,
                           relevance_warmstart=True, warmstart_frac=0.3, warmstart_jitter=0.10,
                           filter_seed=True, swap_prob=0.4)
    fav = FedAwareVariation(bcfg, facfg, R, global_rel, L)

    def init_pop(rng):
        pop = list(seeds[:a.pop])
        pop += fav.seed_population(max(0, a.pop - len(pop)), rng, max_ratio=mr)
        rng.shuffle(pop)
        return pop[:a.pop]

    cache, n_eval = {}, [0]

    def evaluate(genome):
        mask = np.asarray(genome, dtype=bool)
        if mask.sum() == 0:
            mask = mask.copy(); mask[0] = True
        key = mask.tobytes()
        if key in cache:
            return cache[key]
        idx = ",".join(map(str, np.flatnonzero(mask).tolist()))
        outs = {}
        th = [threading.Thread(target=(lambda w=w: outs.__setitem__(w.tier, w.cmd("eval " + idx))))
              for w in ws]
        for t in th:
            t.start()
        for t in th:
            t.join()
        tp = fp = fn = None
        per_macro = []
        for w in ws:
            c = outs[w.tier]
            t_, f_, n_ = np.asarray(c["tp"]), np.asarray(c["fp"]), np.asarray(c["fn"])
            per_macro.append(_macro(t_, f_, n_))
            tp = t_ if tp is None else tp + t_
            fp = f_ if fp is None else fp + f_
            fn = n_ if fn is None else fn + n_
        macro = _macro(tp, fp, fn)
        n_eval[0] += 1
        meta = {"label_f1": per_label_f1(tp, fp, fn),
                "client_risk": float(st.pstdev(per_macro)) if len(per_macro) > 1 else 0.0}
        res = (np.array([1.0 - macro, float(mask.sum() / D)]), meta)
        cache[key] = res
        return res

    def on_gen(gen, pop):
        lfs = [p.meta["label_f1"] for p in pop if p.meta and p.meta.get("label_f1") is not None]
        if lfs:
            fav.update_hardness(np.mean(np.stack(lfs, axis=0), axis=0))
        return None

    print(f"running REAL FedAware-NSGA-II across {len(ws)} regions "
          f"(pop={a.pop}, budget={a.evals}; R={R.shape}, {len(seeds)} filter seeds)...")
    t0 = time.time()
    pop = nsga2(init_population=init_pop, evaluate=evaluate, variation=fav,
                pop_size=a.pop, max_evals=a.evals, crossover_prob=0.9, mutation_prob=0.3, seed=a.seed,
                on_generation=on_gen,
                tie_breaker=lambda ind: (ind.meta or {}).get("client_risk", 0.0),
                stability_blend=0.15)
    dt = time.time() - t0
    for w in ws:
        w.close()

    best = min(pop, key=lambda ind: ind.objectives[0])           # lowest (1 - macro)
    bmask = np.asarray(best.genome, dtype=bool)
    front = sorted({(int(np.asarray(p.genome).sum()), round(1.0 - float(p.objectives[0]), 3)) for p in pop})
    print("\n==== REAL FedAware-NSGA-II, distributed across 3 AWS regions ====")
    print(f"one-time relevance sketch R                : {R.shape} (aggregated from silo statistics)")
    print(f"evaluations (federated, exact aggregation) : {n_eval[0]}  in  {dt:.0f} s")
    print(f"best subset                                : {int(bmask.sum())}/{D} features")
    print(f"global macro-F1 of best subset             : {1.0 - float(best.objectives[0]):.4f}")
    print(f"Pareto front (n_features, macro-F1)         : {front}")


if __name__ == "__main__":
    main()
