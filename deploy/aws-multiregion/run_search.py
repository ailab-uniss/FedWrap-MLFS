#!/usr/bin/env python3
"""Real federated NSGA-II wrapper feature selection executed ACROSS the 3 multi-region AWS silos.

Every candidate subset the search proposes is evaluated by dispatching it to the silos (persistent
workers) in parallel, summing their label-wise TP/FP/FN counters into the EXACT global objective
(Prop. 1), and scoring $(1-\\text{macro-F1},\\ \\text{feature ratio})$. The real search engine
(``fedwrap.nsga2``) drives selection with the bit-string operators (``fedwrap.genotypes``). So the
whole wrapper---search + federated evaluation + exact aggregation---runs on real geo-distributed
infrastructure, not in the in-process simulator. Run deploy_shards.sh first.
"""
import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

# fedwrap on the orchestrator: repo root is the first parent containing a fedwrap/ package.
_here = Path(__file__).resolve()
for cand in [_here.parents[1] / "release", *_here.parents]:
    if (cand / "fedwrap").is_dir():
        sys.path.insert(0, str(cand)); break
from fedwrap.nsga2 import nsga2                                            # noqa: E402
from fedwrap.genotypes import (BitstringConfig, bitstring_crossover,      # noqa: E402
                               bitstring_mutate, init_bitstring)

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

    def evaluate(self, idx_str, out):
        self.p.stdin.write("eval " + idx_str + "\n"); self.p.stdin.flush()
        line = self.p.stdout.readline()
        out[self.tier] = json.loads(line) if line.strip().startswith("{") else None

    def close(self):
        try:
            self.p.stdin.write("quit\n"); self.p.stdin.flush()
        except Exception:
            pass
        self.p.terminate()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=16)
    ap.add_argument("--evals", type=int, default=160)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    sl = silos()
    ws = [Worker(t, ip) for t, r, ip in sl]
    for w in ws:
        print(f"  silo {w.tier}: {'READY' if w.ready else 'FAIL'} (D={w.D})")
    if not all(w.ready for w in ws):
        for w in ws:
            w.close()
        print("worker startup failed"); return
    D = ws[0].D
    cache, n_eval = {}, [0]

    def evaluate(genome):
        mask = np.asarray(genome, dtype=bool)
        key = mask.tobytes()
        if key in cache:
            return cache[key]
        idx_str = ",".join(map(str, np.flatnonzero(mask).tolist()))
        out = {}
        th = [threading.Thread(target=w.evaluate, args=(idx_str, out)) for w in ws]
        for t in th:
            t.start()
        for t in th:
            t.join()
        tp = fp = fn = None
        for w in ws:
            c = out[w.tier]
            t_, f_, n_ = np.asarray(c["tp"]), np.asarray(c["fp"]), np.asarray(c["fn"])
            tp = t_ if tp is None else tp + t_
            fp = f_ if fp is None else fp + f_
            fn = n_ if fn is None else fn + n_
        d = tp + 0.5 * (fp + fn)
        macro = float(np.divide(tp, d, out=np.zeros(len(tp)), where=d > 0).mean())
        n_eval[0] += 1
        res = (np.array([1.0 - macro, float(mask.sum() / D)]), {"macro": macro})
        cache[key] = res
        return res

    cfg = BitstringConfig(init_prob=0.12, bitflip_prob=0.02)

    class Var:
        def crossover(self, x, y, rng):
            return bitstring_crossover(x, y, rng)

        def mutate(self, x, rng):
            return bitstring_mutate(x, cfg, rng)

        def repair(self, x, rng):
            return x

    def init_pop(rng):
        return [init_bitstring(D, cfg, rng) for _ in range(a.pop)]

    print(f"running federated NSGA-II across {len(ws)} regions (pop={a.pop}, budget={a.evals} evals)...")
    t0 = time.time()
    pop = nsga2(init_population=init_pop, evaluate=evaluate, variation=Var(),
                pop_size=a.pop, max_evals=a.evals, crossover_prob=0.9, mutation_prob=0.3, seed=a.seed)
    dt = time.time() - t0
    for w in ws:
        w.close()

    best = max(pop, key=lambda ind: ind.meta["macro"])
    bmask = np.asarray(best.genome, dtype=bool)
    front = sorted({(int(np.asarray(p.genome).sum()), round(p.meta["macro"], 3)) for p in pop})
    print("\n==== distributed federated NSGA-II wrapper search (3 AWS regions) ====")
    print(f"evaluations (federated, exact aggregation) : {n_eval[0]}  in  {dt:.0f} s")
    print(f"best subset                                : {int(bmask.sum())}/{D} features")
    print(f"global macro-F1 of best subset             : {best.meta['macro']:.4f}")
    print(f"Pareto front (n_features, macro-F1)         : {front}")


if __name__ == "__main__":
    main()
