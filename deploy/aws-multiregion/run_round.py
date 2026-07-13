#!/usr/bin/env python3
"""Timed federated round across the 3 multi-region AWS silos, via a PERSISTENT worker per silo.

Each silo runs silo_worker.py (loads shard + mask once, like a resident Flower client). Each round the
orchestrator sends ``eval`` to all silos in parallel over a kept-open SSH pipe and times each reply, so
the round reflects network round-trip + inference (not process/import startup). We then report the
resource-aware SCHEDULER speed-up:
  * full participation  -> wait for the slowest silo (the far edge in Tokyo);
  * resource-aware quorum=K' -> proceed once the fastest K' answer (drop the WAN straggler).
Aggregating the responders' integer counters is EXACT for that subset (Prop. 1), so the quorum result
is a valid global F1 at a fraction of the round time. Real WAN latency, measured. Run deploy_shards.sh
and push silo_worker.py first (deploy_worker below).
"""
import argparse
import json
import statistics as st
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

KEY = str(Path.home() / ".ssh/fedwrap-configb")
SSH = ["ssh", "-i", KEY, "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes",
       "-o", "ConnectTimeout=12", "-o", "ServerAliveInterval=20"]


def silos():
    rows = []
    for ln in (Path(__file__).resolve().parent / "state.tsv").read_text().splitlines():
        if ln.strip():
            tier, region, iid, ip = ln.split("\t")
            rows.append((tier, region, ip))
    return rows


class Worker:
    def __init__(self, tier, region, ip):
        self.tier, self.region, self.ip = tier, region, ip
        self.p = subprocess.Popen([*SSH, f"ubuntu@{ip}", "cd ~/fed && python3 silo_worker.py"],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.ready = (self.p.stdout.readline().strip() == "READY")

    def eval_timed(self, out):
        t = time.time()
        line = self.p.stdout.readline() if self._send() else ""
        out[self.tier] = {"dt": time.time() - t,
                          "cnt": json.loads(line) if line.strip().startswith("{") else None}

    def _send(self):
        try:
            self.p.stdin.write("eval\n"); self.p.stdin.flush(); return True
        except Exception:
            return False

    def close(self):
        try:
            self.p.stdin.write("quit\n"); self.p.stdin.flush()
        except Exception:
            pass
        self.p.terminate()


def a_round(workers):
    out = {}
    th = [threading.Thread(target=w.eval_timed, args=(out,)) for w in workers]
    for t in th:
        t.start()
    for t in th:
        t.join()
    return out


def macro_f1(counters):
    tp = fp = fn = None
    for c in counters:
        t, f, n = np.asarray(c["tp"]), np.asarray(c["fp"]), np.asarray(c["fn"])
        tp = t if tp is None else tp + t
        fp = f if fp is None else fp + f
        fn = n if fn is None else fn + n
    d = tp + 0.5 * (fp + fn)
    return float(np.divide(tp, d, out=np.zeros(len(tp)), where=d > 0).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--quorum", type=int, default=2)
    a = ap.parse_args()
    sl = silos()
    print("connecting persistent workers...")
    ws = [Worker(t, r, ip) for t, r, ip in sl]
    for w in ws:
        print(f"  {w.tier} ({w.region} {w.ip}): {'READY' if w.ready else 'FAILED'}")
    if not all(w.ready for w in ws):
        for w in ws:
            w.close()
        print("some worker failed to start"); return

    a_round(ws)   # warm-up (excluded)
    full_t, quo_t, last = [], [], None
    for rnd in range(a.rounds):
        res = a_round(ws)
        order = sorted((res[w.tier]["dt"], w.tier) for w in ws if res[w.tier]["cnt"] is not None)
        if len(order) < len(ws):
            print(f"round {rnd}: a silo failed to reply"); continue
        full, quo = order[-1][0], order[a.quorum - 1][0]
        full_t.append(full); quo_t.append(quo); last = res
        line = "  <  ".join(f"{t} {d * 1000:.0f}ms" for d, t in order)
        print(f"round {rnd}: {line}   | full={full * 1000:.0f}ms  quorum{a.quorum}={quo * 1000:.0f}ms")

    for w in ws:
        w.close()
    if not full_t:
        print("no successful rounds"); return
    F, Q = st.mean(full_t) * 1000, st.mean(quo_t) * 1000
    order = sorted((last[w.tier]["dt"], w.tier) for w in ws)
    allc = [last[t]["cnt"] for _, t in order]
    quoc = [last[t]["cnt"] for _, t in order[:a.quorum]]
    print("\n==== resource-aware scheduling under REAL WAN latency "
          f"({len(full_t)} rounds, persistent clients) ====")
    print(f"full participation (wait for the far edge)    : {F:6.0f} ms/round")
    print(f"resource-aware quorum={a.quorum} (drop the straggler)  : {Q:6.0f} ms/round")
    print(f"critical-path SPEED-UP                         : {F / Q:5.1f}x")
    print(f"global macro-F1  --  all {len(allc)} silos         : {macro_f1(allc):.4f}")
    print(f"global macro-F1  --  quorum {a.quorum} responders     : {macro_f1(quoc):.4f}  "
          "(exact over the responders, Prop. 1)")


if __name__ == "__main__":
    main()
