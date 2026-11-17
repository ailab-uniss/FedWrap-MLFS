"""Revision experiment (Reviewer 2, comments 1-2): a complete FedWrap-MLFS search on the Flower/Ray runtime.

The submitted manuscript validated the Flower backend on one mask and ran the searches with the
in-process evaluator. Here the complete workflow (S0 relevance sketch and filter seeds, S2-S3 mask
evaluation and count aggregation, S6 final re-evaluation) is executed through Flower messages on the
Ray simulation backend, and the result is compared, mask for mask, with the in-process run obtained
with the same configuration and seed.

Each Flower client loads its own shard from disk and answers two query types:
    prep  -> the five additive statistics of the relevance sketch and its local ANOVA relevance vector
    eval  -> label-wise TP/FP/FN (and n_val) for every mask contained in the message
After shard export the server process keeps no client data (clients are replaced by shape-only stubs),
so every data-dependent quantity used by the search arrives in a Flower reply.

Round mapping (``mode``):
    mask        one mask per round (one request-response exchange per candidate)
    generation  all not-yet-evaluated masks of a generation in one round

Usage: python scripts/revision/run_flower_search.py <dataset> <method> <seed> <mode> [reference_run_dir]
"""
from __future__ import annotations

import copy
import functools
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse

sys.path.insert(0, ".")

DS = sys.argv[1]
METHOD = sys.argv[2]
SEED = int(sys.argv[3])
MODE = sys.argv[4] if len(sys.argv) > 4 else "generation"
REF = sys.argv[5] if len(sys.argv) > 5 else None
RUN_TAG = f"{DS}_{METHOD}_s{SEED}_{MODE}"
OUT = f"runs/revision/flower/{RUN_TAG}"
SHARDS = Path(f"runs/revision/flower/_shards/{RUN_TAG}").resolve()
REPO = str(Path(".").resolve())

# ───────────────────────────── client side (runs inside Ray actors) ─────────────────────────────
_CLIENT_CACHE: dict = {}


def _load_client(shard_dir: str, tag: str, cid: int):
    key = (shard_dir, tag, cid)
    if key not in _CLIENT_CACHE:
        if REPO not in sys.path:
            sys.path.insert(0, REPO)
        from fedwrap.federated.client import ClientEvalConfig, FederatedClient
        z = np.load(Path(shard_dir) / f"{tag}_client{cid}.npz", allow_pickle=False)

        def csr(p):
            return sparse.csr_matrix((z[p + "_data"], z[p + "_indices"], z[p + "_indptr"]), shape=tuple(z[p + "_shape"]))
        cfg = ClientEvalConfig(kind="mlknn", k=5, s=1.0, mlknn_backend="sklearn", mlknn_device="cpu", cv_folds=1)
        _CLIENT_CACHE[key] = FederatedClient(cid, csr("xtr"), csr("ytr"), csr("xva"), csr("yva"),
                                             n_labels=int(z["n_labels"]), cfg=cfg)
    return _CLIENT_CACHE[key]


def make_client_app():
    from flwr.client import ClientApp
    from flwr.common import ArrayRecord, ConfigRecord, Context, Message, RecordDict

    app = ClientApp()

    @app.query()
    def query(msg: Message, context: Context) -> Message:
        cfg = msg.content["cfg"]
        cid = int(context.node_config["partition-id"])
        op = str(cfg["op"])
        if op == "hello":
            return Message(RecordDict({"cfg": ConfigRecord({"cid": cid})}), reply_to=msg)
        client = _load_client(str(cfg["shards"]), str(cfg["tag"]), cid)
        if op == "prep":
            from fedwrap.federated.baselines import _local_relevance
            X = client.x_train.tocsr(); Y = client.y_train.tocsr()
            arrs = [np.asarray((Y.T @ X).todense(), dtype=np.float64),
                    np.asarray(X.sum(axis=0), dtype=np.float64).ravel(),
                    np.asarray(X.multiply(X).sum(axis=0), dtype=np.float64).ravel(),
                    np.asarray(Y.sum(axis=0), dtype=np.float64).ravel(),
                    np.asarray([X.shape[0]], dtype=np.float64),
                    np.asarray(_local_relevance(client.x_train, client.y_train), dtype=np.float64)]
            return Message(RecordDict({"arr": ArrayRecord(numpy_ndarrays=arrs),
                                       "cfg": ConfigRecord({"cid": cid})}), reply_to=msg)
        # op == "eval": a (B, D) batch of masks -> (B, 3, L) counters
        masks = msg.content["arr"].to_numpy_ndarrays()[0].astype(bool)
        out = np.zeros((masks.shape[0], 3, client.n_labels), dtype=np.int64)
        n_val = 0
        for b in range(masks.shape[0]):
            r = client.evaluate_mask(masks[b], mode="full")
            out[b, 0], out[b, 1], out[b, 2] = r["tp"], r["fp"], r["fn"]
            n_val = int(r["n_val"])
        return Message(RecordDict({"arr": ArrayRecord(numpy_ndarrays=[out]),
                                   "cfg": ConfigRecord({"cid": cid, "n_val": n_val})}), reply_to=msg)

    return app


# ───────────────────────────────────── server side ─────────────────────────────────────
STATS = {"rounds": 0, "masks": 0, "bytes_down": 0, "bytes_up": 0, "prep_rounds": 0, "t_rounds": 0.0}


class _Stub:
    """Shape-only stand-in for a client on the server: no feature or label values."""

    def __init__(self, cid, n_train, n_val, D, L):
        self.client_id = cid
        self.x_train = sparse.csr_matrix((n_train, D), dtype=np.float32)
        self.y_train = sparse.csr_matrix((n_train, L), dtype=np.int8)
        self.x_val = sparse.csr_matrix((n_val, D), dtype=np.float32)
        self.y_val = sparse.csr_matrix((n_val, L), dtype=np.int8)
        self.n_train, self.n_val = int(n_train), int(n_val)


def _save_shard(path: Path, c, L: int):
    d = {"n_labels": np.int64(L)}
    for name, m in (("xtr", c.x_train), ("ytr", c.y_train), ("xva", c.x_val), ("yva", c.y_val)):
        m = m.tocsr()
        d[name + "_data"], d[name + "_indices"], d[name + "_indptr"] = m.data, m.indices, m.indptr
        d[name + "_shape"] = np.asarray(m.shape, dtype=np.int64)
    np.savez(path, **d)


def make_server_app(cfg: dict):
    from flwr.common import ArrayRecord, ConfigRecord, Context, Message, RecordDict
    from flwr.server import Grid, ServerApp

    app = ServerApp()

    @app.main()
    def main(grid: Grid, context: Context) -> None:
        import fedwrap.experiment as E
        import fedwrap.federated.baselines as B
        from fedwrap.federated.evaluator import FederatedEvaluator
        from fedwrap.nsga2 import nsga2 as _nsga2

        node_ids = list(grid.get_node_ids())

        def exchange(records: dict[int, RecordDict]) -> dict[int, Message]:
            t0 = time.perf_counter()
            msgs = [Message(rd, nid, "query") for nid, rd in records.items()]
            replies = list(grid.send_and_receive(msgs))
            STATS["t_rounds"] += time.perf_counter() - t0
            out = {}
            for r in replies:
                if r.has_error():
                    raise RuntimeError(f"client error: {r.error}")
                out[int(r.content["cfg"]["cid"])] = r
            return out

        hello = [Message(RecordDict({"cfg": ConfigRecord({"op": "hello"})}), nid, "query") for nid in node_ids]
        cid_of = {}
        for r in grid.send_and_receive(hello):
            cid_of[int(r.content["cfg"]["cid"])] = int(r.metadata.src_node_id)
        K = len(cid_of)

        class FlowerEvaluator(FederatedEvaluator):
            """Same server-side aggregation as FederatedEvaluator; client work arrives via Flower."""

            def _export(self, tag: str):
                self.tag = tag
                SHARDS.mkdir(parents=True, exist_ok=True)
                self.cids = [c.client_id for c in self.clients]
                stubs = []
                for c in self.clients:
                    _save_shard(SHARDS / f"{tag}_client{c.client_id}.npz", c, self.n_labels)
                    stubs.append(_Stub(c.client_id, c.n_train, c.n_val, self.n_features, self.n_labels))
                self.clients = stubs          # the server keeps shapes only
                if self._pool is not None:
                    self._pool.shutdown(wait=False); self._pool = None

            def _remote_counts(self, masks: np.ndarray) -> list[list[dict]]:
                """One Flower round: send the (B, D) batch to every client; return per-mask client results."""
                arr = masks.astype(np.uint8)
                recs = {cid_of[c]: RecordDict({"arr": ArrayRecord(numpy_ndarrays=[arr]),
                                               "cfg": ConfigRecord({"op": "eval", "tag": self.tag, "shards": str(SHARDS)})})
                        for c in self.cids}
                rep = exchange(recs)
                STATS["rounds"] += 1; STATS["masks"] += int(masks.shape[0])
                per_client = {}
                for c in self.cids:
                    a = rep[c].content["arr"]
                    STATS["bytes_up"] += int(a.count_bytes())
                    STATS["bytes_down"] += int(arr.nbytes)
                    per_client[c] = (a.to_numpy_ndarrays()[0], int(rep[c].content["cfg"]["n_val"]))
                out = []
                for b in range(masks.shape[0]):
                    out.append([{"client_id": c, "tp": per_client[c][0][b, 0], "fp": per_client[c][0][b, 1],
                                 "fn": per_client[c][0][b, 2], "n_val": per_client[c][1]} for c in self.cids])
                return out

            def _store(self, mask: np.ndarray, local_results: list[dict]):
                stats = self.aggregator.aggregate(local_results)
                fr = float(mask.sum() / self.n_features)
                res = (self._build_objectives(stats, fr), self._build_mlresult(stats))
                if self.fed.estimate_communication:
                    self.comm.record_round(int(mask.sum()), len(self.cids))
                self.counters["full_evals"] += 1
                key = mask.tobytes()
                self._cache[key] = res; self._stats_cache[key] = stats
                return res

            def evaluate_mask(self, feature_mask, mode="full", client_fraction=None):
                mask = np.asarray(feature_mask, dtype=bool)
                key = mask.tobytes()
                if key in self._cache:
                    self.counters["full_cache_hits"] += 1
                    return self._cache[key]
                return self._store(mask, self._remote_counts(mask[None, :])[0])

            def prefetch(self, masks: list[np.ndarray]):
                todo, seen = [], set()
                for m in masks:
                    m = np.asarray(m, dtype=bool); k = m.tobytes()
                    if k not in self._cache and k not in seen:
                        seen.add(k); todo.append(m)
                if todo:
                    for m, lr in zip(todo, self._remote_counts(np.stack(todo))):
                        self._store(m, lr)

        evaluators: list[FlowerEvaluator] = []
        lrel_by_id: dict[int, np.ndarray] = {}

        def make_evaluator(x_tr, y_tr, x_va, y_va, eval_cfg, config, seed, groups=None):
            from fedwrap.federated import federated_config_from_dict
            fed_cfg = federated_config_from_dict(config, seed=seed)
            if groups is not None:
                fed_cfg.groups_train, fed_cfg.groups_val = groups
            ev = FlowerEvaluator(x_tr, y_tr, x_va, y_va, eval_cfg, fed_cfg)
            ev._export("search" if not evaluators else "test")
            evaluators.append(ev)
            return ev

        def federated_relevance(clients, n_features, n_labels, hard_client_weight=None):
            """S0 through Flower: sum the five additive statistics returned by the clients."""
            ev = evaluators[0]
            rep = exchange({cid_of[c]: RecordDict({"cfg": ConfigRecord({"op": "prep", "tag": ev.tag, "shards": str(SHARDS)})})
                            for c in ev.cids})
            STATS["prep_rounds"] += 1
            YtX = np.zeros((n_labels, n_features)); sumX = np.zeros(n_features); sumX2 = np.zeros(n_features)
            pos = np.zeros(n_labels); n_tot = 0.0
            for stub in ev.clients:                      # same accumulation order as the in-process code
                a = rep[stub.client_id].content["arr"].to_numpy_ndarrays()
                YtX += a[0]; sumX += a[1]; sumX2 += a[2]; pos += a[3]; n_tot += float(a[4][0])
                lrel_by_id[id(stub.x_train)] = a[5]
            n_tot = max(n_tot, 1.0)
            neg = np.maximum(n_tot - pos, 1.0); pos_safe = np.maximum(pos, 1.0)
            mean = sumX / n_tot; sd = np.sqrt(np.maximum(sumX2 / n_tot - mean ** 2, 1e-9))
            R = np.abs(YtX / pos_safe[:, None] - (sumX[None, :] - YtX) / neg[:, None]) / sd[None, :]
            R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
            return R.astype(np.float32), R.mean(axis=0).astype(np.float32)

        def to_mask(g):
            m = np.asarray(g, dtype=bool)
            if m.sum() == 0:
                m = m.copy(); m[0] = True
            return m

        def prefetch(genomes):
            evaluators[0].prefetch([to_mask(g) for g in genomes])

        E.make_evaluator = make_evaluator
        E.federated_relevance = federated_relevance
        B._local_relevance = lambda x, y: lrel_by_id[id(x)]       # filter seeds use the uploaded vectors
        if MODE == "generation":
            E.nsga2 = functools.partial(_nsga2, prefetch=prefetch)

        t0 = time.perf_counter()
        E.run_experiment_from_config(cfg, fold_idx=0)
        STATS["wall_s"] = time.perf_counter() - t0
        STATS["clients"] = K
        STATS["search_full_evals"] = int(evaluators[0].counters["full_evals"])

    return app


def build_cfg() -> dict:
    """Identical to scripts/run_reals_method.py (methods 'faport' and 'bitstring')."""
    import yaml
    base = yaml.safe_load(open("configs/main_bench.yaml"))
    cfg = copy.deepcopy(base)

    def setk(d, dot, v):
        p = dot.split("."); c = d
        for k in p[:-1]:
            c = c.setdefault(k, {})
        c[p[-1]] = v
    nclients = len(set(np.load(f"data/fed_real/{DS}/fold0/trainval_groups.npy", allow_pickle=True).tolist()))
    setk(cfg, "seed", SEED)
    setk(cfg, "dataset.kind", "prefold"); setk(cfg, "dataset.root", "data/fed_real"); setk(cfg, "dataset.name", DS)
    setk(cfg, "cross_validation.enabled", True); setk(cfg, "cross_validation.n_folds", 1)
    setk(cfg, "model.kind", "mlknn"); setk(cfg, "model.mlknn_backend", "sklearn"); setk(cfg, "model.mlknn_device", "cpu")
    setk(cfg, "model.k", 5); setk(cfg, "model.cv_folds", 1)
    setk(cfg, "evolution.genotype", "bitstring")
    cfg.get("evolution", {}).pop("max_evals_per_feature", None)
    setk(cfg, "evolution.max_evals", 300000)
    setk(cfg, "evolution.early_stopping", {"enabled": True, "mode": "window", "window": 10, "rel_tol": 0.002, "patience": 3})
    is_port = METHOD == "faport"
    setk(cfg, "fedaware.enabled", is_port)
    setk(cfg, "fedaware.stability_tiebreak", True); setk(cfg, "fedaware.disagreement_mutation", True)
    setk(cfg, "fedaware.relevance_warmstart", is_port)
    if is_port:
        setk(cfg, "fedaware.warmstart_frac", 0.3); setk(cfg, "fedaware.filter_seed", True); setk(cfg, "fedaware.swap_prob", 0.4)
    setk(cfg, "federated", {"enabled": True, "n_clients": nclients, "partition": "natural_silo",
                            "min_samples_per_client": 32, "client_fraction_full": 1.0, "final_eval_all_clients": True,
                            "return_client_metrics": True, "estimate_communication": True, "n_jobs": 1})
    setk(cfg, "reporting.max_feature_ratio", 0.25)
    setk(cfg, "logging.out_dir", OUT)
    return cfg, nclients


def main():
    import flwr as fl
    cfg, K = build_cfg()
    fl.simulation.run_simulation(server_app=make_server_app(cfg), client_app=make_client_app(),
                                 num_supernodes=K,
                                 backend_config={"client_resources": {"num_cpus": 1.0, "num_gpus": 0.0}})
    res = {"dataset": DS, "method": METHOD, "seed": SEED, "mode": MODE, **STATS}
    out = Path(OUT + "_fold0")
    if REF is not None:
        a = np.load(out / "population_masks.npz"); b = np.load(Path(REF) / "population_masks.npz")
        res["reference"] = REF
        res["masks_identical"] = bool(np.array_equal(a["masks_packed"], b["masks_packed"]))
        res["val_objs_max_abs_diff"] = float(np.max(np.abs(a["val_objs"] - b["val_objs"]))) if a["val_objs"].shape == b["val_objs"].shape else None
        res["test_objs_max_abs_diff"] = float(np.max(np.abs(a["test_objs"] - b["test_objs"]))) if a["test_objs"].shape == b["test_objs"].shape else None
        sa = json.loads((out / "summary.json").read_text()); sb = json.loads((Path(REF) / "summary.json").read_text())
        res["generations"] = sa["final"]["gen"]; res["generations_ref"] = sb["final"]["gen"]
        res["total_evals"] = sa["final"]["total_evals"]; res["total_evals_ref"] = sb["final"]["total_evals"]
        res["hv_val"] = sa["final"]["hv_val"]; res["hv_val_ref"] = sb["final"]["hv_val"]
    Path("reports/revision").mkdir(parents=True, exist_ok=True)
    (Path("reports/revision") / f"flower_search_{RUN_TAG}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2), flush=True)


if __name__ == "__main__":
    main()
