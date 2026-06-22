"""Flower (flwr) backend for the FedWrap-MLFS federated evaluation protocol.

The federated round IS the FedWrap-MLFS evaluation step: the server broadcasts a feature
mask, each client trains a local ML-kNN on the selected features and returns ONLY the
label-wise sufficient statistics (TP/FP/FN), and the server aggregates them into the
exact global micro/macro-F1. Implementing this on Flower (rather than a hand-rolled loop)
gives:

  * a deployable, standards-based federated runtime (real client/server messaging),
  * Flower's simulation engine (Ray) for scalable many-client experiments,
  * drop-in secure-aggregation / differential-privacy strategy wrappers
    (flwr.server.strategy.DifferentialPrivacy*), and
  * reproducible, inspectable communication accounting.

We validate that one Flower round reproduces the exact global F1 of the vectorized
simulator (`FederatedEvaluator`); the large-scale evolutionary inner loop then uses the
equivalent fast simulator, with Flower providing the deployment-realistic protocol,
communication measurements, and the privacy layer.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse

from .client import ClientEvalConfig, FederatedClient
from .metrics import compute_macro_f1, compute_micro_f1


def _make_clients(client_shards, n_labels: int, ccfg: ClientEvalConfig) -> list[FederatedClient]:
    clients = []
    for cid, (xtr, ytr, xva, yva) in enumerate(client_shards):
        clients.append(FederatedClient(cid, xtr.tocsr(), ytr.tocsr(), xva.tocsr(), yva.tocsr(),
                                        n_labels=n_labels, cfg=ccfg))
    return clients


def run_federated_eval_flower(
    client_shards: list[tuple],
    mask: np.ndarray,
    n_labels: int,
    *,
    k: int = 10,
    s: float = 1.0,
    backend: str = "sklearn",
    num_cpus: float = 1.0,
) -> dict[str, Any]:
    """Evaluate one feature mask over virtual clients using a real Flower simulation.

    Returns the aggregated global stats/metrics; numerically identical to the
    FederatedEvaluator (validated in tests). One Flower round = one mask evaluation.
    """
    import flwr as fl
    from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays
    from flwr.server.strategy import Strategy

    ccfg = ClientEvalConfig(kind="mlknn", k=k, s=s, mlknn_backend=backend, mlknn_device="auto")
    clients = _make_clients(client_shards, n_labels, ccfg)
    mask_arr = np.asarray(mask, dtype=np.int8)

    class _StatsClient(fl.client.NumPyClient):
        def __init__(self, fc: FederatedClient):
            self.fc = fc

        def fit(self, parameters, config):
            m = parameters[0].astype(bool)
            r = self.fc.evaluate_mask(m, mode="full")
            stats = np.stack([r["tp"], r["fp"], r["fn"]]).astype(np.float64)  # (3, L)
            return [stats], int(r["n_val"]), {}

    def client_fn(context) -> fl.client.Client:
        cid = int(context.node_config["partition-id"])
        return _StatsClient(clients[cid]).to_client()

    agg: dict[str, np.ndarray] = {}

    class _StatsStrategy(Strategy):
        def initialize_parameters(self, client_manager):
            return ndarrays_to_parameters([mask_arr])

        def configure_fit(self, server_round, parameters, client_manager):
            cfg = {}
            from flwr.common import FitIns
            clients_sel = client_manager.sample(num_clients=len(clients), min_num_clients=len(clients))
            return [(c, FitIns(parameters, cfg)) for c in clients_sel]

        def aggregate_fit(self, server_round, results, failures):
            tp = np.zeros(n_labels); fp = np.zeros(n_labels); fn = np.zeros(n_labels); nval = 0
            for _client, fit_res in results:
                st = parameters_to_ndarrays(fit_res.parameters)[0]
                tp += st[0]; fp += st[1]; fn += st[2]; nval += int(fit_res.num_examples)
            agg["tp"], agg["fp"], agg["fn"], agg["n_val"] = tp, fp, fn, nval
            return None, {}

        def configure_evaluate(self, server_round, parameters, client_manager):
            return []

        def aggregate_evaluate(self, server_round, results, failures):
            return None, {}

        def evaluate(self, server_round, parameters):
            return None

    def server_fn(context):
        from flwr.server import ServerAppComponents, ServerConfig
        return ServerAppComponents(strategy=_StatsStrategy(), config=ServerConfig(num_rounds=1))

    fl.simulation.run_simulation(
        server_app=fl.server.ServerApp(server_fn=server_fn),
        client_app=fl.client.ClientApp(client_fn=client_fn),
        num_supernodes=len(clients),
        backend_config={"client_resources": {"num_cpus": num_cpus, "num_gpus": 0.0}},
    )

    tp, fp, fn = agg["tp"], agg["fp"], agg["fn"]
    return {
        "tp": tp, "fp": fp, "fn": fn, "n_val": agg["n_val"],
        "micro_f1": compute_micro_f1(tp, fp, fn),
        "macro_f1": compute_macro_f1(tp, fp, fn),
        "clients": len(clients),
    }
