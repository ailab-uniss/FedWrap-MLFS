"""Analytical communication-cost estimator for the simulated federation.

For a single-workstation simulation the byte counts are estimated analytically
(research plan, section 15.4):

- server -> client : the feature mask. Either ``n_features`` bits (dense bitmask)
  or a list of selected indices (4 bytes each), whichever is smaller.
- client -> server : 3 * n_labels counters (4 bytes each, TP/FP/FN).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CommunicationEstimator:
    n_features: int
    n_labels: int
    counter_bytes: int = 4
    index_bytes: int = 4

    messages: int = 0
    masks_sent: int = 0
    stat_vectors_received: int = 0
    bytes_down: int = 0  # server -> client
    bytes_up: int = 0    # client -> server
    _per_solution: list = field(default_factory=list, repr=False)

    def mask_bytes(self, n_selected: int) -> int:
        dense = int(np.ceil(self.n_features / 8.0))
        sparse_b = int(n_selected) * self.index_bytes
        return min(dense, sparse_b)

    def stats_bytes(self) -> int:
        return 3 * int(self.n_labels) * self.counter_bytes

    def record_round(self, n_selected: int, n_clients: int) -> None:
        """Record one evaluation round: one mask broadcast, n_clients replies."""
        down = self.mask_bytes(n_selected) * n_clients
        up = self.stats_bytes() * n_clients
        self.bytes_down += down
        self.bytes_up += up
        self.messages += 2 * n_clients
        self.masks_sent += n_clients
        self.stat_vectors_received += n_clients

    def summary(self) -> dict:
        total = self.bytes_down + self.bytes_up
        return {
            "messages": self.messages,
            "masks_sent": self.masks_sent,
            "stat_vectors_received": self.stat_vectors_received,
            "bytes_server_to_client": self.bytes_down,
            "bytes_client_to_server": self.bytes_up,
            "communication_bytes_estimated": total,
        }
