"""Compact, portable I/O for one client's data shard.

A shard bundles a client's local train/val split (the data that, in deployment, never leaves the
client) into a single ``.npz`` file so it can be passed as one CWL ``File`` input. CSR matrices are
stored by their components; ``load_shard`` reconstructs them. Kept dependency-light (numpy + scipy)
so the client tool image stays small.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse


def _pack(prefix: str, m: sparse.csr_matrix, out: dict) -> None:
    m = m.tocsr()
    out[f"{prefix}_data"] = m.data
    out[f"{prefix}_indices"] = m.indices
    out[f"{prefix}_indptr"] = m.indptr
    out[f"{prefix}_shape"] = np.asarray(m.shape, dtype=np.int64)


def _unpack(prefix: str, z) -> sparse.csr_matrix:
    return sparse.csr_matrix((z[f"{prefix}_data"], z[f"{prefix}_indices"], z[f"{prefix}_indptr"]),
                             shape=tuple(z[f"{prefix}_shape"]))


def save_shard(path: str, x_train, y_train, x_val, y_val) -> None:
    out: dict = {}
    _pack("xtr", x_train, out); _pack("ytr", y_train, out)
    _pack("xva", x_val, out); _pack("yva", y_val, out)
    np.savez_compressed(path, **out)


def load_shard(path: str):
    z = np.load(path)
    return _unpack("xtr", z), _unpack("ytr", z), _unpack("xva", z), _unpack("yva", z)
