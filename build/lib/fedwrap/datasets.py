"""Dataset loading for the pre-folded benchmark protocol.

The paper uses ``dataset.kind = "prefold"`` exclusively.  Each dataset
lives in a directory tree::

    data/dense_benchmark_v3/<Name>/fold0/trainval.npz
                                   fold0/test.npz
                                   ...
                                   fold4/trainval.npz
                                   fold4/test.npz

where each ``.npz`` file may contain either dense arrays ``X`` / ``Y``
or sparse CSR arrays ``X_data`` / ``X_indices`` / ``X_indptr`` / ``X_shape``
plus the corresponding ``Y_*`` keys.

The trainval split is further divided into train / val using
scikit-multilearn's :class:`IterativeStratification` to ensure
approximate label-distribution balance.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

try:
    from skmultilearn.model_selection import IterativeStratification
    SKMULTILEARN_AVAILABLE = True
except ImportError:
    SKMULTILEARN_AVAILABLE = False


# ── Data types ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DatasetSplit:
    """Train / validation / test matrices for one fold."""
    x_train: sparse.csr_matrix
    y_train: sparse.csr_matrix
    x_val: sparse.csr_matrix
    y_val: sparse.csr_matrix
    x_test: sparse.csr_matrix
    y_test: sparse.csr_matrix


# ── Helpers ───────────────────────────────────────────────────────────

def _as_csr(x: Any) -> sparse.csr_matrix:
    if sparse.issparse(x):
        return x.tocsr()
    return sparse.csr_matrix(np.asarray(x))


@contextlib.contextmanager
def _temporary_numpy_seed(seed: int):
    """Temporarily set NumPy's global RNG seed.

    Needed because skmultilearn's IterativeStratification relies on the
    global RNG in some versions.
    """
    state = np.random.get_state()
    np.random.seed(int(seed))
    try:
        yield
    finally:
        np.random.set_state(state)


def _load_npz_dense(path: Path) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Load an ``.npz`` with dense arrays ``X`` and ``Y``."""
    data = np.load(path, allow_pickle=True)
    x = _as_csr(np.asarray(data["X"], dtype=np.float32))
    y_key = "Y" if "Y" in data else "y"
    y = _as_csr((np.asarray(data[y_key]) > 0).astype(np.int8))
    return x, y


def _load_npz_any(path: Path) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Load either dense X/Y npz or sparse CSR-pair npz."""
    data = np.load(path, allow_pickle=True)
    if "X_data" in data.files:
        x = sparse.csr_matrix(
            (data["X_data"], data["X_indices"], data["X_indptr"]),
            shape=tuple(data["X_shape"]),
        )
        y = sparse.csr_matrix(
            (data["Y_data"], data["Y_indices"], data["Y_indptr"]),
            shape=tuple(data["Y_shape"]),
        )
        x.sum_duplicates()
        y.sum_duplicates()
        x.data = x.data.astype(np.float32, copy=False)
        y.data = (y.data > 0).astype(np.int8, copy=False)
        return x, y
    x = _as_csr(np.asarray(data["X"], dtype=np.float32))
    y_key = "Y" if "Y" in data else "y"
    y = _as_csr((np.asarray(data[y_key]) > 0).astype(np.int8))
    return x, y


def _stratified_split(
    x: sparse.csr_matrix,
    y: sparse.csr_matrix,
    test_size: float,
    seed: int,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix,
           sparse.csr_matrix, sparse.csr_matrix]:
    """Iterative-stratification split of *x*, *y* into two subsets."""
    if not SKMULTILEARN_AVAILABLE:
        raise ImportError(
            "scikit-multilearn is required for stratified splits.  "
            "Install with: pip install scikit-multilearn"
        )
    try:
        stratifier = IterativeStratification(
            n_splits=2, order=1,
            sample_distribution_per_fold=[1.0 - float(test_size), float(test_size)],
            random_state=int(seed),
        )
    except (TypeError, ValueError):
        stratifier = IterativeStratification(
            n_splits=2, order=1,
            sample_distribution_per_fold=[1.0 - float(test_size), float(test_size)],
        )
    desired = int(round(float(test_size) * float(x.shape[0])))
    best = None
    best_delta = None
    with _temporary_numpy_seed(int(seed)):
        for tr_idx, te_idx in stratifier.split(x, y):
            delta = abs(int(te_idx.size) - desired)
            if best is None or best_delta is None or delta < best_delta:
                best = (tr_idx, te_idx)
                best_delta = delta
    if best is None:
        raise RuntimeError("IterativeStratification produced no splits.")
    train_idx, test_idx = best
    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]


# ── Public API ────────────────────────────────────────────────────────

def load_dataset(
    config: dict[str, Any],
    seed: int,
    fold_idx: int | None = None,
) -> DatasetSplit:
    """Load a dataset according to the configuration.

    Only ``kind = "prefold"`` is supported in this bundle.
    """
    dataset_cfg = config.get("dataset", {})
    kind = dataset_cfg.get("kind", "prefold")

    # npz_split: high-feature suites stored as <root>/<name>/{train.npz,test.npz}.
    # Used for the (label-skew) federated statistical-comparison suite.
    if kind == "npz_split":
        name = dataset_cfg.get("name")
        root = Path(dataset_cfg.get("root", "data"))
        ddir = root / name
        x_tv, y_tv = _load_npz_any(ddir / "train.npz")
        x_te, y_te = _load_npz_any(ddir / "test.npz")
        val_size = float(dataset_cfg.get("split", {}).get("val_size", 0.25))
        x_tr, x_val, y_tr, y_val = _stratified_split(
            _as_csr(x_tv), _as_csr(y_tv), test_size=val_size, seed=seed,
        )
        return DatasetSplit(
            x_train=_as_csr(x_tr), y_train=_as_csr(y_tr),
            x_val=_as_csr(x_val), y_val=_as_csr(y_val),
            x_test=_as_csr(x_te), y_test=_as_csr(y_te),
        )

    if kind != "prefold":
        raise ValueError(
            f"This reproducibility bundle only supports dataset.kind='prefold' "
            f"or 'npz_split' (got {kind!r})."
        )

    name = dataset_cfg.get("name")
    if not name:
        raise ValueError("dataset.name is required.")
    root = Path(dataset_cfg.get("root", "data/dense_benchmark_v3"))
    fold_root = root / name

    cv_cfg = config.get("cross_validation", {})
    if not cv_cfg.get("enabled", False):
        raise ValueError("kind=prefold requires cross_validation.enabled=true")
    if fold_idx is None:
        raise ValueError("fold_idx must be specified for kind=prefold")

    fold_dir = fold_root / f"fold{fold_idx}"
    train_path = fold_dir / "trainval.npz"
    if not train_path.exists():
        train_path = fold_dir / "train.npz"
    test_path = fold_dir / "test.npz"
    if not train_path.exists():
        raise FileNotFoundError(
            f"Fold data not found: {fold_dir}/trainval.npz or train.npz"
        )

    val_size = float(dataset_cfg.get("split", {}).get("val_size", 0.2))
    x_trainval, y_trainval = _load_npz_any(train_path)
    x_test, y_test = _load_npz_any(test_path)

    # Split trainval → train + val with stratification.
    x_train, x_val, y_train, y_val = _stratified_split(
        x_trainval, y_trainval, test_size=val_size, seed=seed,
    )
    return DatasetSplit(
        x_train=_as_csr(x_train),
        y_train=_as_csr(y_train),
        x_val=_as_csr(x_val),
        y_val=_as_csr(y_val),
        x_test=_as_csr(x_test),
        y_test=_as_csr(y_test),
    )
