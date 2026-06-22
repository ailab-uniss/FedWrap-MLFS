"""
GPU-accelerated ML-kNN utilities (CuPy).

This module provides a minimal ML-kNN implementation where the kNN search is
performed on GPU via CuPy. It is meant as a drop-in building block for faster
evaluation pipelines (especially when repeatedly evaluating many feature
budgets).

Notes
-----
- Requires `cupy` to be installed and a working CUDA stack.
- The training-time neighbor graph for ML-kNN is O(n_train^2) if computed via
  brute-force distances. Use with care on very large folds, or plug in an
  external ANN/kNN index and call `fit_mlknn_from_neighbor_counts` directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


def _require_cupy():
    try:
        import cupy as cp  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "CuPy is required for GPU ML-kNN. Install e.g. `pip install cupy-cuda12x` "
            "(choose the CUDA version matching your system)."
        ) from e
    return cp


def knn_indices_gpu(
    X_train: np.ndarray,
    X_query: np.ndarray,
    *,
    k: int,
    chunk_size: int = 2048,
    exclude_self: bool = False,
) -> np.ndarray:
    """
    Return indices of the k nearest neighbors in `X_train` for each row in `X_query`.

    Parameters
    ----------
    exclude_self:
      If True, assumes `X_query` and `X_train` refer to the *same* array and
      removes the identical index from the neighborhood (useful for train→train).
    """
    cp = _require_cupy()
    Xtr = np.asarray(X_train, dtype=np.float32, order="C")
    Xq = np.asarray(X_query, dtype=np.float32, order="C")
    n_tr = int(Xtr.shape[0])
    n_q = int(Xq.shape[0])
    k = int(k)
    if k <= 0:
        raise ValueError("k must be positive.")
    if k >= n_tr and not exclude_self:
        raise ValueError("k must be < n_train.")
    if exclude_self and k >= n_tr:
        raise ValueError("k must be < n_train when exclude_self=True.")

    # Upload once.
    Xtr_g = cp.asarray(Xtr)
    tr_norm = cp.sum(Xtr_g * Xtr_g, axis=1)  # (n_tr,)

    out = np.empty((n_q, k), dtype=np.int64)
    for i0 in range(0, n_q, int(chunk_size)):
        i1 = min(n_q, i0 + int(chunk_size))
        Xc_g = cp.asarray(Xq[i0:i1])
        q_norm = cp.sum(Xc_g * Xc_g, axis=1)  # (chunk,)
        # Squared Euclidean distances via ||a-b||^2 = ||a||^2 + ||b||^2 - 2a·b
        d2 = q_norm[:, None] + tr_norm[None, :] - 2.0 * (Xc_g @ Xtr_g.T)
        if exclude_self:
            # For the i-th query (global index), the matching train index is i.
            rows = cp.arange(i1 - i0, dtype=cp.int64)
            cols = cp.arange(i0, i1, dtype=cp.int64)
            d2[rows, cols] = cp.inf
        idx = cp.argpartition(d2, kth=k - 1, axis=1)[:, :k]
        # Optional: order neighbors by distance for determinism.
        d2_k = cp.take_along_axis(d2, idx, axis=1)
        order = cp.argsort(d2_k, axis=1)
        idx = cp.take_along_axis(idx, order, axis=1)
        out[i0:i1] = cp.asnumpy(idx).astype(np.int64, copy=False)
    return out


@dataclass(frozen=True)
class MLKNNModel:
    k: int
    smooth: float
    prior_pos: np.ndarray  # (L,)
    prior_neg: np.ndarray  # (L,)
    cond_pos: np.ndarray  # (L, k+1)  P(C=c | y=1)
    cond_neg: np.ndarray  # (L, k+1)  P(C=c | y=0)


def _counts_per_label(Y: np.ndarray, knn_idx: np.ndarray) -> np.ndarray:
    """
    Return count matrix C where C[i,l] = #neighbors of i with label l.
    Y: (n, L) in {0,1}; knn_idx: (n, k) indices into Y.
    """
    Y = np.asarray(Y, dtype=np.int8)
    n, L = Y.shape
    n_query = knn_idx.shape[0]
    k = int(knn_idx.shape[1])
    C = np.zeros((n_query, L), dtype=np.int16)
    # Vectorized gather per neighbor (k is small, so a loop is fine)
    for j in range(k):
        C += Y[knn_idx[:, j]]
    return C


def fit_mlknn(
    Y_train: np.ndarray,
    knn_train: np.ndarray,
    *,
    smooth: float = 1.0,
) -> MLKNNModel:
    """
    Fit ML-kNN priors and conditional tables given precomputed train→train neighbors.
    """
    Y = np.asarray(Y_train, dtype=np.int8)
    n, L = Y.shape
    k = int(knn_train.shape[1])
    smooth = float(smooth)

    C = _counts_per_label(Y, knn_train)  # (n, L), values in 0..k

    pos = (Y == 1)
    neg = ~pos
    n_pos = pos.sum(axis=0).astype(np.float64)
    n_neg = neg.sum(axis=0).astype(np.float64)

    prior_pos = (smooth + n_pos) / (2.0 * smooth + float(n))
    prior_neg = 1.0 - prior_pos

    cond_pos = np.zeros((L, k + 1), dtype=np.float64)
    cond_neg = np.zeros((L, k + 1), dtype=np.float64)

    # For each label and each neighbor-count c, estimate P(C=c|y=1) and P(C=c|y=0).
    for l in range(L):
        c_l = C[:, l]
        for c in range(k + 1):
            cond_pos[l, c] = smooth + float(np.sum((c_l == c) & pos[:, l]))
            cond_neg[l, c] = smooth + float(np.sum((c_l == c) & neg[:, l]))
        cond_pos[l, :] /= (smooth * float(k + 1) + float(n_pos[l]))
        cond_neg[l, :] /= (smooth * float(k + 1) + float(n_neg[l]))

    return MLKNNModel(
        k=k,
        smooth=smooth,
        prior_pos=prior_pos,
        prior_neg=prior_neg,
        cond_pos=cond_pos,
        cond_neg=cond_neg,
    )


def predict_proba_mlknn(
    model: MLKNNModel,
    *,
    Y_train: np.ndarray,
    knn_test: np.ndarray,
) -> np.ndarray:
    """
    Compute per-label posterior probabilities P(y=1 | neighbor-count) for test instances.
    """
    Y = np.asarray(Y_train, dtype=np.int8)
    C = _counts_per_label(Y, knn_test)  # (n_test, L)
    n_test, L = C.shape
    k = int(model.k)
    if k != int(knn_test.shape[1]):
        raise ValueError("model.k and knn_test.shape[1] mismatch.")

    proba = np.empty((n_test, L), dtype=np.float64)
    for l in range(L):
        c = np.asarray(C[:, l], dtype=np.int64)
        p1 = model.prior_pos[l] * model.cond_pos[l, c]
        p0 = model.prior_neg[l] * model.cond_neg[l, c]
        proba[:, l] = p1 / np.maximum(p1 + p0, np.finfo(np.float64).eps)
    return proba


def predict_mlknn(
    model: MLKNNModel,
    *,
    Y_train: np.ndarray,
    knn_test: np.ndarray,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (Y_pred, proba) for test instances.
    """
    proba = predict_proba_mlknn(model, Y_train=Y_train, knn_test=knn_test)
    Yhat = (proba >= float(threshold)).astype(np.int8)
    return Yhat, proba


def fit_and_predict_gpu(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    k: int = 10,
    smooth: float = 1.0,
    chunk_size: int = 2048,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience: compute GPU kNN, fit ML-kNN, and predict.
    """
    knn_tr = knn_indices_gpu(X_train, X_train, k=int(k), chunk_size=int(chunk_size), exclude_self=True)
    model = fit_mlknn(Y_train, knn_tr, smooth=float(smooth))
    knn_te = knn_indices_gpu(X_train, X_test, k=int(k), chunk_size=int(chunk_size), exclude_self=False)
    return predict_mlknn(model, Y_train=Y_train, knn_test=knn_te, threshold=float(threshold))


__all__ = [
    "knn_indices_gpu",
    "MLKNNModel",
    "fit_mlknn",
    "predict_proba_mlknn",
    "predict_mlknn",
    "fit_and_predict_gpu",
]

