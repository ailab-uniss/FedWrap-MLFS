"""Virtual federated client.

A client owns its local train/validation data and never exposes it. Given a
feature mask it trains a local ML-kNN model on the selected features and returns
only label-wise sufficient statistics (TP/FP/FN per label) plus a few counters.
The client only knows ``receive mask -> evaluate locally -> return statistics``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse

from ..mlknn_impl import MLkNNConfig, MLkNNModel


@dataclass
class ClientEvalConfig:
    kind: str = "mlknn"  # mlknn only on the client side for now
    k: int = 10
    s: float = 1.0
    mlknn_backend: str = "auto"
    mlknn_device: str = "auto"
    threshold: float = 0.5
    # Cross-validated selection: when cv_folds>1, a full evaluation scores a mask by
    # k-fold CV over the client's pooled train+val data (every sample predicted once as
    # held-out), summing TP/FP/FN over folds. This makes the search objective robust and
    # avoids overfitting a single small validation split. cv_folds=1 keeps the single split.
    cv_folds: int = 1
    # Adaptive backend: dense masks (many selected features) are much faster on
    # GPU, sparse masks faster on CPU (GPU transfer overhead dominates). When
    # mlknn_backend == "adaptive" we pick per-evaluation by selected-feature count.
    gpu_feature_threshold: int = 600


def _cupy_available() -> bool:
    """Lazy CuPy/CUDA check (kept lazy so it can run inside spawned workers)."""
    try:
        import cupy as _cp  # noqa: F401
        return _cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _make_mlknn(cfg: ClientEvalConfig, n_selected: int, k: int | None = None) -> MLkNNModel:
    return MLkNNModel(MLkNNConfig(k=int(k if k is not None else cfg.k), s=float(cfg.s),
                                  backend=str(cfg.mlknn_backend), device=str(cfg.mlknn_device)))


def _calibrate_thresholds(Ycal: np.ndarray, Pcal: np.ndarray) -> np.ndarray:
    """Per-label decision threshold maximizing F1 on a held-out calibration split.
    A fixed 0.5 threshold is poor for rare multi-label classes (their base rate is
    below 0.5, so they are never predicted); calibrating per label fixes this. Labels
    with no calibration positives fall back to 0.5."""
    L = Ycal.shape[1]
    thr = np.full(L, 0.5, dtype=np.float32)
    grid = np.linspace(0.05, 0.95, 19)
    for l in range(L):
        pos = Ycal[:, l].sum()
        if pos == 0:
            continue
        p = Pcal[:, l]; y = Ycal[:, l]
        best_f1, best_t = -1.0, 0.5
        for t in grid:
            pred = (p >= t)
            tp = np.logical_and(pred, y == 1).sum()
            fp = np.logical_and(pred, y == 0).sum()
            fn = np.logical_and(~pred, y == 1).sum()
            den = 2 * tp + fp + fn
            f1 = (2 * tp / den) if den > 0 else 0.0
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thr[l] = best_t
    return thr


def _logreg_predict_calibrated(Xtr: np.ndarray, Ytr: np.ndarray, Xva: np.ndarray, seed: int = 0) -> np.ndarray:
    """Binary-relevance logistic regression with per-label threshold calibration. Thresholds are
    tuned on an internal 25% split of the TRAIN data (never on Xva), then the model is refit on all
    train and the calibrated thresholds applied to Xva. Returns the binary prediction matrix."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.preprocessing import StandardScaler
    L = int(Ytr.shape[1])
    sc = StandardScaler().fit(Xtr)
    Xtr = sc.transform(Xtr); Xva = sc.transform(Xva)
    n = Xtr.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n); ncal = max(1, int(round(0.25 * n))); trn = perm[ncal:]; cal = perm[:ncal]
    thr = np.full(L, 0.5, dtype=np.float32)
    if len(trn) >= 2 and len(cal) >= 1 and Ytr[trn].sum() > 0:
        clf0 = OneVsRestClassifier(LogisticRegression(max_iter=300, C=1.0)).fit(Xtr[trn], Ytr[trn])
        Pcal = np.asarray(clf0.predict_proba(Xtr[cal]))
        if Pcal.ndim == 1:
            Pcal = Pcal.reshape(-1, 1)
        thr = _calibrate_thresholds(Ytr[cal], Pcal)
    clf = OneVsRestClassifier(LogisticRegression(max_iter=300, C=1.0)).fit(Xtr, Ytr)
    Pva = np.asarray(clf.predict_proba(Xva))
    if Pva.ndim == 1:
        Pva = Pva.reshape(-1, 1)
    return (Pva >= thr[None, :]).astype(np.int8)


def _predict_gpu(x_tr_sel, y_tr_dense: np.ndarray, x_va_sel, cfg: ClientEvalConfig, k_eff: int) -> np.ndarray:
    """GPU ML-kNN via CuPy (fedwrap.mlknn_gpu) — fast for the dense, high-dimensional
    masks of federated multi-label FS. Returns the binary prediction matrix."""
    from ..mlknn_gpu import fit_and_predict_gpu
    Xtr = x_tr_sel.toarray().astype(np.float32) if sparse.issparse(x_tr_sel) else np.asarray(x_tr_sel, np.float32)
    Xva = x_va_sel.toarray().astype(np.float32) if sparse.issparse(x_va_sel) else np.asarray(x_va_sel, np.float32)
    # mlknn_gpu ranks neighbours by Euclidean distance; row-L2-normalising makes that
    # equivalent to the cosine-metric kNN used by the CPU path (consistent results).
    Xtr /= (np.linalg.norm(Xtr, axis=1, keepdims=True) + 1e-12)
    Xva /= (np.linalg.norm(Xva, axis=1, keepdims=True) + 1e-12)
    y_pred, _proba = fit_and_predict_gpu(
        Xtr, y_tr_dense.astype(np.int8), Xva,
        k=int(k_eff), smooth=float(cfg.s), threshold=float(cfg.threshold),
    )
    return y_pred.astype(np.int8)


class FederatedClient:
    def __init__(
        self,
        client_id: int,
        x_train: sparse.csr_matrix,
        y_train: sparse.csr_matrix,
        x_val: sparse.csr_matrix,
        y_val: sparse.csr_matrix,
        n_labels: int,
        cfg: ClientEvalConfig,
    ) -> None:
        self.client_id = int(client_id)
        self.x_train = x_train.tocsr()
        self.y_train = y_train.tocsr()
        self.x_val = x_val.tocsr()
        self.y_val = y_val.tocsr()
        self.n_labels = int(n_labels)
        self.cfg = cfg
        self._y_val_dense = (self.y_val.toarray() > 0).astype(np.int8)
        self._y_train_dense = (self.y_train.toarray() > 0).astype(np.int8)

    @property
    def n_train(self) -> int:
        return int(self.x_train.shape[0])

    @property
    def n_val(self) -> int:
        return int(self.x_val.shape[0])

    def _zero_prediction_stats(self) -> dict[str, Any]:
        # Empty mask -> predict all-zero -> TP=0, FP=0, FN = positives in val.
        fn = self._y_val_dense.sum(axis=0).astype(np.int64)
        return {
            "client_id": self.client_id,
            "tp": np.zeros(self.n_labels, dtype=np.int64),
            "fp": np.zeros(self.n_labels, dtype=np.int64),
            "fn": fn,
            "n_val": self.n_val,
            "mode": "empty",
            "eval_time": 0.0,
        }

    def evaluate_mask(
        self,
        mask: np.ndarray,
        mode: str = "full",
    ) -> dict[str, Any]:
        selected = np.flatnonzero(np.asarray(mask, dtype=bool))
        if selected.size == 0:
            return self._zero_prediction_stats()

        t0 = time.perf_counter()
        if int(self.cfg.cv_folds) > 1:
            tp, fp, fn, nval = self._eval_cv(selected)
        else:
            tp, fp, fn, nval = self._eval_split(
                selected, self.x_train, self.y_train, self.x_val, self._y_val_dense)

        return {
            "client_id": self.client_id,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "n_val": int(nval),
            "mode": mode,
            "eval_time": time.perf_counter() - t0,
        }

    def _eval_split(self, selected, x_tr, y_tr, x_va, y_va):
        """Train ML-kNN on (x_tr,y_tr) restricted to ``selected`` and return summed
        (TP, FP, FN, n_val) over the held-out (x_va,y_va)."""
        x_tr_sel = x_tr[:, selected]
        x_va_sel = x_va[:, selected]
        # Tiny silos (quantity-skew) may hold <k local rows; ML-kNN needs k<n_train.
        # With <2 rows no neighbourhood can form -> predict all-zero (only FN), so the
        # held-out positives are counted as false negatives in the global confusion.
        n_tr_eff = int(x_tr_sel.shape[0])
        ytrue = y_va if isinstance(y_va, np.ndarray) else (y_va.toarray() > 0).astype(np.int8)
        if n_tr_eff < 2:
            z = np.zeros(self.n_labels, np.int64)
            return z, z.copy(), ytrue.sum(axis=0).astype(np.int64), int(ytrue.shape[0])
        # Alternative wrapper evaluator: binary-relevance logistic regression (linear,
        # parametric, non-neighbour). Used only when cfg.kind=='logreg'; the default mlknn
        # path below is unchanged.
        if str(self.cfg.kind) == "logreg":
            ytr_dense = y_tr.toarray().astype(np.int8) if sparse.issparse(y_tr) else np.asarray(y_tr, np.int8)
            Xtr = x_tr_sel.toarray().astype(np.float32); Xva = x_va_sel.toarray().astype(np.float32)
            y_pred = _logreg_predict_calibrated(Xtr, ytr_dense, Xva, seed=self.client_id)
            tp = np.logical_and(y_pred == 1, ytrue == 1).sum(axis=0).astype(np.int64)
            fp = np.logical_and(y_pred == 1, ytrue == 0).sum(axis=0).astype(np.int64)
            fn = np.logical_and(y_pred == 0, ytrue == 1).sum(axis=0).astype(np.int64)
            return tp, fp, fn, int(ytrue.shape[0])
        k_eff = max(1, min(int(self.cfg.k), n_tr_eff - 1))
        backend = str(self.cfg.mlknn_backend)
        use_gpu = backend == "gpu" or (backend == "adaptive" and _cupy_available())
        ytr_dense = y_tr.toarray().astype(np.int8) if sparse.issparse(y_tr) else np.asarray(y_tr, np.int8)
        # Single-label multi-class task: every training row is one-hot -> score by top-1
        # projection. Multi-label tasks (some row with >1 active label) -> per-label threshold.
        single_label = ytr_dense.shape[1] > 1 and bool((ytr_dense.sum(axis=1) == 1).all())
        if use_gpu:
            y_pred = _predict_gpu(x_tr_sel, ytr_dense, x_va_sel, self.cfg, k_eff)
        else:
            model = _make_mlknn(self.cfg, n_selected=int(selected.size), k=k_eff)
            model.fit(x_tr_sel, sparse.csr_matrix(ytr_dense))
            y_score = model.predict_proba(x_va_sel)
            if single_label:
                y_pred = np.zeros_like(np.asarray(y_score), dtype=np.int8)
                y_pred[np.arange(y_pred.shape[0]), np.asarray(y_score).argmax(axis=1)] = 1
            else:
                y_pred = (y_score >= float(self.cfg.threshold)).astype(np.int8)
        tp = np.logical_and(y_pred == 1, ytrue == 1).sum(axis=0).astype(np.int64)
        fp = np.logical_and(y_pred == 1, ytrue == 0).sum(axis=0).astype(np.int64)
        fn = np.logical_and(y_pred == 0, ytrue == 1).sum(axis=0).astype(np.int64)
        return tp, fp, fn, int(ytrue.shape[0])

    def _eval_cv(self, selected):
        """k-fold cross-validated evaluation over the client's pooled train+val data:
        every sample is predicted once while held out, and TP/FP/FN are summed over folds.
        This is the robust selection objective (a single split overfits under search)."""
        X = sparse.vstack([self.x_train, self.x_val]).tocsr()
        Y = np.vstack([self._y_train_dense, self._y_val_dense]).astype(np.int8)
        n = X.shape[0]
        folds = int(self.cfg.cv_folds)
        if n < 2 * folds:                       # too few rows for k folds: clamp
            folds = max(2, n // 2)
        if n < 4:                               # degenerate: fall back to single split
            return self._eval_split(selected, self.x_train, self.y_train, self.x_val, self._y_val_dense)
        rng = np.random.default_rng(self.client_id)
        order = rng.permutation(n)
        bounds = np.linspace(0, n, folds + 1).astype(int)
        tp = np.zeros(self.n_labels, np.int64); fp = tp.copy(); fn = tp.copy(); nval = 0
        for f in range(folds):
            te = order[bounds[f]:bounds[f + 1]]
            tr = np.concatenate([order[:bounds[f]], order[bounds[f + 1]:]])
            if te.size == 0 or tr.size == 0:
                continue
            a, b, c, m = self._eval_split(selected, X[tr], Y[tr], X[te], Y[te])
            tp += a; fp += b; fn += c; nval += m
        return tp, fp, fn, nval
