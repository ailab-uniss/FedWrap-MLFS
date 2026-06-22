"""Inner k-fold ML-kNN evaluator for search-time fitness.

This evaluator is intended for CC-FedWrap-MLFS search only. It keeps the external
validation/test protocol unchanged, but replaces the search-time holdout score
with an out-of-fold score computed on the current training split.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse

try:
    from skmultilearn.model_selection import IterativeStratification
    SKMULTILEARN_AVAILABLE = True
except ImportError:
    SKMULTILEARN_AVAILABLE = False

from .metrics import MLResult, multilabel_metrics
from .ml_eval import EvalConfig, Evaluator


@contextlib.contextmanager
def _temporary_numpy_seed(seed: int):
    state = np.random.get_state()
    np.random.seed(int(seed))
    try:
        yield
    finally:
        np.random.set_state(state)


def _iterative_kfold_indices(
    x: sparse.csr_matrix,
    y: sparse.csr_matrix,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if not SKMULTILEARN_AVAILABLE:
        raise ImportError(
            "scikit-multilearn is required for inner CV splits. "
            "Install with: pip install scikit-multilearn"
        )
    dist = [1.0 / float(n_splits)] * int(n_splits)
    try:
        stratifier = IterativeStratification(
            n_splits=int(n_splits),
            order=1,
            sample_distribution_per_fold=dist,
            random_state=int(seed),
        )
    except (TypeError, ValueError):
        stratifier = IterativeStratification(
            n_splits=int(n_splits),
            order=1,
            sample_distribution_per_fold=dist,
        )

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    with _temporary_numpy_seed(int(seed)):
        for train_idx, val_idx in stratifier.split(x, y):
            folds.append((
                np.asarray(train_idx, dtype=np.int64),
                np.asarray(val_idx, dtype=np.int64),
            ))
    if len(folds) != int(n_splits):
        raise RuntimeError(
            f"IterativeStratification returned {len(folds)} folds, expected {n_splits}."
        )
    return folds


@dataclass(frozen=True)
class _TorchFold:
    train_idx: np.ndarray
    val_idx: np.ndarray
    x_train: Any
    x_val: Any
    y_train: Any
    y_train_bool: Any
    y_val_true: np.ndarray
    prior_true: Any
    prior_false: Any


@dataclass(frozen=True)
class _SparseFold:
    train_idx: np.ndarray
    val_idx: np.ndarray
    x_train: sparse.csr_matrix
    x_val: sparse.csr_matrix
    y_train: sparse.csr_matrix
    y_val: sparse.csr_matrix


class InnerCVEvaluator:
    """ML-kNN evaluator that uses inner CV for search-time fitness."""

    def __init__(
        self,
        x_train: sparse.csr_matrix,
        y_train: sparse.csr_matrix,
        config: EvalConfig,
        n_folds: int = 3,
        seed: int = 0,
    ) -> None:
        self.x_train = x_train.tocsr()
        self.y_train = y_train.tocsr()
        self.config = config
        self.n_folds = int(n_folds)
        self.seed = int(seed)
        self._cache: dict[bytes, tuple[np.ndarray, MLResult]] = {}

        self._n_samples, self._n_features = self.x_train.shape
        self._n_labels = self.y_train.shape[1]
        self._y_true_dense = self.y_train.toarray().astype(int, copy=False)
        self._fold_indices = _iterative_kfold_indices(
            self.x_train, self.y_train, n_splits=self.n_folds, seed=self.seed
        )

        self._backend_selected = "sklearn"
        self._device = None
        self._torch_folds: list[_TorchFold] = []
        self._sparse_folds: list[_SparseFold] = []
        self._prepare_backend()

    def _prepare_backend(self) -> None:
        backend = str(self.config.mlknn_backend).strip().lower()
        device = str(self.config.mlknn_device).strip().lower()

        use_torch = backend != "sklearn"
        torch = None
        if use_torch:
            try:
                import torch as _torch
                torch = _torch
            except ImportError:
                if backend == "torch":
                    raise RuntimeError(
                        "InnerCVEvaluator requested backend='torch' but PyTorch is not installed."
                    )
                use_torch = False

        if use_torch and torch is not None:
            if device == "cpu":
                self._device = torch.device("cpu")
            elif device == "cuda":
                self._device = torch.device("cuda")
            else:
                self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._backend_selected = "torch"
            self._prepare_torch_folds(torch)
            return

        self._backend_selected = "sklearn"
        for train_idx, val_idx in self._fold_indices:
            self._sparse_folds.append(_SparseFold(
                train_idx=train_idx,
                val_idx=val_idx,
                x_train=self.x_train[train_idx].tocsr(),
                x_val=self.x_train[val_idx].tocsr(),
                y_train=self.y_train[train_idx].tocsr(),
                y_val=self.y_train[val_idx].tocsr(),
            ))

    def _prepare_torch_folds(self, torch: Any) -> None:
        x_dense = self.x_train.toarray().astype(np.float32, copy=False)
        y_dense = self.y_train.toarray().astype(np.float32, copy=False)
        s = float(self.config.s)

        for train_idx, val_idx in self._fold_indices:
            y_train_np = y_dense[train_idx]
            y_train_t = torch.from_numpy(y_train_np).float().to(self._device)
            pos = y_train_t.sum(dim=0)
            n_train = int(train_idx.size)
            prior_true = (s + pos) / (2.0 * s + n_train)

            self._torch_folds.append(_TorchFold(
                train_idx=train_idx,
                val_idx=val_idx,
                x_train=torch.from_numpy(x_dense[train_idx]).float().to(self._device),
                x_val=torch.from_numpy(x_dense[val_idx]).float().to(self._device),
                y_train=y_train_t,
                y_train_bool=y_train_t.bool(),
                y_val_true=(y_dense[val_idx] > 0).astype(int, copy=False),
                prior_true=prior_true,
                prior_false=1.0 - prior_true,
            ))

    def evaluate_mask(self, feature_mask: np.ndarray) -> tuple[np.ndarray, MLResult]:
        mask = np.asarray(feature_mask, dtype=bool)
        key = mask.tobytes()
        if key in self._cache:
            return self._cache[key]

        n_obj = len(self.config.objective_names) if self.config.objective_names else 2
        if mask.sum() == 0:
            worst = (
                np.ones(n_obj, dtype=float),
                MLResult(
                    hamming=1.0,
                    ranking=1.0,
                    avg_precision=0.0,
                    f1_micro=0.0,
                    f1_macro=0.0,
                    one_error=1.0,
                    zero_one_loss=1.0,
                ),
            )
            self._cache[key] = worst
            return worst

        if self._backend_selected == "torch":
            objectives, ml = self._evaluate_mask_torch(mask)
        else:
            objectives, ml = self._evaluate_mask_sklearn(mask)
        self._cache[key] = (objectives, ml)
        return objectives, ml

    def _evaluate_mask_torch(self, mask: np.ndarray) -> tuple[np.ndarray, MLResult]:
        import torch

        idx_np = np.flatnonzero(mask).astype(np.int64, copy=False)
        idx_t = torch.from_numpy(idx_np).to(self._device)
        all_scores = np.zeros((self._n_samples, self._n_labels), dtype=np.float32)

        with torch.no_grad():
            for fold in self._torch_folds:
                xt = torch.index_select(fold.x_train, 1, idx_t)
                xv = torch.index_select(fold.x_val, 1, idx_t)

                xt_norm = torch.nn.functional.normalize(xt, p=2, dim=1, eps=1e-12)
                sim_tt = torch.mm(xt_norm, xt_norm.t())

                k_req = int(self.config.k)
                target_k = min(k_req + 1, xt_norm.shape[0])
                _, indices = torch.topk(sim_tt, k=target_k, dim=1)
                neigh = indices[:, 1:] if indices.shape[1] > k_req else indices
                k_eff = int(neigh.shape[1])
                if k_eff <= 0:
                    all_scores[fold.val_idx] = 0.5
                    continue

                flat = neigh.reshape(-1)
                nl = torch.index_select(fold.y_train, 0, flat).view(
                    xt_norm.shape[0], k_eff, self._n_labels
                )
                lc = nl.sum(dim=1).long()

                pos = fold.y_train.sum(dim=0)
                neg = float(xt_norm.shape[0]) - pos
                cond_true = torch.zeros((self._n_labels, k_eff + 1), device=self._device)
                cond_false = torch.zeros((self._n_labels, k_eff + 1), device=self._device)

                for c in range(k_eff + 1):
                    mask_c = lc == c
                    ct = (mask_c & fold.y_train_bool).sum(dim=0).float()
                    cf = (mask_c & ~fold.y_train_bool).sum(dim=0).float()
                    cond_true[:, c] = (float(self.config.s) + ct) / (
                        float(self.config.s) * (k_eff + 1) + pos
                    )
                    cond_false[:, c] = (float(self.config.s) + cf) / (
                        float(self.config.s) * (k_eff + 1) + neg
                    )

                xv_norm = torch.nn.functional.normalize(xv, p=2, dim=1, eps=1e-12)
                sim_vt = torch.mm(xv_norm, xt_norm.t())
                _, val_idx = torch.topk(sim_vt, k=k_eff, dim=1)
                flat = val_idx.reshape(-1)
                nl = torch.index_select(fold.y_train, 0, flat).view(
                    xv_norm.shape[0], k_eff, self._n_labels
                )
                lc = nl.sum(dim=1).long()

                counts_t = lc.t()
                pt_neigh = torch.gather(cond_true, 1, counts_t)
                pf_neigh = torch.gather(cond_false, 1, counts_t)

                prob_true = fold.prior_true.unsqueeze(1) * pt_neigh
                prob_false = fold.prior_false.unsqueeze(1) * pf_neigh
                probs = prob_true / (prob_true + prob_false + 1e-10)
                all_scores[fold.val_idx] = probs.t().cpu().numpy()

        y_pred = (all_scores >= 0.5).astype(int)
        ml = multilabel_metrics(self._y_true_dense, y_pred, all_scores)
        feature_ratio = float(mask.sum() / mask.size)
        return self._build_objectives(ml, feature_ratio), ml

    def _evaluate_mask_sklearn(self, mask: np.ndarray) -> tuple[np.ndarray, MLResult]:
        fold_metrics: list[MLResult] = []
        for fold in self._sparse_folds:
            evaluator = Evaluator(fold.x_train, fold.y_train, fold.x_val, fold.y_val, self.config)
            _, ml = evaluator.evaluate_mask(mask)
            fold_metrics.append(ml)

        if not fold_metrics:
            raise RuntimeError("Inner CV produced no folds.")

        ml = MLResult(
            hamming=float(np.mean([m.hamming for m in fold_metrics])),
            ranking=float(np.mean([m.ranking for m in fold_metrics])),
            avg_precision=float(np.mean([m.avg_precision for m in fold_metrics])),
            f1_micro=float(np.mean([m.f1_micro for m in fold_metrics])),
            f1_macro=float(np.mean([m.f1_macro for m in fold_metrics])),
            one_error=float(np.mean([m.one_error for m in fold_metrics])),
            zero_one_loss=float(np.mean([m.zero_one_loss for m in fold_metrics])),
        )
        feature_ratio = float(mask.sum() / mask.size)
        return self._build_objectives(ml, feature_ratio), ml

    def _build_objectives(self, ml: MLResult, feature_ratio: float) -> np.ndarray:
        if not self.config.objective_names:
            return np.array([ml.hamming, feature_ratio], dtype=float)

        obj: list[float] = []
        for name in self.config.objective_names:
            canon = Evaluator._canonical(name)
            if canon == "hamming":
                obj.append(ml.hamming)
            elif canon == "ranking":
                obj.append(ml.ranking)
            elif canon == "one_minus_micro_f1":
                obj.append(1.0 - float(ml.f1_micro))
            elif canon == "one_minus_macro_f1":
                obj.append(1.0 - float(ml.f1_macro))
            elif canon == "one_minus_avg_precision":
                obj.append(1.0 - float(ml.avg_precision))
            elif canon == "one_minus_gmean_f1":
                obj.append(1.0 - float(np.sqrt(ml.f1_micro * ml.f1_macro)))
            elif canon == "feature_ratio":
                obj.append(feature_ratio)
            else:
                raise ValueError(f"Unknown objective: {name}")
        return np.array(obj, dtype=float)
