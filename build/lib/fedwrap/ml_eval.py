"""Wrapper evaluator for the evolutionary search.

The paper uses **ML-kNN** (k=5, s=1.0) as the sole classifier inside
the evolutionary loop.  The tri-objective formulation minimises:

    (1 − Macro-F1,  1 − Micro-F1,  feature_ratio)

This module caches mask → (objectives, MLResult) so that identical
masks are never evaluated twice within the same fold.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse

from .metrics import MLResult, multilabel_metrics
from .mlknn_impl import MLkNNConfig, MLkNNModel


@dataclass
class EvalConfig:
    """Evaluation configuration (ML-kNN only)."""
    kind: str = "mlknn"
    primary_objective: str = "one_minus_macro_f1"
    objective_names: list[str] | None = None   # e.g. ["one_minus_macro_f1","one_minus_micro_f1","feature_ratio"]
    random_state: int = 0
    k: int = 5           # ML-kNN neighbours
    s: float = 1.0       # ML-kNN Laplace smoothing
    mlknn_backend: str = "auto"   # auto | torch | sklearn
    mlknn_device: str = "auto"    # auto | cpu | cuda
    cv_folds: int = 1             # k-fold CV for the (federated) selection objective; 1 = single split


class Evaluator:
    """Fitness evaluator: ML-kNN + tri-objective scoring.

    Results are cached by the boolean mask bytes to avoid redundant
    model training when NSGA-II revisits the same feature subset.
    """

    # Canonical name mapping (keeps config files readable).
    _ALIASES: dict[str, str] = {
        "hamming_loss": "hamming",
        "ranking_loss": "ranking",
        "micro_f1_loss": "one_minus_micro_f1",
        "macro_f1_loss": "one_minus_macro_f1",
        "avg_precision_loss": "one_minus_avg_precision",
        "gmean_f1_loss": "one_minus_gmean_f1",
    }

    @staticmethod
    def _canonical(name: str) -> str:
        return Evaluator._ALIASES.get(name.strip().lower(), name.strip().lower())

    def __init__(
        self,
        x_train: sparse.csr_matrix,
        y_train: sparse.csr_matrix,
        x_val: sparse.csr_matrix,
        y_val: sparse.csr_matrix,
        config: EvalConfig,
    ) -> None:
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.config = config
        self._cache: dict[bytes, tuple[np.ndarray, MLResult]] = {}

    # ── Public API ────────────────────────────────────────────────

    def evaluate_mask(self, feature_mask: np.ndarray) -> tuple[np.ndarray, MLResult]:
        """Evaluate a boolean feature mask and return (objectives, MLResult)."""
        mask = np.asarray(feature_mask, dtype=bool)
        key = mask.tobytes()
        if key in self._cache:
            return self._cache[key]

        n_obj = len(self.config.objective_names) if self.config.objective_names else 2
        if mask.sum() == 0:
            worst = (np.ones(n_obj, dtype=float),
                     MLResult(hamming=1.0, ranking=1.0, avg_precision=0.0,
                              f1_micro=0.0, f1_macro=0.0, one_error=1.0, zero_one_loss=1.0))
            self._cache[key] = worst
            return worst

        x_tr = self.x_train[:, mask]
        x_va = self.x_val[:, mask]
        y_tr = self.y_train.toarray().astype(int, copy=False)
        y_va = self.y_val.toarray().astype(int, copy=False)

        # Train ML-kNN
        model = MLkNNModel(MLkNNConfig(
            k=int(self.config.k),
            s=float(self.config.s),
            backend=str(self.config.mlknn_backend),
            device=str(self.config.mlknn_device),
        ))
        model.fit(x_tr, sparse.csr_matrix(y_tr))
        y_score = model.predict_proba(x_va)
        # Single-label multi-class tasks (every training row one-hot) are scored by top-1
        # projection onto the one-hot simplex; genuine multi-label tasks use per-label
        # thresholding. Detected from the training labels, so no per-dataset configuration.
        if y_tr.shape[1] > 1 and bool((y_tr.sum(axis=1) == 1).all()):
            y_pred = np.zeros_like(y_score, dtype=int)
            y_pred[np.arange(y_score.shape[0]), np.asarray(y_score).argmax(axis=1)] = 1
        else:
            y_pred = (y_score >= 0.5).astype(int)

        ml = multilabel_metrics(y_va, y_pred, y_score)
        feature_ratio = float(mask.sum() / mask.size)

        # Build objective vector.
        objectives = self._build_objectives(ml, feature_ratio)
        self._cache[key] = (objectives, ml)
        return objectives, ml

    # ── Objective construction ────────────────────────────────────

    def _build_objectives(self, ml: MLResult, feature_ratio: float) -> np.ndarray:
        if not self.config.objective_names:
            return np.array([ml.hamming, feature_ratio], dtype=float)

        obj: list[float] = []
        for name in self.config.objective_names:
            canon = self._canonical(name)
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
