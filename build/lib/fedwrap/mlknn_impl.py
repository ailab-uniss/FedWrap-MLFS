"""ML-kNN implementation with sklearn and PyTorch backends.

The paper uses ML-kNN (Zhang & Zhou, 2007) as the wrapper evaluator
with K=5 and Laplace smoothing s=1.0.  The PyTorch backend accelerates
the dominant cosine-similarity kNN computation on GPU; the sklearn
backend is the reference fallback.

Only the ``torch`` and ``sklearn`` backends are included in this bundle.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class MLkNNConfig:
    k: int = 5          # number of neighbours
    s: float = 1.0      # Laplace smoothing parameter
    metric: str = "cosine"
    backend: str = "auto"  # auto | torch | sklearn
    device: str = "auto"   # auto | cpu | cuda


class MLkNNModel:
    """ML-kNN classifier (Zhang & Zhou, 2007).

    Two execution paths:

    * **torch**: densifies X, computes cosine kNN on GPU via
      ``torch.topk``, and vectorises the conditional-probability
      tables.  Fast when N is moderate (≤ ~15 k) and a GPU is
      available.
    * **sklearn**: uses ``NearestNeighbors`` with sparse inputs on CPU.
      Slower but always available.
    """

    def __init__(self, cfg: MLkNNConfig) -> None:
        if cfg.k <= 0:
            raise ValueError("k must be > 0")
        if cfg.s <= 0:
            raise ValueError("s must be > 0")
        self.cfg = cfg

        # Torch state
        self._use_torch = False
        self._device = None
        self._backend_selected: str | None = None

        backend = str(cfg.backend).strip().lower()
        device = str(cfg.device).strip().lower()

        if backend in {"auto", "torch"}:
            try:
                import torch
                if device == "cpu":
                    self._device = torch.device("cpu")
                elif device == "cuda":
                    self._device = torch.device("cuda")
                else:
                    self._device = torch.device(
                        "cuda" if torch.cuda.is_available() else "cpu"
                    )
                self._use_torch = True
            except ImportError:
                self._use_torch = False

        # Sklearn state
        self._nn: NearestNeighbors | None = None
        self._y_train: sparse.csr_matrix | None = None
        self._prior_true: np.ndarray | None = None
        self._prior_false: np.ndarray | None = None
        self._cond_true: np.ndarray | None = None
        self._cond_false: np.ndarray | None = None

    # ── Backend selection ─────────────────────────────────────────

    def _select_backend(self, x_train: sparse.csr_matrix) -> str:
        backend = str(self.cfg.backend).strip().lower()
        if backend == "sklearn":
            return "sklearn"
        if backend == "torch":
            if not self._use_torch:
                raise RuntimeError(
                    "backend='torch' requested but PyTorch is not installed."
                )
            return "torch"
        # auto: prefer torch when available
        if self._use_torch:
            return "torch"
        return "sklearn"

    # ── Torch backend ─────────────────────────────────────────────

    def _fit_torch(
        self,
        x_train: sparse.csr_matrix,
        y_train: sparse.csr_matrix,
    ) -> None:
        import torch

        k = int(self.cfg.k)
        s = float(self.cfg.s)
        n, _ = x_train.shape
        m = y_train.shape[1]

        xt_t = torch.from_numpy(x_train.toarray()).float().to(self._device)
        yt_t = torch.from_numpy(y_train.toarray()).float().to(self._device)

        self._xt_train = xt_t
        self._yt_train = yt_t

        # Cosine kNN via normalised dot product.
        xt_norm = torch.nn.functional.normalize(xt_t, p=2, dim=1)
        sim = torch.mm(xt_norm, xt_norm.t())

        target_k = min(k + 1, n)
        _, indices = torch.topk(sim, k=target_k, dim=1)
        # Drop self (usually first column).
        neigh = indices[:, 1:] if indices.shape[1] > k else indices

        flat = neigh.reshape(-1)
        nl = torch.index_select(yt_t, 0, flat).view(n, k, m)
        lc = nl.sum(dim=1)  # (n, m) label counts

        pos = yt_t.sum(dim=0)
        neg = n - pos
        prior_true = (s + pos) / (2.0 * s + n)

        cond_true = torch.zeros((m, k + 1), device=self._device)
        cond_false = torch.zeros((m, k + 1), device=self._device)
        lc_long = lc.long()
        yt_bool = yt_t.bool()

        for c in range(k + 1):
            mask_c = lc_long == c
            ct = (mask_c & yt_bool).sum(dim=0).float()
            cf = (mask_c & ~yt_bool).sum(dim=0).float()
            cond_true[:, c] = (s + ct) / (s * (k + 1) + pos)
            cond_false[:, c] = (s + cf) / (s * (k + 1) + neg)

        self._prior_true_t = prior_true
        self._prior_false_t = 1.0 - prior_true
        self._cond_true_t = cond_true
        self._cond_false_t = cond_false

    def _predict_torch(self, x_val: sparse.csr_matrix) -> np.ndarray:
        import torch

        xv_t = torch.from_numpy(x_val.toarray()).float().to(self._device)
        n_val = xv_t.shape[0]
        k = int(self.cfg.k)

        xv_norm = torch.nn.functional.normalize(xv_t, p=2, dim=1)
        xt_norm = torch.nn.functional.normalize(self._xt_train, p=2, dim=1)
        sim = torch.mm(xv_norm, xt_norm.t())

        target_k = min(k, self._xt_train.shape[0])
        _, indices = torch.topk(sim, k=target_k, dim=1)

        m = self._yt_train.shape[1]
        flat = indices.reshape(-1)
        nl = torch.index_select(self._yt_train, 0, flat).view(
            n_val, target_k, m
        )
        lc = nl.sum(dim=1).long()

        counts_t = lc.t()  # (m, n_val)
        pt_neigh = torch.gather(self._cond_true_t, 1, counts_t)
        pf_neigh = torch.gather(self._cond_false_t, 1, counts_t)

        prob_true = self._prior_true_t.unsqueeze(1) * pt_neigh
        prob_false = self._prior_false_t.unsqueeze(1) * pf_neigh
        probs = prob_true / (prob_true + prob_false + 1e-10)

        return probs.t().cpu().numpy()

    # ── Sklearn backend ───────────────────────────────────────────

    def _fit_sklearn(
        self,
        x_train: sparse.csr_matrix,
        y_train: sparse.csr_matrix,
    ) -> None:
        n, _ = x_train.shape
        m = y_train.shape[1]
        k = int(self.cfg.k)
        s = float(self.cfg.s)

        nn = NearestNeighbors(
            n_neighbors=min(k + 1, n),
            metric=self.cfg.metric,
            algorithm="brute",
        )
        nn.fit(x_train)
        neigh = nn.kneighbors(x_train, return_distance=False)
        if neigh.shape[1] > k:
            neigh = neigh[:, 1 : k + 1]

        y_dense = y_train.toarray().astype(np.int8, copy=False)
        # Vectorized neighbour label counts: (n, k, m) -> (n, m).
        if neigh.shape[1] > 0:
            label_counts = y_dense[neigh].sum(axis=1).astype(np.int16, copy=False)
        else:
            label_counts = np.zeros((n, m), dtype=np.int16)

        pos = y_dense.sum(axis=0).astype(np.int64)
        neg = (n - pos).astype(np.int64)

        self._prior_true = (s + pos) / (2.0 * s + n)
        self._prior_false = 1.0 - self._prior_true

        # Vectorized conditional counts: for each count value c in [0, k], tally
        # how many train samples with neighbour-count c are positive / negative,
        # per label. Equivalent to the original double loop (verified identical).
        lc_clip = np.clip(label_counts, 0, k)
        yt_bool = y_dense.astype(bool)
        ct = np.zeros((m, k + 1), dtype=np.float64)
        cf = np.zeros((m, k + 1), dtype=np.float64)
        for c in range(k + 1):
            mc = lc_clip == c
            ct[:, c] = (mc & yt_bool).sum(axis=0)
            cf[:, c] = (mc & ~yt_bool).sum(axis=0)
        self._cond_true = (s + ct) / (s * (k + 1) + pos[:, None])
        self._cond_false = (s + cf) / (s * (k + 1) + neg[:, None])

        self._nn = nn
        self._y_train = y_train

    def _predict_sklearn(self, x_val: sparse.csr_matrix) -> np.ndarray:
        assert self._nn is not None and self._y_train is not None
        k = int(self.cfg.k)
        m = self._y_train.shape[1]
        n = x_val.shape[0]

        neigh = self._nn.kneighbors(x_val, return_distance=False)
        if neigh.shape[1] > k:
            neigh = neigh[:, :k]

        y_dense = self._y_train.toarray().astype(np.int8, copy=False)
        if neigh.shape[1] == 0:
            return np.tile(self._prior_true, (n, 1))
        # Vectorized: neighbour label counts per (sample, label), then gather the
        # matching conditional probability. Equivalent to the original loop.
        counts = np.clip(y_dense[neigh].sum(axis=1), 0, k)  # (n, m)
        label_idx = np.arange(m)[None, :]
        ct_sel = self._cond_true[label_idx, counts]         # (n, m)
        cf_sel = self._cond_false[label_idx, counts]
        pt = self._prior_true[None, :] * ct_sel
        pf = self._prior_false[None, :] * cf_sel
        denom = pt + pf
        return np.where(denom == 0.0, 0.5, pt / np.where(denom == 0.0, 1.0, denom))

    # ── Public API ────────────────────────────────────────────────

    def fit(
        self,
        x_train: sparse.csr_matrix,
        y_train: sparse.csr_matrix,
    ) -> "MLkNNModel":
        self._backend_selected = self._select_backend(x_train)
        if self._backend_selected == "torch":
            self._fit_torch(x_train.tocsr(), y_train.tocsr())
        else:
            self._fit_sklearn(x_train.tocsr(), y_train.tocsr())
        return self

    def predict_proba(self, x: sparse.csr_matrix) -> np.ndarray:
        if self._backend_selected == "torch":
            return self._predict_torch(x.tocsr())
        return self._predict_sklearn(x.tocsr())
