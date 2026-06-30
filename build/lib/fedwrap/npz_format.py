"""NPZ save helper (sparse CSR pairs)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import sparse


def save_csr_pair(
    path: str | Path,
    x: sparse.csr_matrix,
    y: sparse.csr_matrix,
) -> None:
    """Persist a feature-matrix / label-matrix pair in compressed NPZ."""
    x = x.tocsr()
    y = y.tocsr()
    x.sum_duplicates()
    y.sum_duplicates()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        X_data=x.data,
        X_indices=x.indices,
        X_indptr=x.indptr,
        X_shape=np.array(x.shape, dtype=np.int64),
        Y_data=y.data,
        Y_indices=y.indices,
        Y_indptr=y.indptr,
        Y_shape=np.array(y.shape, dtype=np.int64),
    )
