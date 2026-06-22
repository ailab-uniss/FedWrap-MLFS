"""Parametrizable synthetic FEDERATED multi-label benchmark for the controlled scalability study.

A clean generative model where feature selection is meaningful and federation properties can be
varied independently of the real datasets:
  - n_inf = informative_ratio * D informative features carry all label signal; the rest are pure
    noise (so selecting the informative subset helps and noise dilutes a nearest-neighbour metric).
  - each label l is a sparse linear function of a random subset of informative features; a per-label
    bias sets prevalence to give SPARSE multi-label targets.
  - features observed with additive Gaussian noise of scale `noise`.
  - clients/silos are partitioned with Dirichlet(alpha) label skew: alpha large -> IID,
    alpha small -> strong non-IID label skew. Each client is split train/test.

Returns CSR X/Y and per-row client-group vectors, plus the true informative-feature indices.
"""
from __future__ import annotations
import numpy as np
from scipy import sparse


def make_synth(N=10000, D=1000, L=20, K=8, informative_ratio=0.10, noise=0.3,
               alpha=1.0, labels_per_feature=3, target_card=2.5, signal_strength=5.0,
               noise_scale=0.5, n_informative=None, test_frac=0.25, seed=0):
    rng = np.random.default_rng(seed)
    # n_informative (absolute) fixes the informative subspace independent of D: for the D-sweep this
    # keeps the kNN-friendly signal fixed while only noise features grow. If None, fall back to ratio.
    n_inf = int(n_informative) if n_informative is not None else max(L, int(round(informative_ratio * D)))
    n_inf = min(n_inf, D)
    inf_idx = rng.choice(D, size=n_inf, replace=False)
    noise_idx = np.setdiff1d(np.arange(D), inf_idx)

    # label weight matrix W (L x D), nonzero only on informative features. Every label depends on
    # the SHARED informative subspace (dense weights over inf_idx) so a single nearest-neighbour
    # distance over those features serves all labels -- the regime ML-kNN is designed for. The
    # pure-noise features are what dilutes the distance and motivates selection.
    W = np.zeros((L, D), dtype=np.float32)
    W[:, inf_idx] = rng.normal(0, 1.0, size=(L, n_inf)).astype(np.float32)

    # latent clean features drive the labels; observed features add noise
    Xc = rng.normal(0, 1.0, size=(N, D)).astype(np.float32)
    S = Xc @ W.T                                    # (N, L) label scores
    S /= (S.std(axis=0, keepdims=True) + 1e-6)
    S *= float(signal_strength)                     # difficulty: lower -> noisier labels (oracle macro down)
    # per-label bias to hit a target average cardinality (sparse multi-label)
    bias = np.full(L, 0.0, dtype=np.float32)
    # binary-search a single global offset so mean labels/instance ~= target_card
    lo, hi = -6.0, 6.0
    for _ in range(40):
        mid = (lo + hi) / 2
        card = (1 / (1 + np.exp(-(S + mid)))).mean(axis=1).mean() * L
        if card > target_card:
            hi = mid
        else:
            lo = mid
    bias[:] = (lo + hi) / 2
    P = 1.0 / (1.0 + np.exp(-(S + bias)))
    Y = (rng.random(P.shape) < P).astype(np.int8)   # logistic sampling already injects label stochasticity
    # drop all-zero rows by forcing their top label on
    empty = Y.sum(axis=1) == 0
    if empty.any():
        Y[empty, np.argmax(S[empty] + bias, axis=1)] = 1

    # observed features: informative carry signal+noise; pure-noise features are down-scaled so
    # all-features is degraded (kNN distance dilution) but not catastrophically zero.
    Xobs = Xc + noise * rng.normal(0, 1.0, size=Xc.shape).astype(np.float32)
    if noise_idx.size:
        Xobs[:, noise_idx] *= float(noise_scale)

    # Dirichlet(alpha) label-skew client assignment
    pref = rng.dirichlet(np.full(L, alpha), size=K)          # (K, L) client label preferences
    aff = Y @ pref.T + 1e-3                                  # (N, K) affinity
    aff /= aff.sum(axis=1, keepdims=True)
    groups = np.array([rng.choice(K, p=aff[i]) for i in range(N)])

    # per-client train/test split
    is_test = np.zeros(N, bool)
    for k in range(K):
        idx = np.flatnonzero(groups == k)
        if idx.size == 0:
            continue
        nte = max(1, int(round(test_frac * idx.size)))
        te = rng.choice(idx, size=nte, replace=False)
        is_test[te] = True
    tr = ~is_test
    Xtr = sparse.csr_matrix(Xobs[tr]); Ytr = sparse.csr_matrix(Y[tr])
    Xte = sparse.csr_matrix(Xobs[is_test]); Yte = sparse.csr_matrix(Y[is_test])
    return Xtr, Ytr, groups[tr], Xte, Yte, groups[is_test], np.sort(inf_idx)


def materialize(path, N, D, L, K, informative_ratio, noise, alpha, seed, n_informative=None):
    """Write a prefold dataset (fold0) consumable by run_experiment_from_config."""
    from pathlib import Path
    Xtr, Ytr, gtr, Xte, Yte, gte, inf = make_synth(N=N, D=D, L=L, K=K,
        informative_ratio=informative_ratio, noise=noise, alpha=alpha,
        n_informative=n_informative, seed=seed)
    out = Path(path) / "fold0"; out.mkdir(parents=True, exist_ok=True)

    def save(p, x, y):
        x = x.tocsr(); y = y.tocsr()
        np.savez(p, X_data=x.data, X_indices=x.indices, X_indptr=x.indptr, X_shape=np.array(x.shape),
                 Y_data=y.data, Y_indices=y.indices, Y_indptr=y.indptr, Y_shape=np.array(y.shape))
    save(out / "trainval.npz", Xtr, Ytr); save(out / "test.npz", Xte, Yte)
    np.save(out / "trainval_groups.npy", gtr); np.save(out / "test_groups.npy", gte)
    np.save(Path(path) / "true_informative.npy", inf)
    return Xtr.shape, Yte.shape, len(set(gtr.tolist()))


if __name__ == "__main__":
    # sanity check
    Xtr, Ytr, gtr, Xte, Yte, gte, inf = make_synth(N=4000, D=500, L=20, K=8,
        informative_ratio=0.1, noise=0.3, alpha=1.0, seed=0)
    print("Xtr", Xtr.shape, "Xte", Xte.shape, "n_inf", inf.size)
    print("mean labels/inst (train)", Ytr.sum() / Xtr.shape[0])
    print("label prevalence range", Ytr.toarray().mean(0).min().round(3), Ytr.toarray().mean(0).max().round(3))
    print("clients", sorted(set(gtr.tolist())), "sizes", np.bincount(gtr))
    # IID vs non-IID skew check
    for a in [10.0, 0.1]:
        _, Y2, g2, *_ = make_synth(N=4000, D=500, L=20, K=8, alpha=a, seed=0)
        Yd = Y2.toarray()
        perclient = np.array([Yd[g2 == k].mean(0) for k in range(8) if (g2 == k).any()])
        skew = perclient.std(0).mean()
        print(f"alpha={a}: cross-client label-prevalence std (skew)={skew:.4f}")
