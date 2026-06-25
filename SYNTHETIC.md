# Exact synthetic data specification

The synthetic federations used in the paper are **not shipped as files**: their generation is
**byte-deterministic**, so the exact parameters below regenerate them *identically*. Re-generating
`int_D300_f0.3_s2` with these parameters reproduces the shipped reference bit-for-bit (verified by
SHA-256). One command regenerates everything:

```bash
bash scripts/reproduce_paper_synthetic.sh
```

The rest of this file is the explicit recipe, in case you want to generate any point by hand.

## Generator

All data come from `make_synth(...)` in [`scripts/synth_scaling.py`](scripts/synth_scaling.py), wrapped
by `materialize(path, ...)` which writes a `fold0/` prefold (`trainval.npz`, `test.npz`,
`*_groups.npy`, `true_informative.npy`). Randomness is **`numpy.random.default_rng(seed)`** (PCG64),
whose stream is stable across NumPy versions — hence the byte-for-byte reproducibility.

```python
from synth_scaling import materialize
materialize(out_dir, N=..., D=..., L=..., K=..., informative_ratio=..., noise=...,
            alpha=..., n_informative=..., seed=..., interaction_frac=..., signal_strength=...)
```

### Fixed for every study (generator defaults, never overridden)

| parameter | value | meaning |
|---|---|---|
| `labels_per_feature` | `3` | each informative feature drives ≤3 labels |
| `target_card` | `2.5` | mean labels per instance (sparse multi-label) |
| `noise_scale` | `0.5` | scale of additive observation noise |
| `test_frac` | `0.25` | per-client train/test split |
| RNG | `default_rng(seed)` | PCG64, version-stable |

## The four studies (exact parameters, grids, seeds)

### 1. Scaling law in feature dimension `D`  → `reports/synth_scaling_D_{base,fedaware}.csv`
Driver: `scripts/run_synth_scaling.py D <seeds> {base|fedaware}` (with `SC_GRID="100,500,1000,2000,5000"`).

| N | L | K | informative_ratio | noise | alpha | n_informative | interaction_frac | signal_strength |
|---|---|---|---|---|---|---|---|---|
| 4000 | 20 | 8 | 0.10 | 0.3 | 0.5 | 30 | 0.0 | 5.0 |

- **swept**: `D ∈ {100, 500, 1000, 2000, 5000}`
- **seeds**: `0..9` (10)  ·  **arms**: `base` and `fedaware`

### 2. Heterogeneity sweep in label skew `alpha`  → `reports/fedaware_synth_alpha.csv`
Driver: `scripts/run_fedaware_synth.py alpha <seeds>`.

| N | D | L | K | informative_ratio | noise | n_informative | interaction_frac | signal_strength |
|---|---|---|---|---|---|---|---|---|
| 4000 | 300 | 20 | 8 | 0.10 | 0.3 | 30 | 0.0 | 5.0 |

- **swept**: `alpha ∈ {10.0, 1.0, 0.3, 0.1}`  (large = IID, small = strong non-IID)
- **seeds**: `0..4` (5)

### 3. Heterogeneity sweep in client count `K`  → `reports/fedaware_synth_K.csv`
Driver: `scripts/run_fedaware_synth.py K <seeds>`.

| N | D | L | informative_ratio | noise | alpha | n_informative | interaction_frac | signal_strength |
|---|---|---|---|---|---|---|---|---|
| 4000 | 300 | 20 | 0.10 | 0.3 | 0.5 | 30 | 0.0 | 5.0 |

- **swept**: `K ∈ {2, 4, 8, 16, 32}`
- **seeds**: `0..4` (5)

### 4. Feature-interaction campaign  → `reports/interaction_sweep.csv`
Driver: `scripts/run_interaction_campaign.py <interaction_frac> <seeds>`.

| N | D | L | K | informative_ratio | noise | alpha | n_informative | signal_strength |
|---|---|---|---|---|---|---|---|---|
| 4000 | 300 | 12 | 8 | 0.10 | 0.1 | 1.0 | 16 | 8.0 |

- **swept**: `interaction_frac ∈ {0.0, 0.3, 0.5, 0.7}`  (fraction of label signal from feature *pairs*)
- **seeds**: `0..4` (5)

> Note: a few stale `data/synth_scaling/sc_*`/`synth_het_*` directories may exist locally from earlier
> exploration with an older default (`D=1000`); they are **not** the paper's data. The reports above
> (and this spec) use the parameters listed here.
