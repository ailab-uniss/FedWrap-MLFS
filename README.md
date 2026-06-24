# FedWrap-MLFS

A **federated wrapper** for multi-objective **multi-label feature selection**, cast as a scientific
workflow over the compute continuum. A central evolutionary optimizer (FedAware-NSGA-II) orchestrates
many distributed wrapper evaluations across data-owning silos; each client trains a local multi-label
ML-kNN on the selected features and returns **only** label-wise TP/FP/FN counters, from which the
server reconstructs the **exact** global micro/macro-F1. Each round exchanges only a feature mask and
integer counters --- never raw data, predictions, or gradients.

Code accompanying the paper *"FedWrap-MLFS: A Federation-Aware Wrapper Workflow for Multi-Objective
Multi-Label Feature Selection in the Compute Continuum."*

## What's here

| path | contents |
|---|---|
| `fedwrap/` | the framework: federated evaluator, FedAware-NSGA-II (relevance warm start + guided mutation + client-stability), filters, metrics, scheduling |
| `fedwrap/federated/` | federated evaluation protocol, Flower backend, resource-aware scheduling |
| `workflow/` | the executable CWL workflow (one federated evaluation round) + Docker/StreamFlow configs (see `workflow/README.md`) |
| `scripts/` | experiment runners (real datasets, synthetic D-law, ablation, scheduling, performance) |
| `data/fed_real/` | natural-silo federations (ECG, eICU, ExtraSensory), obtained separately; synthetic via `scripts/synth_scaling.py` |

## Setup

```bash
pip install -e .            # or: pip install -r requirements.txt
```
Core dependencies: numpy, scipy, scikit-learn, pyyaml (see `requirements.txt`). Optional extras:
`pip install -e .[gpu]` (GPU ML-kNN), `.[workflow]` (cwltool/StreamFlow), `.[data]` (dataset loaders).

## Data

A small **synthetic demo federation is bundled** (`data/synth_demo/`, with prebuilt workflow shards in
`workflow/shards/synth_demo/`) so the workflow demo runs straight from a clean clone (see
`workflow/README.md`). The **paper datasets are not committed**: the synthetic sweeps are regenerated on
the fly by `scripts/synth_scaling.py` / `scripts/run_interaction_campaign.py`, and the three real
federations are derived from external sources and must be obtained separately, then placed under
`data/fed_real/<dataset>/fold0/` as `trainval.npz`, `test.npz`, `trainval_groups.npy`,
`test_groups.npy`:

- **ECG** — PhysioNet/Computing in Cardiology 2021 (public).
- **ExtraSensory** — public.
- **eICU-CRD** — PhysioNet *credentialed* access (CITI training + signed Data Use Agreement); each
  user with access must be individually credentialed, and the data must not be redistributed.

## Reproduce the main results

```bash
# real federations, one method/seed at a time (the operator portfolio = "faport"; the headline method)
python scripts/run_reals_method.py eICU_expl_k12 faport 0,1,2,3,4,5,6,7,8,9

# 10-seed main table (FedWrap vs base vs filters, with Wilcoxon)
python scripts/build_maintable_warm.py        # -> reports/maintable_warm.csv

# synthetic scaling law in feature dimension D (base vs FedAware)
python scripts/run_synth_scaling.py D 0,1,2,3,4,5,6,7,8,9 fedaware
python scripts/run_synth_scaling.py D 0,1,2,3,4,5,6,7,8,9 base

# resource-aware scheduling study; workflow throughput
python scripts/run_scheduling_study.py
python scripts/run_perf.py
```

## Run the federated workflow (local or HPC)

The distributable federated-evaluation round is packaged as a portable CWL workflow that runs
unchanged locally (Docker) and on an HPC cluster (Singularity + SLURM). See **`workflow/README.md`**
for the validated `cwltool` run and the StreamFlow deployment bindings. An HPC run produces result
files (per-round `global.json` outputs plus StreamFlow timing/provenance) that can be collected for
downstream analysis.
