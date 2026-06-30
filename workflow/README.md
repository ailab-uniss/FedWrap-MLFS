# FedWrap-MLFS federated-evaluation workflow

This directory packages the distributable unit of FedWrap-MLFS --- one **federated evaluation round**
(stages S2--S3 of the paper) --- as a portable [CWL](https://www.commonwl.org/) workflow. The server
broadcasts a feature mask; each client silo trains a local ML-kNN on the selected features and returns
**only** its label-wise TP/FP/FN counters; the server sums them into the exact global micro/macro-F1.
The same workflow runs locally (Docker) and on an HPC cluster (Singularity + SLURM); only the
StreamFlow deployment binding changes.

## Validation

The containerized CWL workflow reproduces the in-process federated aggregation **exactly**. On the
8-silo ECG federation with a 15-feature mask:

| | macro-F1 | micro-F1 |
|---|---|---|
| in-process (`fedwrap`) | 0.339178 | 0.520903 |
| CWL workflow (cwltool + Docker) | 0.339178 | 0.520903 |

(Counters are integer sums, so the aggregate is exact for whichever clients respond --- the property
the resource-aware scheduler also relies on.)

## Files

| file | role |
|---|---|
| `cwl/client_eval.cwl` | client step: local ML-kNN on selected features -> TP/FP/FN counters |
| `cwl/aggregate.cwl` | server step: sum counters -> exact global micro/macro-F1 |
| `cwl/fed_eval.cwl` | workflow: scatter the mask to client silos, gather, aggregate |
| `client_eval.py`, `aggregate.py`, `shard_io.py` | the step implementations (reuse the validated `fedwrap` protocol) |
| `prepare_shards.py` | split a federation into one local data shard per silo |
| `make_job.py` | emit a CWL input job from a shard directory + a mask |
| `Dockerfile` | the step container (Docker locally; Singularity on HPC) |
| `streamflow-local.yml` | StreamFlow binding: run locally in Docker |
| `streamflow-slurm.yml.template` | StreamFlow binding: run on SLURM + Singularity (fill placeholders) |

## Run locally

A small **synthetic federation is bundled** (`data/synth_demo/`, 8 silos, $D{=}100$, with feature
interactions) together with prebuilt shards in `workflow/shards/synth_demo/`, so this demo runs
straight from a clean clone --- no data download.

```bash
# 1. build the step image
docker build -f workflow/Dockerfile -t fedwrap-workflow:latest .

# 2. build the CWL job (bundled demo mask = the true informative subset)
python workflow/make_job.py workflow/shards/synth_demo workflow/mask_synth_demo.npy > workflow/job_demo.yml

# 3a. run with cwltool (reference CWL runner --- this is the validated path)
cwltool --outdir out workflow/cwl/fed_eval.cwl workflow/job_demo.yml
cat out/global.json          # exact global micro/macro-F1 aggregated across silos

# 3b. or run with StreamFlow using the local Docker binding
streamflow run workflow/streamflow-local.yml
```

To run on your **own dataset** instead, place a prefold under `data/fed_real/<name>/` and build its
shards (synthetic or public data only --- not eICU on shared infra):
`python workflow/prepare_shards.py --dataset <name> --root data/fed_real --out workflow/shards`.

## Run on an HPC cluster

The cluster needs **SLURM** (or PBS/Flux) and a **shared filesystem** visible to the compute nodes
(StreamFlow stages the shard/mask/counter files through it — no live inter-node networking required).
A container runtime is **optional**: the workflow steps are pure Python and the CWL's
`DockerRequirement` is a *hint*, so you can run **with no container at all** or with Apptainer.

**No Docker on the cluster?** That is the norm — Docker needs a root daemon. You have two Docker-free
paths; the first needs nothing but the Python that is already on every cluster.

### Option A — no container (simplest; verified)

The steps install as console scripts (`fedwrap-client-eval` / `fedwrap-aggregate`), so they run in a
plain conda/venv. Verified end-to-end: `cwltool --no-container` reproduces the in-process simulator to
the bit (max |Δ| = 0).

```bash
conda create -n fedwrap python=3.10 -y && conda activate fedwrap   # or python -m venv
pip install .                                                       # deps + the two console scripts

# local check (no container, no cluster):
cwltool --no-container --outdir out workflow/cwl/fed_eval.cwl job.yml && cat out/global.json

# on the cluster (StreamFlow over SLURM, no container layer):
cp workflow/streamflow-slurm-nocontainer.yml.template workflow/streamflow-slurm-nocontainer.yml
streamflow run workflow/streamflow-slurm-nocontainer.yml   # ensure the conda env is active for the jobs
```

### Option B — Apptainer/Singularity (reproducible image, still no Docker)

For a bit-pinned, portable image, build the `.sif` from [`fedwrap.def`](fedwrap.def) (pulls
`python:3.10-slim` and `pip install .` — no Docker daemon). Three routes, easiest first:

1. **`apptainer build --fakeroot fedwrap-workflow.sif workflow/fedwrap.def`** — on the cluster, if your
   admin enabled rootless `--fakeroot` (most modern clusters do).
2. **Build elsewhere, copy the file.** If `--fakeroot` is disabled, run the same `apptainer build` on
   any machine where you have it, then `scp fedwrap-workflow.sif <cluster>:<path>/`. *Running* a `.sif`
   is unprivileged, so this always works.
3. **Pull from a registry** (if the image is published): `apptainer pull fedwrap-workflow.sif
   docker://ghcr.io/ailab-uniss/fedwrap-workflow:latest` — no Docker, no build, no root.

Then `cp workflow/streamflow-slurm.yml.template workflow/streamflow-slurm.yml` (point `image:` at the
`.sif`) and `streamflow run`. `apptainer` and `singularity` are CLI-compatible; use whichever exists.

The full evolutionary search (thousands of rounds) is driven by the fast in-process simulator, whose
aggregation is numerically identical to this workflow (verified to 1e-9); the workflow is the
deployment-realistic, portable form of each federated evaluation round.

> Data note: eICU shards derive from PhysioNet credentialed data and may be used on the cluster
> provided every person with access is individually PhysioNet-credentialed (CITI + DUA) and the files
> are restricted to credentialed users (not world-readable); do not redistribute. ECG, ExtraSensory,
> and synthetic data carry no such restriction.
