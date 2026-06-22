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

```bash
# 1. build the step image
docker build -f workflow/Dockerfile -t fedwrap-workflow:latest .

# 2. materialize per-silo shards (use synthetic or public datasets on shared infra; not eICU)
python workflow/prepare_shards.py --dataset ECG_cinc2021 --root data/fed_real --out workflow/shards

# 3. pick a mask (any boolean .npy of length D) and build the CWL job
python workflow/make_job.py workflow/shards/ECG_cinc2021 workflow/mask.npy > workflow/job_ecg.yml

# 4a. run with cwltool (reference CWL runner --- this is the validated path)
cwltool --outdir out workflow/cwl/fed_eval.cwl workflow/job_ecg.yml
cat out/global.json

# 4b. or run with StreamFlow using the local Docker binding
streamflow run workflow/streamflow-local.yml
```

## Run on an HPC cluster (Folino)

The cluster needs **SLURM** (or PBS/Flux), **Singularity/Apptainer**, and a **shared filesystem**
visible to the compute nodes. No live inter-node networking is required: StreamFlow stages the
shard/mask/counter files through the shared filesystem.

```bash
# build the Singularity image from the same Docker image
singularity build fedwrap-workflow.sif docker-daemon://fedwrap-workflow:latest

# fill <PLACEHOLDERS> (login host, user, partition, account, image path) in the template
cp workflow/streamflow-slurm.yml.template workflow/streamflow-slurm.yml
streamflow run workflow/streamflow-slurm.yml
```

The full evolutionary search (thousands of rounds) is driven by the fast in-process simulator, whose
aggregation is numerically identical to this workflow (verified to 1e-9); the workflow is the
deployment-realistic, portable form of each federated evaluation round.

> Data note: eICU shards derive from PhysioNet credentialed data and may be used on the cluster
> provided every person with access is individually PhysioNet-credentialed (CITI + DUA) and the files
> are restricted to credentialed users (not world-readable); do not redistribute. ECG, ExtraSensory,
> and synthetic data carry no such restriction.
