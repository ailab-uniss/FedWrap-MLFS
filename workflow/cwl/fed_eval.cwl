#!/usr/bin/env cwl-runner
cwlVersion: v1.2
class: Workflow
label: "One FedWrap-MLFS federated evaluation round: scatter a mask to client silos, gather counters, aggregate exactly"
doc: |
  The distributable unit of the FedWrap-MLFS workflow (stages S2--S3 of the paper): the server
  broadcasts one feature mask; each client silo trains a local ML-kNN on the selected features and
  returns only its label-wise TP/FP/FN counters; the server sums them into the exact global
  micro/macro-F1. The CWL is identical for local (Docker) and HPC (Singularity/SLURM) execution;
  only the StreamFlow deployment binding changes.

requirements:
  ScatterFeatureRequirement: {}

inputs:
  shards:
    type: File[]
    doc: "one local data shard per client silo"
  client_ids:
    type: int[]
    doc: "client identifiers, aligned with shards"
  mask:
    type: File
    doc: "the broadcast feature mask (.npy boolean vector)"
  k:
    type: int
    default: 5
  s:
    type: float
    default: 1.0

outputs:
  global_metrics:
    type: File
    outputSource: aggregate/global_metrics

steps:
  client_eval:
    run: client_eval.cwl
    scatter: [shard, client_id]
    scatterMethod: dotproduct
    in:
      shard: shards
      client_id: client_ids
      mask: mask
      k: k
      s: s
    out: [counters]
  aggregate:
    run: aggregate.cwl
    in:
      counters: client_eval/counters
    out: [global_metrics]
