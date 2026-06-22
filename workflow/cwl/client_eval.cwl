#!/usr/bin/env cwl-runner
cwlVersion: v1.2
class: CommandLineTool
label: "FedWrap-MLFS client step: train local ML-kNN on selected features, return TP/FP/FN counters"

requirements:
  DockerRequirement:
    dockerImageId: fedwrap-workflow:latest

baseCommand: [python, /app/workflow/client_eval.py]
arguments: ["--out", "counters.json"]

inputs:
  shard:
    type: File
    inputBinding: {prefix: --shard}
  mask:
    type: File
    inputBinding: {prefix: --mask}
  client_id:
    type: int
    default: 0
    inputBinding: {prefix: --client-id}
  k:
    type: int
    default: 5
    inputBinding: {prefix: --k}
  s:
    type: float
    default: 1.0
    inputBinding: {prefix: --s}

outputs:
  counters:
    type: File
    outputBinding: {glob: counters.json}
