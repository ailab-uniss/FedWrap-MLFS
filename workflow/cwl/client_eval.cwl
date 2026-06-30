#!/usr/bin/env cwl-runner
cwlVersion: v1.2
class: CommandLineTool
label: "FedWrap-MLFS client step: train local ML-kNN on selected features, return TP/FP/FN counters"

# DockerRequirement is a HINT, not a hard requirement: with a container runtime the step runs in the
# image; with `cwltool --no-container` (or a StreamFlow deployment without a container) it runs the
# `fedwrap-client-eval` console script from a plain conda/venv (`pip install .`). Same command both ways.
hints:
  DockerRequirement:
    dockerImageId: fedwrap-workflow:latest

baseCommand: [fedwrap-client-eval]
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
