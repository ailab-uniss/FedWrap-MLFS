#!/usr/bin/env cwl-runner
cwlVersion: v1.2
class: CommandLineTool
label: "FedWrap-MLFS server step: sum per-client counters into the exact global micro/macro-F1"

# DockerRequirement is a HINT (see client_eval.cwl): runs in the image when a container runtime is
# present, or via the `fedwrap-aggregate` console script (`pip install .`) under `--no-container`.
hints:
  DockerRequirement:
    dockerImageId: fedwrap-workflow:latest

baseCommand: [fedwrap-aggregate]
arguments: ["--out", "global.json"]

inputs:
  counters:
    type: File[]
    inputBinding: {prefix: --counters}

outputs:
  global_metrics:
    type: File
    outputBinding: {glob: global.json}
