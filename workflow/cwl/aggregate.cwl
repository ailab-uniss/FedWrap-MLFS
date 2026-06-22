#!/usr/bin/env cwl-runner
cwlVersion: v1.2
class: CommandLineTool
label: "FedWrap-MLFS server step: sum per-client counters into the exact global micro/macro-F1"

requirements:
  DockerRequirement:
    dockerImageId: fedwrap-workflow:latest

baseCommand: [python, /app/workflow/aggregate.py]
arguments: ["--out", "global.json"]

inputs:
  counters:
    type: File[]
    inputBinding: {prefix: --counters}

outputs:
  global_metrics:
    type: File
    outputBinding: {glob: global.json}
