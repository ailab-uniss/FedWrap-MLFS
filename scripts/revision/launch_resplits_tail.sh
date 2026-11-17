#!/bin/bash
# Auxiliary queue: base NSGA-II (bitstring) on the last partitions, to shorten the wall-clock time of launch_resplits.sh.
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for r in 10 9; do
  for ds in ECG_cinc2021 eICU_expl_k12 ExtraSensory; do
    .venv/bin/python scripts/revision/run_resplit_method.py "${ds}__r${r}" bitstring "$r" >> logs/revision/resplit_qC.log 2>&1
  done
done
