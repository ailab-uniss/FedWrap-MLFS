#!/bin/bash
# Complete searches on the Flower/Ray runtime, compared with the in-process runs of the same split/seed.
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
R=runs/revision/resplit
for job in "ECG_cinc2021__r1 faport 1 mask" "eICU_expl_k12__r1 faport 1 generation" "ExtraSensory__r1 faport 1 generation" "ECG_cinc2021__r1 bitstring 1 generation"; do
  set -- $job
  .venv/bin/python scripts/revision/run_flower_search.py $1 $2 $3 $4 $R/$1_$2_s$3_fold0 > logs/revision/flower_$1_$2_$4.log 2>&1
done
echo FLOWERDONE > logs/revision/flower_done.flag
