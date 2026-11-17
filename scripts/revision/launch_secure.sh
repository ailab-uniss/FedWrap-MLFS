#!/bin/bash
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for ds in eICU_expl_k12__r1 ExtraSensory__r1; do
  .venv/bin/python scripts/revision/run_secure_e2e.py $ds 1 runs/revision/resplit/${ds}_faport_s1_fold0 > logs/revision/secure_$ds.log 2>&1
done
echo SECUREDONE > logs/revision/secure_done.flag
