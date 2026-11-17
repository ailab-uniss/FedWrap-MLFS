#!/bin/bash
# Reviewer 2, comment 3: FedWrap-MLFS (faport) and base NSGA-II (bitstring) on ten independent
# train/test resplits; split r uses search seed r. Two worker queues (8 client threads each).
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p logs/revision
q() {  # $1 = queue id; remaining = list of "DS METHOD r"
  local id=$1; shift
  for job in "$@"; do
    set -- $job
    .venv/bin/python scripts/revision/run_resplit_method.py "$1__r$3" "$2" "$3" >> logs/revision/resplit_q$id.log 2>&1
  done
}
A=(); B=()
for r in 1 2 3 4 5 6 7 8 9 10; do
  for ds in ECG_cinc2021 eICU_expl_k12 ExtraSensory; do
    A+=("$ds faport $r"); B+=("$ds bitstring $r")
  done
done
q A "${A[@]}" &
q B "${B[@]}" &
wait
echo ALLDONE >> logs/revision/resplit_qA.log
