#!/bin/bash
# usage: cache_worker.sh <dataset> <split> [<split> ...] ; waits for both searches of a split, then caches its table row
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ds=$1; shift
for r in "$@"; do
  until [ -f runs/revision/resplit/${ds}__r${r}_faport_s${r}_fold0/summary.json ] && [ -f runs/revision/resplit/${ds}__r${r}_bitstring_s${r}_fold0/summary.json ]; do sleep 30; done
  .venv/bin/python scripts/revision/build_resplit_table.py --one $ds $r >> logs/revision/cache_${ds}.log 2>&1
done
echo "WORKERDONE $ds $*" >> logs/revision/cache_done.log
