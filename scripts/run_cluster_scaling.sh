#!/usr/bin/env bash
# Strong/weak scaling driver for the HPC-cluster section (deliverable i): run ONE federated evaluation
# round of the workflow at increasing client counts K, recording wall-clock time and throughput.
# Uses a synthetic federation per K (no real data needed). Writes reports/cluster/scaling.csv.
#
# Locally this times the cwltool run on one node. On the cluster, replace the cwltool line with
#   streamflow run workflow/streamflow-slurm.yml
# so per-client tasks distribute across SLURM nodes (then also vary node count for strong/weak scaling).
set -euo pipefail
KS="${1:-4 8 16 32}"; D="${2:-300}"; N="${3:-4000}"
OUT=reports/cluster; mkdir -p "$OUT"
echo "n_clients,seconds,throughput_per_s,macro_f1" > "$OUT/scaling.csv"
for K in $KS; do
  name="scale_K${K}_D${D}"
  python3 - "$name" "$N" "$D" "$K" <<'PY'
import sys; sys.path.insert(0,'scripts')
import numpy as np
from synth_scaling import materialize
name,N,D,K = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
materialize(f'data/{name}', N=N, D=D, L=12, K=K, informative_ratio=0.1, noise=0.1, alpha=1.0,
            seed=0, n_informative=14, interaction_frac=0.0, signal_strength=8.0)
m=np.zeros(D,bool); m[np.load(f'data/{name}/true_informative.npy')]=True
np.save('workflow/mask_scale.npy', m)
PY
  python3 workflow/prepare_shards.py --dataset "$name" --root data --out workflow/shards >/dev/null
  python3 workflow/make_job.py "workflow/shards/$name" workflow/mask_scale.npy > "workflow/job_$name.yml"
  t0=$(date +%s.%N)
  cwltool --outdir "out_$name" workflow/cwl/fed_eval.cwl "workflow/job_$name.yml" >/dev/null 2>&1
  t1=$(date +%s.%N)
  python3 - "$K" "$t0" "$t1" "out_$name/global.json" "$OUT/scaling.csv" <<'PY'
import sys, json
K=int(sys.argv[1]); secs=float(sys.argv[3])-float(sys.argv[2])
mac=json.load(open(sys.argv[4]))['macro_f1']
line=f"{K},{secs:.2f},{1.0/secs if secs>0 else 0:.3f},{mac:.4f}"
open(sys.argv[5],'a').write(line+"\n"); print(line)
PY
  rm -rf "out_$name" "data/$name" "workflow/shards/$name" "workflow/job_$name.yml"
done
echo "wrote $OUT/scaling.csv"
