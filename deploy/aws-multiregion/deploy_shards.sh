#!/usr/bin/env bash
# Put one local data shard on each silo (its private data) + the broadcast mask + the resident worker.
# cloud <- client_0, server <- client_1, edge <- client_2  (from the repo's bundled synth_demo, D=100).
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$DIR/../.." && pwd)"
KEY="$HOME/.ssh/fedwrap-configb"
SRC="$REPO/workflow/shards/synth_demo"                  # per-silo shards
MASK="$REPO/workflow/mask_synth_demo.npy"               # broadcast feature mask
O=(-i "$KEY" -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=10)
declare -A SHARD=( [cloud]=client_0.npz [server]=client_1.npz [edge]=client_2.npz )
while read -r tier region iid ip; do
  echo "=== $tier ($ip) <- ${SHARD[$tier]} + mask + worker ==="
  ssh -n "${O[@]}" "ubuntu@$ip" 'mkdir -p ~/fed'
  scp "${O[@]}" "$SRC/${SHARD[$tier]}"    "ubuntu@$ip:~/fed/shard.npz"
  scp "${O[@]}" "$MASK"                   "ubuntu@$ip:~/fed/mask.npy"
  scp "${O[@]}" "$DIR/silo_worker.py"     "ubuntu@$ip:~/fed/silo_worker.py"
done < "$DIR/state.tsv"
echo "shard + mask + worker deployed to all silos"
