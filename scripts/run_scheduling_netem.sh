#!/usr/bin/env bash
# OPTIONAL: empirical confirmation of the resource-aware scheduler under INJECTED tier heterogeneity on
# real nodes (cluster deliverable iii). Requires root / CAP_NET_ADMIN (tc/netem) + cgroup/cpulimit.
#
# A homogeneous SLURM cluster has no tier disparity or WAN latency, so to MEASURE the scheduler speed-up
# (rather than model it, as scripts/run_scheduling_study.py does) you inject it on real nodes:
#   - per-tier network delay with tc/netem  (edge = farther/slower link than cloud)
#   - per-tier compute throttling with cpulimit or a cgroup cpu.max cap  (edge = slower CPU)
# then run the per-client tasks and time the synchronous round (slowest surviving participant) under each
# policy (full / resource-aware quorum / deadline), as in run_scheduling_study.py.
#
# This is a TEMPLATE -- fill in <iface>, the cgroup/cpulimit caps, and your tier->node assignment.
set -euo pipefail
echo "Requires root/CAP_NET_ADMIN. Sketch (adapt to your cluster):"
cat <<'EOF'
TIERS:  edge = 20x slower / server = 5x / cloud = 1x   (round-robin over the K client tasks)

# inject, per client task, before running workflow/client_eval.py:
# edge tier (slow link + throttled CPU):
sudo tc qdisc add    dev <iface> root netem delay 60ms
cpulimit --limit 5  -- python3 workflow/client_eval.py --shard <edge_shard>   --mask <mask> --out e.json
# server tier:
sudo tc qdisc change dev <iface> root netem delay 30ms
cpulimit --limit 20 -- python3 workflow/client_eval.py --shard <server_shard> --mask <mask> --out s.json
# cloud tier (fast):
sudo tc qdisc change dev <iface> root netem delay 10ms
python3 workflow/client_eval.py --shard <cloud_shard>  --mask <mask> --out c.json
sudo tc qdisc del    dev <iface> root          # clean up

# round critical-path = max over PARTICIPATING tasks of (compute + injected delay); replay the policies
# (full / quorum / deadline) as in scripts/run_scheduling_study.py and compare MEASURED to the model.
EOF
echo "See scripts/run_scheduling_study.py (analytic model) and run_scheduling_empirical.py (no-root)."
