# Multi-region AWS test of the resource-aware scheduler (real WAN latency)

This deploys the FedWrap-MLFS federated evaluation round across **three AWS regions** (three tiers:
`cloud` / `server` / `edge`) and measures the **resource-aware scheduler's speed-up** under **real
wide-area latency** — no `netem` injection. Each silo runs the workflow's own per-client evaluator
(`fedwrap-client-eval`, wrapped here in a persistent worker that models a Flower client).

## What it shows (our run: Milan / Frankfurt / Tokyo)

| tier | region | per-round time |
|---|---|---|
| cloud | eu-south-1 (Milan) | ~30 ms |
| server | eu-central-1 (Frankfurt) | ~41 ms |
| edge | ap-northeast-1 (Tokyo) | ~241 ms |

- **full participation** (wait for the far Tokyo edge): **~240 ms/round**
- **resource-aware quorum=2** (proceed without the straggler): **~40 ms/round**
- **critical-path speed-up: 6.2×** (mean over 120 rounds = three independent 40-round sessions; ±0.6×),
  with the aggregation still **exact** over the responders (global macro-F1 0.3918 on all 3 silos vs
  0.3773 on the fastest 2 — Prop. 1). `run_round.py` prints per-tier / full / quorum mean ± std.

This is the paper's claim — *resource-aware partial participation cuts critical-path time with limited
quality loss* — measured on real geographically distributed nodes.

## Prerequisites

- An AWS account with **AWS CLI v2 configured** (`aws sts get-caller-identity` must succeed).
- SSH client (`ssh`, `scp`) and Python 3 with `numpy` on the machine you run this from (the orchestrator).
- **vCPU quota:** three `t3.micro` = 6 vCPU. New accounts sometimes cap On-Demand at 5 vCPU — if
  `launch.sh` fails with a quota error, request an increase (Service Quotas → EC2 → *Running On-Demand
  Standard instances*) or set `TYPE=t2.micro` (1 vCPU) in `launch.sh`.

## Run it (from the repo root)

```bash
bash deploy/aws-multiregion/launch.sh          # launch 3 silos in 3 regions (BILLABLE, ~$0.03/hr)

# wait ~5 min for the bootstrap (installs the workflow), then check it's ready:
while read -r tier region iid ip; do
  ssh -n -i ~/.ssh/fedwrap-configb -o StrictHostKeyChecking=no ubuntu@$ip \
    'command -v fedwrap-client-eval >/dev/null && echo READY || echo installing'
done < deploy/aws-multiregion/state.tsv

bash   deploy/aws-multiregion/deploy_shards.sh  # push one shard + mask + worker to each silo
python3 deploy/aws-multiregion/run_round.py  --sessions 3 --rounds 40 --quorum 2   # scheduler: full-vs-quorum speed-up (mean±std)
python3 deploy/aws-multiregion/run_search.py --pop 16 --evals 160    # full FedAware-NSGA-II search (relevance sketch + fed-aware operators) across the silos

bash   deploy/aws-multiregion/teardown.sh       # ALWAYS run when done -> deletes everything
```

> The scripts read/write `state.tsv` (instance ids + public IPs) in this folder, and use an SSH key
> `~/.ssh/fedwrap-configb` created by `launch.sh`. Everything is tagged `Project=fedwrap-configb`, so
> `teardown.sh` removes instances, security groups and key pairs **by tag** in all three regions.

### Reuse a coauthor's already-launched silos (no re-launch)

If someone already ran `launch.sh` on the same AWS account, you can run the experiments against their
silos instead of spinning up your own. You need the shared `~/.ssh/fedwrap-configb` key (place it at
that path, `chmod 600`), then, **from your machine**, open your IP and pick up the instance list:

```bash
bash deploy/aws-multiregion/join_existing.sh    # opens SSH from your IP + rebuilds state.tsv by tag
python3 deploy/aws-multiregion/run_round.py --sessions 3 --rounds 40 --quorum 2   # verify: scheduler speed-up (mean±std)
python3 deploy/aws-multiregion/run_search.py --pop 16 --evals 160    # verify: distributed search
```

Do **not** run `teardown.sh` on shared silos unless you agreed to (it removes everyone's, by tag).

## Cost & teardown

`t3.micro` × 3 ≈ **$0.03/hour** total — negligible, but the instances bill until removed. **Run
`teardown.sh` when finished.** To confirm nothing is left:
`aws ec2 describe-instances --filters Name=tag:Project,Values=fedwrap-configb --query 'Reservations[].Instances[].State.Name' --output text` (repeat per region).

## Customizing

- **Regions / tiers:** edit `TIERS` in `launch.sh` (e.g. add `sa-east-1` São Paulo for a second far
  edge, or swap Tokyo for `ap-southeast-2` Sydney). More/farther regions → larger latency disparity.
- **Instance size:** `TYPE` in `launch.sh` (bigger = faster, but for this latency test `t3.micro` is
  enough — the workers are persistent, so inference is off the critical path).
- **Quorum / rounds:** `--quorum` and `--rounds` on `run_round.py`.

## Files

| file | role |
|---|---|
| `launch.sh` | create key/SG and launch one silo per region; writes `state.tsv` |
| `bootstrap.sh` | EC2 user-data: install the workflow on each silo (upgrades setuptools first) |
| `deploy_shards.sh` | push shard + mask + `silo_worker.py` to each silo |
| `silo_worker.py` | resident per-silo evaluator (persistent Flower-like client) |
| `run_round.py` | timed rounds; reports full-vs-quorum speed-up and exact aggregation |
| `run_search.py` | runs the real **FedAware**-NSGA-II search across the silos: federated relevance sketch (from silo statistics) + fed-aware operators + client-stability tie-break + exact aggregation |
| `join_existing.sh` | reuse a coauthor's already-launched silos: open SSH from your IP + rebuild `state.tsv` (no re-launch) |
| `teardown.sh` | delete all `Project=fedwrap-configb` resources in the 3 regions |
