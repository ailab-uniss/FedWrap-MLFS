#!/usr/bin/env bash
# Join an ALREADY-LAUNCHED Config B deployment (a coauthor ran launch.sh) instead of launching your own:
#   - open SSH (port 22) from YOUR public IP on the three security groups, and
#   - rebuild state.tsv (tier region instance-id public-ip) from the running instances, by tag.
# No re-launch, no shared files. Then fetch the SSH key from SSM and run run_round.py / run_search.py.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT=fedwrap-configb
MYIP="$(curl -fsSL https://checkip.amazonaws.com)/32"
echo "your public IP: $MYIP"
: > "$DIR/state.tsv"
for r in eu-south-1 eu-central-1 ap-northeast-1; do
  sg=$(aws ec2 describe-security-groups --region "$r" --filters "Name=group-name,Values=$PROJECT" \
       --query 'SecurityGroups[0].GroupId' --output text)
  if [ "$sg" != "None" ] && [ -n "$sg" ]; then
    aws ec2 authorize-security-group-ingress --region "$r" --group-id "$sg" --protocol tcp --port 22 \
      --cidr "$MYIP" >/dev/null 2>&1 && echo "  $r: opened SSH from $MYIP" || echo "  $r: SSH already allowed from $MYIP"
  fi
  aws ec2 describe-instances --region "$r" \
    --filters "Name=tag:Project,Values=$PROJECT" "Name=instance-state-name,Values=running" \
    --query 'Reservations[].Instances[].[Tags[?Key==`Tier`]|[0].Value, InstanceId, PublicIpAddress]' \
    --output text | while read -r tier iid ip; do
      [ -n "${tier:-}" ] && printf '%s\t%s\t%s\t%s\n' "$tier" "$r" "$iid" "$ip" >> "$DIR/state.tsv"
    done
done
echo "state.tsv:"; cat "$DIR/state.tsv"
