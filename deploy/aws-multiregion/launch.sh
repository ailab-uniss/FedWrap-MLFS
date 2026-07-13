#!/usr/bin/env bash
# Config B: launch 3 EC2 "silos" (edge/server/cloud tiers) in 3 regions for the multi-region
# WAN-latency test. Everything is tagged Project=fedwrap-configb so teardown.sh removes it by tag.
# t3.micro x3 = 6 vCPU -> if your On-Demand quota is 5, request an increase or set TYPE=t2.micro.
# BILLABLE but tiny (~$0.03/hr total). Run teardown.sh when done.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT=fedwrap-configb
KEY=fedwrap-configb
TYPE=t3.micro
SSM=/aws/service/canonical/ubuntu/server/jammy/stable/current/amd64/hvm/ebs-gp2/ami-id
# tier -> region  (cloud=near/low-latency, server=~30ms, edge=~230ms WAN)
TIERS="cloud:eu-south-1 server:eu-central-1 edge:ap-northeast-1"
STATE="$DIR/state.tsv"

[ -f "$HOME/.ssh/$KEY" ] || ssh-keygen -t ed25519 -f "$HOME/.ssh/$KEY" -N "" -q
MYIP="$(curl -fsSL https://checkip.amazonaws.com)/32"
echo "orchestrator ingress IP: $MYIP"

: > "$STATE"
for tr in $TIERS; do
  tier=${tr%%:*}; r=${tr##*:}
  echo "=== [$tier] $r ==="
  aws ec2 import-key-pair --key-name "$KEY" --public-key-material "fileb://$HOME/.ssh/$KEY.pub" --region "$r" >/dev/null 2>&1 || true
  vpc=$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true --region "$r" --query 'Vpcs[0].VpcId' --output text)
  sg=$(aws ec2 describe-security-groups --filters "Name=group-name,Values=$PROJECT" "Name=vpc-id,Values=$vpc" --region "$r" --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || echo None)
  if [ "$sg" = "None" ] || [ -z "$sg" ]; then
    sg=$(aws ec2 create-security-group --group-name "$PROJECT" --description "fedwrap configb SSH" --vpc-id "$vpc" --region "$r" \
         --tag-specifications "ResourceType=security-group,Tags=[{Key=Project,Value=$PROJECT}]" --query GroupId --output text)
    aws ec2 authorize-security-group-ingress --group-id "$sg" --protocol tcp --port 22 --cidr "$MYIP" --region "$r" >/dev/null
    echo "  created SG $sg (SSH from $MYIP)"
  fi
  ami=$(aws ssm get-parameters --names "$SSM" --region "$r" --query 'Parameters[0].Value' --output text)
  iid=$(aws ec2 run-instances --image-id "$ami" --instance-type "$TYPE" --key-name "$KEY" --security-group-ids "$sg" \
        --region "$r" --count 1 --user-data "file://$DIR/bootstrap.sh" \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Project,Value=$PROJECT},{Key=Tier,Value=$tier}]" \
        --query 'Instances[0].InstanceId' --output text)
  echo "  launched $iid ($TYPE), waiting to be running..."
  aws ec2 wait instance-running --instance-ids "$iid" --region "$r"
  ip=$(aws ec2 describe-instances --instance-ids "$iid" --region "$r" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
  printf '%s\t%s\t%s\t%s\n' "$tier" "$r" "$iid" "$ip" >> "$STATE"
  echo "  -> $tier  $iid  $ip"
done
echo; echo "=== instances (tier region id ip) ==="; cat "$STATE"
echo "SSH key: ~/.ssh/$KEY   user: ubuntu   (bootstrap installs the workflow; give it ~5 min)"
