#!/usr/bin/env bash
# Remove ALL Config B resources (tagged Project=fedwrap-configb) in the 3 regions. Run when done.
set -uo pipefail
PROJECT=fedwrap-configb
KEY=fedwrap-configb
for r in eu-south-1 eu-central-1 ap-northeast-1; do
  echo "=== $r ==="
  ids=$(aws ec2 describe-instances --region "$r" \
        --filters "Name=tag:Project,Values=$PROJECT" "Name=instance-state-name,Values=pending,running,stopping,stopped" \
        --query 'Reservations[].Instances[].InstanceId' --output text)
  if [ -n "$ids" ]; then
    echo "  terminating: $ids"; aws ec2 terminate-instances --instance-ids $ids --region "$r" >/dev/null
    aws ec2 wait instance-terminated --instance-ids $ids --region "$r"
  fi
  sg=$(aws ec2 describe-security-groups --region "$r" --filters "Name=group-name,Values=$PROJECT" --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || echo None)
  if [ "$sg" != "None" ] && [ -n "$sg" ]; then aws ec2 delete-security-group --group-id "$sg" --region "$r" 2>/dev/null && echo "  deleted SG $sg"; fi
  aws ec2 delete-key-pair --key-name "$KEY" --region "$r" >/dev/null 2>&1 && echo "  deleted key pair"
done
echo "teardown complete"
