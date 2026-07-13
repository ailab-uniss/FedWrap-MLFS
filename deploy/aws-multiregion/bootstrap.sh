#!/bin/bash
# EC2 user-data: prepare a FedWrap-MLFS "silo" instance (installs the container-optional workflow).
exec > /var/log/fedwrap-bootstrap.log 2>&1
set -x
apt-get update -y
apt-get install -y python3-pip git
# Ubuntu 22.04 ships setuptools < 61, which cannot read our PEP 621 pyproject (installs "UNKNOWN"
# and creates no console scripts) -- upgrade the build tools first.
pip3 install --upgrade pip setuptools wheel
cd /home/ubuntu
git clone https://github.com/ailab-uniss/FedWrap-MLFS.git
cd FedWrap-MLFS && pip3 install .
chown -R ubuntu:ubuntu /home/ubuntu/FedWrap-MLFS
echo "fedwrap bootstrap done"
