# Container for the FedWrap-MLFS federated-evaluation workflow steps.
# Built locally with Docker for the executable demo; on HPC the same image runs under
# Singularity/Apptainer, so the CWL workflow is portable without change.
#   docker build -f workflow/Dockerfile -t fedwrap-workflow:latest .
FROM python:3.10-slim

RUN pip install --no-cache-dir numpy==2.* scipy==1.* scikit-learn==1.* pyyaml

WORKDIR /app
COPY fedwrap /app/fedwrap
COPY workflow /app/workflow
ENV PYTHONPATH=/app
