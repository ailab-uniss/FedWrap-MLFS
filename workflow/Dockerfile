# Container for the FedWrap-MLFS federated-evaluation workflow steps.
# Built locally with Docker for the executable demo; on HPC the same image runs under
# Singularity/Apptainer, so the CWL workflow is portable without change.
#   docker build -f workflow/Dockerfile -t fedwrap-workflow:latest .
FROM python:3.10-slim

WORKDIR /app
COPY pyproject.toml /app/
COPY fedwrap /app/fedwrap
COPY workflow /app/workflow
# installs the deps + the fedwrap/workflow packages + the console scripts
# (fedwrap-client-eval / fedwrap-aggregate), which the CWL baseCommands invoke.
RUN pip install --no-cache-dir .
ENV PYTHONPATH=/app
