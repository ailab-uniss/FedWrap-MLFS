"""FedWrap-MLFS workflow steps (client evaluation, counter aggregation, shard I/O).

Packaged so the per-client and aggregation steps install as console scripts
(``fedwrap-client-eval`` / ``fedwrap-aggregate``) and run identically inside a container or in a plain
conda/venv environment -- i.e. the CWL/StreamFlow workflow is container-optional.
"""
