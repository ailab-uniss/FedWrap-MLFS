# FedWrap-MLFS: a federation-aware multi-objective WRAPPER for multi-label feature selection.
#
# The server runs a federation-aware NSGA-II search over binary feature masks; each client
# evaluates a candidate subset with its own multi-label classifier and returns only label-wise
# TP/FP/FN sufficient statistics, from which the server reconstructs the EXACT global macro/micro-F1.
#
# This package is the minimal, self-contained implementation used to produce the paper's results.
