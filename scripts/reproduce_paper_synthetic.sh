#!/usr/bin/env bash
# Regenerate the EXACT synthetic federations (and their results) used in the manuscript.
#
# Generation is BYTE-DETERMINISTIC: synth_scaling.materialize draws from numpy's version-stable
# default_rng(seed) (PCG64 bit generator), so re-running on any machine yields the identical data --
# verified by hashing (same seed -> identical SHA-256; different seed -> different). The exact
# parameters live in the committed scripts (run_synth_scaling.DEFAULT/GRIDS, run_interaction_campaign).
# Hence this recipe *is* the synthetic dataset: there is no need to ship ~280 MB of .npz binaries.
#
# This reproduces all three synthetic studies in the paper. It regenerates the data as a side effect
# and also re-runs the wrapper, writing the report CSVs under reports/.
set -euo pipefail
cd "$(dirname "$0")/.."
S10=0,1,2,3,4,5,6,7,8,9   # headline seeds (10)
S5=0,1,2,3,4              # secondary seeds (5)

# 1. scaling law in feature dimension D  -- D in {100,500,1000,2000,5000}, 10 seeds, base + fedaware
SC_GRID="100,500,1000,2000,5000" python3 scripts/run_synth_scaling.py D "$S10" base
SC_GRID="100,500,1000,2000,5000" python3 scripts/run_synth_scaling.py D "$S10" fedaware

# 2. heterogeneity sweeps  -- alpha in {10,1,0.3,0.1} and K in {2,4,8,16,32}, 5 seeds
python3 scripts/run_fedaware_synth.py alpha "$S5"
python3 scripts/run_fedaware_synth.py K     "$S5"

# 3. interaction campaign  -- interaction_frac in {0.0,0.3,0.5,0.7}, 5 seeds
for f in 0.0 0.3 0.5 0.7; do python3 scripts/run_interaction_campaign.py "$f" "$S5"; done

echo "Done. Regenerated the exact paper synthetic (data) + reports under reports/."
echo "Byte-identical to the manuscript's data by construction (deterministic default_rng seeds)."
