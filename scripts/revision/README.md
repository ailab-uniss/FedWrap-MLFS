# Experiments added in the first revision of the manuscript

All commands are run from the repository root. Outputs go to `runs/revision/` and `reports/revision/`.

| Script | Purpose | Manuscript |
|---|---|---|
| `make_resplits.py` | Ten independent per-silo 75/25 non-test/test partitions of each natural federation (`data/fed_real/{DS}__r{r}`) | Sec. 5.1, 6.1 |
| `run_resplit_method.py` | One search on one partition; same configuration as `scripts/run_reals_method.py` (`faport` = FedWrap-MLFS, `bitstring` = standard NSGA-II) | Sec. 6.1 |
| `launch_resplits.sh` | Runs both searches on the ten partitions of the three federations (search seed = partition index) | Sec. 6.1 |
| `build_resplit_table.py` | Test re-evaluation, matched baselines, FMLFS per partition, paired Wilcoxon tests with Holm adjustment, wrapper and federation-aware effects | repeated-partition and decomposition tables |
| `decomposition_stats.py` | Wrapper and federation-aware effects on the ten-seed runs with the fixed test partition | Sec. 6.1 |
| `run_flower_search.py` | Complete search on the Flower/Ray runtime (`mask` or `generation` round mapping), compared mask for mask with an in-process run | Sec. 4 |
| `run_secure_e2e.py` | Complete search with pairwise-masked sums for the sketch, the filter seeds, the counts, and the stability sums | Sec. 3.6, 7.4 |
| `run_groupwise_robust.py` | Count flooding: aggregate consistency check and group-wise secure aggregation with outlier removal | Sec. 3.6, 7.4 |

`fedwrap/nsga2.py` gained an optional `prefetch` argument (default `None`, behaviour unchanged): it receives each batch of
genomes before the batch is evaluated, so that a federated evaluator can dispatch one generation in a single round.

Reproduction order:

```bash
python scripts/revision/make_resplits.py 10
bash   scripts/revision/launch_resplits.sh
python scripts/revision/build_resplit_table.py 10
python scripts/revision/run_flower_search.py ECG_cinc2021__r1 faport 1 generation runs/revision/resplit/ECG_cinc2021__r1_faport_s1_fold0
python scripts/revision/run_flower_search.py ECG_cinc2021__r1 faport 1 mask       runs/revision/resplit/ECG_cinc2021__r1_faport_s1_fold0
python scripts/revision/run_secure_e2e.py   ECG_cinc2021__r1 1 runs/revision/resplit/ECG_cinc2021__r1_faport_s1_fold0
python scripts/revision/run_groupwise_robust.py
```
