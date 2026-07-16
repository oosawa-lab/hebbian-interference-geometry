# Data dictionary

## `data/aggregated/`

- `L4_V03b_cost_summary.csv`: condition-level intensive and cumulative costs.
- `L4_V03b_deltaF_summary.csv`: correlated-minus-uncorrelated excess costs.
- `L4_V03b_finite_size_nearest_load_summary.csv`: nearest-load finite-size matching.
- `analytical_prediction_comparison.csv`: observed and overlap-moment predicted costs.
- `analytical_prediction_metrics_corrected.csv`: corrected analytical benchmark metrics.
- `correlated_minus_uncorrelated_summary.csv`: retrieval degradation and excess costs.
- `pooled_cost_retrieval_correlations.csv`: pooled Pearson/Spearman associations.
- `retrieval_link_bootstrap_summary.csv`: bootstrap uncertainty for retrieval-risk links.
- `retrieval_sign_summary.csv`: sign frequencies for excess cost and retrieval damage.
- `trial_condition_summary.csv`: retrieval summaries by condition.
- `realization_condition_summary.csv`: geometry summaries by realization condition.
- `L4_V04_robustness_summary.csv`: robustness summaries across ensembles and parameters.
- `L4_V04_robustness_nearest_load_scaling.csv`: robustness finite-size matches.
- `L4_V04_robustness_finite_size_power_fits.csv`: diagnostic power-law fits.

## `data/samples/`

The two sample files contain the header and first 1,000 records of the large
trial-level tables. The complete tables are distributed in the GitHub Release
asset, not in the Git history.

## Main trial-level variables

- `cost_int`: intensive Hebbian writing cost.
- `cost_cum`: cumulative Hebbian writing cost.
- `final_overlap`: final overlap with the target memory after recall.
- `retrieval_error`: `1 - final_overlap`.
- `success`: one when final overlap is at least 0.90.
- `corr`: pattern-correlation control parameter.
- `cue_noise`: cue-corruption probability.
