# Matched force-weighting pilot on eight added paths

Two seeds and 807 updates per arm used the same original eight added paths and frozen batch schedule. The full arm used the original force weights. The two reduced arms applied force-only multipliers to the added interior images. Energy-dependent weighting used a 0.50 peak discount. The uniform arm matched its cumulative force-loss coefficient within each path and seed. Both evaluation panels are historical.

| Step 807 barrier MAE change (meV) | Primary, 91 materials | Auxiliary, 88 materials |
| --- | --- | --- |
| Energy weighting minus full | −2.39 [−4.58, −0.93] | −0.99 [−3.31, +1.01] |
| Uniform weighting minus full | −2.21 [−4.43, −0.77] | −0.94 [−2.98, +0.73] |
| Energy weighting minus uniform | −0.18 [−0.35, −0.02] | −0.04 [−0.37, +0.37] |

The main-panel barrier MAE improved under both reductions at step 807. The energy-specific difference is smaller than the 0.25 meV practical threshold and the seeds disagree in sign. At step 400, the contrasts to the full arm span zero on both panels. Both reductions shift the average signed barrier error toward zero at step 807. The full arm remains the comparator for the exploratory dose check. The matched effect of energy placement has yet to be established.

Intervals resample evaluation materials conditional on the two frozen seeds. The four sealed test materials were excluded. The accompanying JSON contains signed barrier and near-peak errors, seed-specific effects and execution provenance. The CSVs contain paired material-level results from locally verified predictions.

On an RTX 4090 with 24 GB, the single-process short test ran at 2.77 updates/s and two processes at 4.75 total updates/s. The three-process worst-batch test exhausted memory. The full pilot used two processes with a sequential retry for out-of-memory failures. The six arms passed completion and prediction-checksum checks.

## Exploratory dose and matched follow-up

Primary-panel MAE relative to the full arm decreased by 1.13, 2.39 and 4.11 meV at peak discounts of 0.25, 0.50 and 0.75, respectively. For the 0.75 arm, the interval was [−7.77, −1.65] meV and the two seed differences were −3.59 and −4.64 meV. The auxiliary-panel mean change was −1.84 meV with interval [−5.75, +1.54]. Both panels had a less negative mean signed barrier error under the 0.75 treatment.

The matched uniform reduction at 0.75 improved the primary MAE by 3.56 meV. Energy weighting reduced the MAE by another 0.56 meV relative to that control, with interval [−0.91, −0.22] and seed contrasts −0.15 and −0.96 meV. The interval includes improvements smaller than the 0.25 meV practical threshold. The auxiliary contrast was −0.27 [−1.00, +0.58] meV. These results support force downweighting for these eight added paths. The additional benefit of targeting high-energy images is provisional.

The best tested peak discount was 0.75. It sets the minimum force multiplier to 0.25 on the added interior images. The energy-loss weight and all other images retain their original coefficients. This choice remains exploratory because the dose was selected on the two historical panels. No images were removed in this run. All 12 completed arms were scored against the same locally held references.
