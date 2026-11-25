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
