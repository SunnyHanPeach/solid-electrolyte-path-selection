# Eight-path training degradation audit

## Evidence and scope

The original paired run contains 807 updates with identical exposure to the 278-frame D₀ set and different contents at 97 matched insertion slots. Seeds 17 and 29 ran both sides. All results here use fixed DFT-NEB geometries. The 91-material main and 88-material auxiliary panels are historical panels. The reserved test set was excluded.

Step 400 is the first recorded checkpoint to reach the planned degradation criterion and retain it at the next checkpoint. Main-panel MAE differences at steps 200, 400, 600 and 807 are +6.37, +19.21, +24.10 and +30.09 meV. The step-200 interval spans zero. The exact onset within the checkpoint interval is unresolved.

## New diagnostic result

Let r be the replay prediction minus the DFT barrier and d the augmented prediction minus the replay prediction. The observed paired absolute error difference is |r+d|−|r|. Signed errors have been reproduced from the blind predictions and cross-checked against every archived material, seed and checkpoint score.

| Step 807 quantity (meV) | Main (91) | Auxiliary (88) |
| --- | ---: | ---: |
| Mean signed barrier change, d | −42.75 | −69.17 |
| Observed MAE difference | +30.09 | −2.40 |
| MAE difference under one pooled shift per seed | +35.39 | +27.31 |
| MAE difference with the observed shifts randomly reassigned across materials (mean of 5,000 permutations) | +32.31 | +45.41 |

A shared mean shift qualitatively reproduces main-panel degradation. It also predicts degradation in the auxiliary panel, where the observed mean is near zero. Randomly reassigning the observed shifts produces the same mismatch. This indicates that which materials receive the largest shifts matters in the auxiliary panel. The shuffle uses the two seed shifts jointly for each material. It is a descriptive reference on these panels.

The early bias groups are defined using step-8 replay, a model that has seen only D₀ or D₀ replay; its signed errors are averaged across the two seeds. Early underprediction is below −25 meV and early overprediction is at least +25 meV. All three groups in the main panel degrade on average at step 807. The early-overprediction group in the auxiliary panel (49 materials) improves by 34.19 meV, whereas the corresponding main-panel group (37 materials) degrades by 31.45 meV. The corresponding replay-to-augmented group mean signed errors are +133.70 to +39.15 meV on the auxiliary panel and +40.10 to −24.40 meV on the main panel. These are group means. Balancing the panels on the three early-bias bins reduces the endpoint difference from 32.48 to 27.66 meV, so the early-bias bin proportions explain only a small part of the point difference.

## Spatial localization along fixed NEB images

The average signed change concentrates in high-energy interior images. The high-energy region uses the highest quarter of the DFT relative energies among interior images of each path, rounded up.

| Step 807 relative-energy change (meV) | Main (91) | Auxiliary (88) |
| --- | ---: | ---: |
| High-energy interior | −40.86 | −60.60 |
| Remaining interior | −15.49 | −25.90 |
| Final endpoint | −1.00 | +1.68 |

Relative to replay, the high-energy interior absolute error on the main panel increased by 16.16 meV at step 400 and 26.60 meV at step 807. The other interior images increased by 2.21 and 3.66 meV.

At the replay model's existing peak image, the augmented energy was lower by 43.43 meV (main) and 70.78 meV (auxiliary) on average at step 807. Letting the augmented prediction choose a different peak added only +0.68 and +1.61 meV. The barrier-height decrease is therefore present at the original peak, with peak reassignment slightly offsetting it. This analysis locates the error within the energy profiles and does not identify its training cause.

## Remaining explanations worth testing

1. **Task-specific energy-profile drift.** Added paths may progressively change relative energies near high-energy images. Existing main-panel results show +9.32 meV in profile MAE, −1.59 meV in endpoint difference MAE and a −0.55 percentage point change in peak-image match rates at step 807. With the high-energy profile drift localized, compare the step-200 and step-400 checkpoints on force RMSE over fixed images, separating migrating Li and all atoms. This can test whether force errors worsen in the same images and time window.
2. **Material-dependent correction or overshoot.** Evaluate which cases account for large useful or harmful downward shifts, comparing r and d on both panels and using chemistry and barrier range as predeclared descriptive strata. The current group means and shuffled reference support heterogeneous signed drift. They do not identify a chemical cause.
3. **Optimizer history.** If a follow-on update experiment is warranted, start at step 200, run fixed batches, and cross the two stored model weights with both Adam states. This separates intervention on history from the already accrued parameter changes. With only two seeds and historical panels, no general Adam or force-loss mechanism is claimed.

All outputs were generated without new training or GPU use. The first step is an audit of the training effect under the fixed MACE setup. The separate project goal is to improve path selection on future materials under equal realized DFT core-hours.

## Reproduction

Run from the repository root with Python and NumPy. The signed CSV is already included.

    python src/audit_signed_barriers.py analyze results/protected_trajectory/signed_barrier_errors.csv audit.json

To reconstruct the signed CSV from local archives, supply the original protected-run GPU bundle, the blind trajectory predictions archive, and the project root holding the audited LiTraj dataset and preparation code.

    python src/audit_signed_barriers.py extract PROTECTED_BUNDLE.zip BLIND_PREDICTIONS.tar.gz PROJECT_ROOT results/protected_trajectory/trajectory_paired_materials.csv signed.csv

To reproduce the fixed-image localization from the same archives:

    python src/check_barrier_profile_regions.py PROTECTED_BUNDLE.zip BLIND_PREDICTIONS.tar.gz PROJECT_ROOT profile_region_summary.json
