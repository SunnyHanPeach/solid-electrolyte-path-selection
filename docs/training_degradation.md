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

## Should high-energy images be omitted?

The audit locates the prediction drift on high-energy held-out images. The effect of withholding high-energy training images is a separate question.

| Primary source | Result relevant to this project | Boundary |
| --- | --- | --- |
| [Dembitskiy et al., LiTraj, 2025](https://doi.org/10.1038/s41524-025-01571-z) | Using only configurations sampled during vacancy-endpoint optimization to fine-tune SevenNet (SevenNet-ft-vac) led to pronounced migration-barrier underestimation. | Their intervention removed coverage of the migration interior. It does not isolate the effect of thinning the highest quarter of the interior images in our MACE protocol. |
| [Deng et al., systematic softening, 2025](https://doi.org/10.1038/s41524-024-01500-6) | The three tested universal potentials exhibited energy and force underprediction. In the focused CHGNet example, fine-tuning with high-energy configurations reduced force error on the same material. | The fine-tuning example used an AIMD-derived, material-specific dataset, so it cannot establish cross-material transfer from our eight paths. |
| [High-throughput CHGNet-NEB study, 2025](https://doi.org/10.1039/D5TA05355B) | Fine-tuning with transition-state configurations reduced the separate-test barrier MAE from 0.23 to 0.09 eV. | Those DFT references were single-point energies at CHGNet-predicted structures. The comparison used a pretrained control. |
| [Ge et al., simulation-oriented training, 2024](https://doi.org/10.1021/acs.jpclett.4c00746) | Energy weighting on a glycine PES improved the mean saddle-point error from 0.19 to 0.055 kcal/mol across five ANI fits, even as the test RMSE increased. | This supports testing weights against the downstream observable. The specific weights and energy scale are molecule- and task-specific. |
| [Gibson et al., 2025](https://doi.org/10.1103/PhysRevMaterials.9.043802) | Training on an overly broad potential-energy landscape degraded relevant predictions for a finite-capacity linear Si-N potential. | The result supports testing a task-specific change in data composition. It provides no universal cutoff for relevant saddle-region samples. |

### CPU force audit

Use steps 8, 200 and 400 from both existing arms. Compare model forces with the unprojected DFT atomic forces on the original DFT-NEB images. Keep the published image regions and aggregate paired changes by material. Report migrating-Li and all-atom RMSE within each region, and the signed force-error projection along the full configuration path on the approach to the high-energy region. The projection includes moving atoms other than Li. A barrier-height error can accrue on the approach and coexist with a small local force error at the peak. Report force–barrier associations per seed and panel. This analysis can locate a correlation. Causation requires a training intervention. [FP-DeErr protocol](https://mogroupumd.github.io/FP-DeErr/ion-migration-neb.html)

### Minimal controlled training ablation

Freeze the existing eight paths and use each path's DFT energy relative to its own initial image. Predefine the high-energy region as the highest quarter of the interior frames rounded up, with a deterministic tie rule. Match eligibilities and counts across the two pruning arms. Keep both endpoints in all arms.

1. **Full path.** Use the complete existing paths.
2. **Omit selected high-energy images.** Replace a fixed number of highest interior-image presentations with atom-count-matched D₀ replay.
3. **Omit other interior images.** Replace the same number of non-high interior-image presentations from the same paths with matched D₀ replay under a frozen sampling rule.

Hold fixed the base D₀ batch sequence, total updates, matched replacement exposure, initialization, seeds 17 and 29, and the original energy/force weights, optimizer and clipping. Preserve the evaluation images, including peak-region images, and the full forward barrier metric. The comparison between the two omission arms tests specificity to the high-energy region. Verify whether any improvement on the main panel comes with deterioration in the auxiliary panel or an increase in barrier underestimation. Report both seed effects and paired material intervals. Use the same-runtime full-path control when available. A continuation from step 200 would test only omissions after step 200.

If both omissions behave alike, reduced new-data exposure is an alternative explanation. If high-energy omission raises barrier underestimation, preserve those images. With evidence specific to high-energy images, a second ablation can reduce only their force-loss weight while retaining energy supervision, with the same reduction on matched other-interior configurations as a control. The Adam 2×2 intervention remains a separate test of history, using the same continuation batches and measuring change from each model's own starting point.

These are training-time interventions with existing labels. Any future claim about avoiding DFT work requires a separate acquisition experiment. Adaptive Gaussian-process NEB provides a precedent for reducing individual evaluations while targeting poorly known parts of a path. [Koistinen et al., 2017](https://doi.org/10.1063/1.4986787)

### Energy-dependent force weighting as a separate hypothesis

Test a nonzero floor before considering removal. For an interior training image i, let its excess energy be the positive part of its DFT energy above the lower of the two endpoint energies. Divide by the largest excess energy among interior images of that path to obtain s_i in [0,1]. With zero interior excess, keep all weights at one. The first fixed exploratory schedule is w_F(i) = 1 - 0.5 s_i for added interior images. Keep energy supervision, both endpoints, and every D₀ image at the original weight. This is a path-relative training heuristic with the existing DFT labels.

Compare three arms with the same images, optimizer, training sequence and seeds. Use the original weights, energy-dependent force weights, and a uniform force-weight reduction for the same eligible interior images. Set the uniform reduction separately for each path so it matches the total force-loss coefficient of the energy-dependent arm. Account for repeated image presentations and the original batch and atom-count normalization, and keep D₀ exposure and its loss coefficients fixed. This tests whether energy-based placement of reduced weights improves the outcome beyond an equal reduction in force supervision.

Use the full-path barrier MAE as the main endpoint, including the original highest-energy evaluation images. Report signed barrier bias, high-energy profile error, and paired material and seed effects on both panels. A higher barrier-underestimation risk, or equal performance for the two reduced-weight arms, limits the case for energy-dependent weighting. A favorable historical result needs confirmation on unused materials before adoption. The coefficient 0.5 is a fixed exploratory dose, not an estimated optimum. [Ge et al., task-oriented energy weighting](https://doi.org/10.1021/acs.jpclett.4c00746) · [Deng et al., high-energy supervision and softening](https://doi.org/10.1038/s41524-024-01500-6)

Run both seed commands with the original protected-run bundle. The new comparison uses the same frozen augmented schedule in all three arms. Step 807 is the primary endpoint and step 400 describes the trajectory. Both panels were used previously. The independently held test materials remain outside this run. Use the existing 0.25 meV practical threshold for the exploratory comparison against both controls. Inspect both seed effects and the paired material intervals alongside the auxiliary-panel and signed-error results.

    python src/run_force_weighting.py PROTECTED_BUNDLE.zip OUTPUT --seed 17
    python src/run_force_weighting.py PROTECTED_BUNDLE.zip OUTPUT --seed 29
    python src/run_force_weighting.py PROTECTED_BUNDLE.zip OUTPUT --project PROJECT_ROOT --until 400
    python src/run_force_weighting.py PROTECTED_BUNDLE.zip OUTPUT --project PROJECT_ROOT

Add `--device cuda` to both training commands for a GPU run. Keep the device and thread count the same across the matrix. The scorer validates the common execution environment and checksums before comparing predictions. Raw models and training logs stay in the run directory.

### Follow-up decisions

When the first treatment reduces the primary-panel error without a clear auxiliary-panel MAE or signed-bias penalty, compare peak discounts of 0.25 and 0.75 to the original curve and to the original unit weights. The discount is the `--strength` parameter. A favorable historical dose is exploratory. Verify a matched uniform reduction at the selected strength before attributing the effect to energy placement. Reuse the initial arms on the same GPU and software, and keep these comparisons separate from any later untouched-material verification.

If the initial force weighting has no favorable error trade-off, compare `--omit high` and `--omit other` with the original full arm. Both omissions replace the same count of training presentations in each path and batch atom-count stratum. The replay indices come from the previously frozen schedule. Original endpoints, D₀ batches and evaluation images are retained. The primary ablation contrast is high minus other omission. If it does not favor high-energy omission, use both panels and signed profiles to determine which alternative mechanism merits testing.

Additional arms use separate output directories. For local scoring, append arguments such as `--compare alpha025=OUT025@energy_weighted --compare alpha075=OUT075@energy_weighted --score-output FOLLOWUP` to a scoring command for the original run directory.

## Reproduction

Run from the repository root with Python and NumPy. The signed CSV is already included.

    python src/audit_signed_barriers.py analyze results/protected_trajectory/signed_barrier_errors.csv audit.json

To reconstruct the signed CSV from local archives, supply the original protected-run GPU bundle, the blind trajectory predictions archive, and the project root holding the audited LiTraj dataset and preparation code.

    python src/audit_signed_barriers.py extract PROTECTED_BUNDLE.zip BLIND_PREDICTIONS.tar.gz PROJECT_ROOT results/protected_trajectory/trajectory_paired_materials.csv signed.csv

To reproduce the fixed-image localization from the same archives:

    python src/check_barrier_profile_regions.py PROTECTED_BUNDLE.zip BLIND_PREDICTIONS.tar.gz PROJECT_ROOT profile_region_summary.json
