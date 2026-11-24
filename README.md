# Solid Electrolyte Path Selection

Benchmarking which ion-migration paths to prioritize for DFT labeling under a limited budget to improve machine-learned interatomic potentials for solid electrolytes.

## Research question

Which complete migration paths improve forward-barrier predictions on held-out materials after fine-tuning a fixed MACE baseline? The design compares augmented training sets against pure D₀ and compares selection policies at matched training exposure. Realized DFT core-hours are a prospective endpoint.

## Experimental progress

- A retrospective audit of 38 candidate paths found no reliably improving single-path result under the predeclared 0.25 meV screening criterion.
- A four-path pilot compared random selection and the highest forward BVSE score in matched cost strata. Each arm added 22 images to the 278-frame baseline and used seeds 17, 29 and 43.
- On the 91 project-held-out oxides, changes in mean absolute error relative to D₀ were +23.97 and +12.84 meV for the first random and BVSE arms, and +13.99 and +6.89 meV for the second. Positive values indicate higher error. BVSE reproducibility across the two candidate blocks remains unresolved.
- The eight-path pilot retains the original four paths and adds one matched path per stratum. The analysis reuses the same 91 materials. The dose-sensitivity experiment is complete. See [paired results](results/eight_paths_summary.json). In the first candidate block, moving from four to eight paths increased MAE by 7.13 meV for random selection and 11.47 meV for BVSE. Both second-block intervals for the four-to-eight-path comparison span zero. All four eight-path arms have higher point-estimate MAE than pure D₀.

## Data and code

- experiments/ contains frozen candidate selections, evaluation IDs and baseline frame IDs.
- results/ contains compact summaries and per-material, per-seed absolute errors. Material bootstrap intervals condition on the three training seeds.
- src/ contains the archived training and scoring code. The archived runners use PyTorch 2.8.0 with CUDA 12.8, MACE 0.3.14, ASE 3.29.0 and NumPy 2.2.6.

The inputs originate from the [LiTraj project](https://github.com/AIRI-Institute/LiTraj). Running training requires the original input bundles with DFT labels and the MACE-MP-0 Small checkpoint. Model weights and raw input bundles are maintained separately. Evaluation labels are read locally after blind predictions are committed. The evaluation cohort comes from the upstream Train split and is held out within this project.

## Run or score an archived experiment

    python src/run_batch_pilot.py FOUR_PATH_BUNDLE.zip --preflight
    python src/run_eight_path_pilot.py EIGHT_PATH_BUNDLE.zip --smoke-gpu
    python src/run_eight_path_pilot.py EIGHT_PATH_BUNDLE.zip --out results

For offline scoring, supply the verified run archives and a project root containing 迁移路径数据/nebDFT2k.zip and experiment2/prepare_data.py.

    python src/score_eight_path_pilot.py FOUR_PATH_BUNDLE.zip FOUR_RESULTS.tar EIGHT_PATH_BUNDLE.zip EIGHT_RESULTS.tar PROJECT_ROOT SCORED.zip

Four-path and eight-path effects are compared on the same geometries, environment, and paired seeds. At a fixed optimization budget, increasing the number of paths also changes the sampling frequency of D₀.
