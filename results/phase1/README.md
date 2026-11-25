# Phase 1 paired single-path pilot

The 21 frozen jobs completed on one RTX 4090. Three shared pure-D₀ runs and nine exposure-matched candidate/replay pairs used seeds 17, 29 and 43. The primary endpoint was the forward barrier MAE at the final checkpoint on 107 project-held-out materials. This panel previously supported the separate Bi–Li–O–V audit. It was locked for this comparison and is to be distinguished from the final sealed test.

| Complete path | Absolute gain (meV) | Net gain (meV) | Decision on this panel |
| --- | ---: | ---: | --- |
| mp-1277232_0_3_0_0_0, Co–Cr–Li–O | −5.02 [−9.37, −0.99] | −4.01 [−7.73, −0.80] | Both negative |
| mp-13725_0_4_1_0_0, Li–O–P | −0.86 [−1.63, −0.07] | −0.09 [−0.66, +0.43] | Both unresolved at the practical margin |
| mp-753143_5_13_0_0_0, F–Li–Ni–O | +0.10 [−0.58, +0.83] | +1.11 [+0.34, +1.84] | Net positive, absolute unresolved |

Positive gains favor the actual path. The intervals resample exact chemical systems and preserve each selected material's paired seed effects. They are conditional on the three seeds and apply to individual pilot contrasts. The 0.25 meV practical margin was set before training. The three paths are an exploratory sample of the candidate pool.

The F–Li–Ni–O net gain was +1.25, +1.00 and +1.07 meV in the three seeds. It was positive on 71 of 107 materials, and excluding any single material leaves the mean between +1.01 and +1.27 meV. The corresponding replay arm was 1.01 meV worse than pure D₀ by point estimate. The net gain establishes an advantage over replay under the matched schedule. Improvement over pure D₀ remains unresolved. The Co–Cr–Li–O mean net effect also keeps its sign under all single-material omissions.

The training package used Recipe A, with all five or seven labeled images presented five times. The protected schedules retain each of the 710 original D₀ updates. The exposure control matches the inserted positions, companion image, candidate-slot atom count, image count and total steps. The model files remain in the run data store. The stored hashes identify the inputs, final and step-400 predictions, training traces and model checkpoints.

[Sampling coverage](figures/feature_coverage.png) · [Paired gain intervals](figures/paired_gains.png) · [Material-level responses](figures/material_gains.png)
