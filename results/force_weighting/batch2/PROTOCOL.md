# Replication on the second path block

Use the original `protected8_batch2_random` block and `protected_replay` schedule from the locked protected-training bundle. The eight training materials in this block are disjoint from those used to select the discount. Reuse the previously locked 91-material primary and 88-material auxiliary evaluation panels. This tests transfer across added training paths. The evaluation panels are historical and the four sealed test materials are outside this run.

Use paired seeds 17 and 29, Recipe A, the same initialization and 807 optimizer updates, including 97 slot-matched insertions. The three augmented arms use exactly the same paths and training order. The fourth arm uses atom-count-matched base replay at the insertion slots.

| Arm | Added-image force coefficients |
| --- | --- |
| Replay | Original weights on replayed baseline frames |
| Full | Original weights on every image |
| Uniform | Constant within each path's added interior images, matched to the energy arm after atom-count normalization and repeated presentations |
| Energy | Multiplier 1 − 0.75 × normalized relative DFT energy on the added interior images |

The discount is frozen at 0.75. The minimum force multiplier is 0.25. Energy supervision, endpoints and original baseline images retain Recipe A weights. Recipe A remains the reference recipe.

The primary endpoint is forward barrier MAE after 807 steps. Use paired material bootstrap intervals conditional on both seeds. Confirm that the energy arm beats replay only if both seed contrasts favor the energy arm and the primary-panel 95% interval upper endpoint is at or below −0.25 meV. Apply the same gate separately to energy versus matched uniform. Report energy and uniform versus full to quantify mitigation. Report auxiliary-panel contrasts, signed barrier error and near-peak profile error. Treat step 400 as a trajectory diagnostic. An auxiliary-panel deterioration or worsening underestimation limits the interpretation. This design measures benefits under an already computed label set and compares neither DFT core-hours nor pre-DFT selection policies.

Training entry point and scoring from the repository root:

```sh
PYTHON_BIN=python bash scripts/run_force_weighting_batch2.sh PROTECTED_BUNDLE.zip OUTPUT
python src/run_force_weighting.py PROTECTED_BUNDLE.zip OUTPUT --replicate-batch2 --project PROJECT_ROOT
```
