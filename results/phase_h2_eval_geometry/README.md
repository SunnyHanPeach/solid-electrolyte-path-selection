# H2 evaluation panel geometry preparation

This run produced a local shortlist of 12 pilot and 8 reserve migration paths. The pilot tests whether a balanced group of independent structures can reach the labeling stage. The panel currently has zero primary evaluation paths with DFT labels. The FP-DeErr external fixed-geometry panel remains available for scoring.

| Source stratum | Eligible materials before detailed matching | Screened | Pilot | Reserve |
| --- | ---: | ---: | ---: | ---: |
| Oxides | 921 | 40 | 3 | 2 |
| Sulfides | 60 | 40 | 3 | 2 |
| Halides | 319 | 40 | 3 | 2 |
| Mixed and other | 291 | 40 | 3 | 2 |

## Selection and validation

The source was the archived BVSE low-cost geometry screen. The selection used its physical pass and prior overlap flags, then excluded exact material identities and chemical systems shared with the historical pool or H1 training candidates. A fixed, documented SHA-256 ranking selected one edge per material and 40 materials per stratum. The ranking used identities and the salt in [summary.json](summary.json). It used no DFT energy, prior model performance, or BVSE barrier ranking.

The review ran 5,821 structural comparisons. It checked the Li-free host frameworks against historical parents, all 24 frozen H1 training candidates, 278 D₀ structures, 62 FP-DeErr reference paths, and other evaluated candidates. Matching used primitive, scaled, anonymous framework fits and joined matched candidates transitively. Eleven candidate components touched a reference and were excluded. Selected structures represent 20 distinct chemical systems and framework groups under this screen. This prescreen covers the 160 sampled candidates and listed references. Recheck the panel against any later training cohort and check local migration-environment overlap before freezing the final panel.

An independent read-back verified SHA-256 hashes, origin geometry, periodic Li displacements, atom ordering, reaction coordinate, exact exclusions, and reference-connected components. Each exported seed contains five ordered, unlabeled, fixed-cell images with one moving Li and a fixed framework. The four reserved DFT labels remained sealed. The public summary records counts and checksums. Material-to-slot mappings, structural match identities, and seed geometries are retained in the local audit directory.

## Label readiness

All 20 candidates require review for at least one element outside the existing QE test scope used by this audit. Per-path element requirements are recorded in the local manifest. Before production labeling, verify pseudopotentials and magnetic or charge branches, establish endpoint and path convergence, and record numerical comparisons and cost for the target protocol. Use the preserved A-to-B ordering and score the forward barrier as the converged path maximum minus the A endpoint energy.

The 12-pilot and 8-reserve allocation is an engineering choice. For scale, the dispersion of 42 family-level effects from the existing FP-DeErr single-path experiment gives idealized 95% half-widths of 0.33 meV for absolute gain and 0.57 meV for net gain with 12 independent families. These calculations assume the same between-family dispersion and equal weighting. They describe precision under that historical scenario. The variance and sample requirement for a future strategy comparison remain to be estimated.
