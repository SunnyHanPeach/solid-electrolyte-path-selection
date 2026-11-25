# Phase A0.1 coverage sensitivity and reference composition

The reference bank is the unchanged 278-frame D₀. Greedy deduplication operates within each
material and element-count pattern, in original frame and Li order. Every removed site is
within 0.03 Å fingerprint distance of a retained site. Cross-material memberships are preserved.
The exact same Li-centered distance fingerprint from A0 is used throughout.

Per-image nearest distance and any-match coverage describe the existence of a close reference.
Material-balanced support averages the per-material fraction of matching Li environments,
giving each material equal weight and assigning zero to materials with no matching environment.
Pooled support weights every reference environment equally. Images do not constitute independent paths.

| Radius (Å) | Raw Li | Deduplicated Li | Materials |
| --- | ---: | ---: | ---: |
| 2.5 | 9,517 | 2,052 | 50 |
| 2.8 | 9,517 | 2,166 | 50 |
| 3.0 | 9,517 | 2,469 | 50 |
| 3.2 | 9,517 | 2,537 | 50 |
| 3.5 | 9,517 | 2,869 | 50 |

The A0 first and second radial groups change under the 0.25–0.45 Å gap scan in 19/34 images. A1 retains the continuous radial distributions alongside coordination counts.

See the sensitivity figure and CSVs for each path and stage.
