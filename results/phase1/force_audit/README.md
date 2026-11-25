# Force-error audit of the P0 contrast cases

The RTX 4090 evaluated both saved checkpoints for each of five arms and three seeds. Forces were scored locally against the same 107 materials used in P0. Independent agreement between the Init moving-Li row and the largest relaxed-path lithium displacement was required for the moving-Li metric. This retained 106 materials for that metric and all 107 for the all-atom metric.

| Candidate minus matched replay, final checkpoint | High interior Li force RMSE difference (eV/Å) | High interior all-atom RMSE difference (eV/Å) |
| --- | ---: | ---: |
| Co–Cr–Li–O | +0.000503 [−0.000123, +0.001154] | −0.000493 [−0.005616, +0.003527] |
| F–Li–Ni–O | −0.000025 [−0.000187, +0.000137] | +0.000183 [−0.000179, +0.000538] |

The signs and intervals leave both high-interior force-error contrasts unresolved. In this exploratory comparison, the force RMSE does not explain the different P0 net barrier gains. The analysis compares predicted forces evaluated at fixed DFT geometries. Projection along the full reaction path and training gradient alignment remain distinct diagnostic questions.

Intervals resample exact chemical systems. The bootstrap and summary retain the pairing across seeds and materials.
