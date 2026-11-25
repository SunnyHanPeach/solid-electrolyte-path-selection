# Phase A0 periodic geometry and reference coverage

The reference bank contains 9,517 Li environments from 278 canonical D₀ frames.
Each path has separate Init and Relaxed results. Init carries BVSE energies. Relaxed carries DFT energies and force labels.
All counts summarize cases. Images and Li sites within a frame are correlated observations.

## Integrity

| Path | Stage | Images | Li row | Arc (Å) | Steps with a different geometry assignment |
| --- | --- | ---: | ---: | ---: | ---: |
| mp-753143_5_13_0_0_0 | init | 5 | 1 | 3.202 | 0 |
| mp-753143_5_13_0_0_0 | relaxed | 5 | 1 | 2.934 | 0 |
| mp-1177833_2_3_0_0_1 | init | 5 | 1 | 3.702 | 0 |
| mp-1177833_2_3_0_0_1 | relaxed | 5 | 1 | 3.768 | 0 |
| mp-768690_0_0_1_0_0 | init | 7 | 1 | 5.477 | 0 |
| mp-768690_0_0_1_0_0 | relaxed | 7 | 1 | 5.251 | 0 |

Init archives supply the atom index, offset and unique unfrozen site. The relaxed archive lacks
these IDs. Relaxed rows are checked by species and adjacent minimum-image assignment; this is
geometric evidence for the archived ordering. The indexed site has the longest Li arc in every stage.
Init–Relaxed framework matching holds the migrating Li fixed.
Cross-stage shifts use archived image pairing and record each side's independently normalized path coordinate.

## Coordination and environment coverage

Each atomic site contributes its closest periodic image. For each species, shell 1 starts at the nearest site.
A gap above 0.35 Å starts a new radial group. Shell counts are also reported at 0.25 and 0.45 Å.
The search extends to 5 Å. An empty second group or a shell near that boundary should be interpreted in that context.
Wide groups can form when consecutive distances have small gaps. Reported group counts need not be bond counts.
The image CSV records entrants and leavers by archived atom row. The path CSV records how many images
change at least one element-specific count across the three shell gaps.

Local fingerprints require matching element counts. Their distance is the larger of the radial and
inter-neighbor distance RMS differences after sorting within the relevant species classes. Neighbor
vectors are unwrapped about the center. This detects similarity in the represented distances;
different environments can share a fingerprint. It is a screening measure, rather than an atom-wise match.
A site outside the chosen radius changes the fingerprint membership. Results for three radii show this sensitivity.
The 0.03, 0.10 and 0.20 Å tolerances are applied to the fingerprint distance, with 0.03 Å
as a tight scale informed by prior project geometry checks. The metrics differ from the previous
cell-level maximum residual. Matched environments and matched D₀ frames are counted separately.

| Path | Stage | Radius (Å) | Comparable internal pairs | Within tolerance (0.03 / 0.10 / 0.20 Å) | Images with comparable D₀ environments | Images covered at tolerance |
| --- | --- | ---: | ---: | --- | ---: | --- |
| mp-753143_5_13_0_0_0 | init | 2.8 | 0/10 | 0 / 0 / 0 | 3/5 | 0 / 2 / 2 |
| mp-753143_5_13_0_0_0 | init | 3.0 | 0/10 | 0 / 0 / 0 | 1/5 | 0 / 1 / 1 |
| mp-753143_5_13_0_0_0 | init | 3.2 | 0/10 | 0 / 0 / 0 | 3/5 | 0 / 1 / 1 |
| mp-753143_5_13_0_0_0 | relaxed | 2.8 | 0/10 | 0 / 0 / 0 | 4/5 | 0 / 2 / 2 |
| mp-753143_5_13_0_0_0 | relaxed | 3.0 | 0/10 | 0 / 0 / 0 | 2/5 | 0 / 1 / 1 |
| mp-753143_5_13_0_0_0 | relaxed | 3.2 | 0/10 | 0 / 0 / 0 | 3/5 | 0 / 1 / 1 |
| mp-1177833_2_3_0_0_1 | init | 2.8 | 2/10 | 2 / 2 / 2 | 5/5 | 0 / 2 / 4 |
| mp-1177833_2_3_0_0_1 | init | 3.0 | 2/10 | 1 / 1 / 1 | 5/5 | 0 / 2 / 4 |
| mp-1177833_2_3_0_0_1 | init | 3.2 | 2/10 | 2 / 2 / 2 | 4/5 | 0 / 0 / 2 |
| mp-1177833_2_3_0_0_1 | relaxed | 2.8 | 6/10 | 2 / 2 / 6 | 5/5 | 0 / 2 / 4 |
| mp-1177833_2_3_0_0_1 | relaxed | 3.0 | 6/10 | 2 / 2 / 6 | 5/5 | 0 / 2 / 4 |
| mp-1177833_2_3_0_0_1 | relaxed | 3.2 | 6/10 | 2 / 2 / 6 | 5/5 | 0 / 0 / 4 |
| mp-768690_0_0_1_0_0 | init | 2.8 | 3/21 | 3 / 3 / 3 | 2/7 | 0 / 2 / 2 |
| mp-768690_0_0_1_0_0 | init | 3.0 | 3/21 | 3 / 3 / 3 | 0/7 | 0 / 0 / 0 |
| mp-768690_0_0_1_0_0 | init | 3.2 | 2/21 | 2 / 2 / 2 | 0/7 | 0 / 0 / 0 |
| mp-768690_0_0_1_0_0 | relaxed | 2.8 | 3/21 | 2 / 3 / 3 | 2/7 | 0 / 2 / 2 |
| mp-768690_0_0_1_0_0 | relaxed | 3.0 | 3/21 | 2 / 3 / 3 | 0/7 | 0 / 0 / 0 |
| mp-768690_0_0_1_0_0 | relaxed | 3.2 | 2/21 | 2 / 2 / 2 | 0/7 | 0 / 0 / 0 |

## Change on relaxation

| Path | Largest Li shift (Å) | Largest image framework RMS (Å) | Images with a different framework assignment |
| --- | ---: | ---: | ---: |
| mp-753143_5_13_0_0_0 | 0.157 | 0.067 | 0 |
| mp-1177833_2_3_0_0_1 | 0.281 | 0.213 | 0 |
| mp-768690_0_0_1_0_0 | 0.252 | 0.195 | 0 |

Each case plot shows its Li trajectory, anion distances and shell counts,
cross-stage shifts, D₀ coverage, and comparable within-path fingerprint distances.
