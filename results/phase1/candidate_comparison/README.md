# Phase 1 comparison on the 26-material panel

The comparison evaluated all seven completed paths using the common seeds 17, 29 and 43. It reused the 45 archived checkpoints and the geometry from the earlier mp-753143 replication. The path and replay arms match atom counts and exposure for each candidate. All candidates share the same pure D₀ checkpoints. Positive gain means lower forward barrier MAE after adding the path. The 95% intervals resample exact chemical systems and hold the seeds fixed. The panel was previously scored for mp-753143, so the seven-path comparison is exploratory.

| Candidate | Gain over matched replay (meV) | Gain over pure D₀ (meV) |
| --- | ---: | ---: |
| mp-1277232 | −0.11 [−9.85, +9.19] | −3.50 [−14.00, +4.14] |
| mp-13725 | −0.16 [−0.89, +0.48] | −0.66 [−1.74, +0.14] |
| mp-753143 | +5.61 [+0.55, +14.46] | +5.46 [+0.23, +15.31] |
| mp-753520 | +0.19 [−2.26, +2.38] | −0.27 [−1.42, +0.89] |
| mp-758054 | −3.69 [−9.13, −0.38] | −3.32 [−8.88, −0.54] |
| mp-759652 | −0.66 [−4.85, +2.61] | −0.81 [−3.90, +1.65] |
| mp-768690 | −9.16 [−19.30, −0.49] | −9.59 [−20.21, −0.28] |

mp-753143 alone cleared the +0.25 meV practical gate against its replay. No candidate cleared it against pure D₀. Omitting mp-1277814 left the mp-753143 net gain at +2.32 meV [0.55, 4.73]. Its net mean was positive for each omission of one chemical system, with a range of +1.45 to +6.17 meV. This common-seed estimate supplements the separate five-seed replication. The comparison has seven candidate paths and does not estimate the frequency of beneficial paths in a wider pool.

[Locked inputs](lock.json) · [Results and sensitivity](summary.json) · [Paired results by material](paired_materials.csv)
