# Phase A2 trajectory novelty robustness

The scope is three case-study paths. All profiles reuse the A0/A1 geometries and one frozen
seed-17 MACE checkpoint. Only Init is available for DFT-free use. No gain labels enter this analysis.

Recomputed A1 raw and deduplicated profiles with maximum absolute disagreement of 1.33e-15. Total CPU run time was 185 seconds wall clock. MACE took 172 seconds and the largest single SOAP job took 2.0 seconds.

## Cr–Ga–Li–O central peaks

| Representation | Comparison | Init central | Relaxed central | Excess smaller after relaxation |
| --- | --- | ---: | ---: | ---: |
| rdf_geometry | parameters | 5/5 | 5/5 | 5/5 |
| rdf_geometry | reference_policies | 5/5 | 5/5 | 5/5 |
| rdf_geometry | balanced_draws | 16/16 | 16/16 | 16/16 |
| soap_geometry | parameters | 10/10 | 10/10 | 10/10 |
| soap_geometry | reference_policies | 5/5 | 5/5 | 5/5 |
| soap_geometry | balanced_draws | 16/16 | 16/16 | 16/16 |
| soap_elements | parameters | 10/10 | 10/10 | 10/10 |
| soap_elements | reference_policies | 5/5 | 5/5 | 5/5 |
| soap_elements | balanced_draws | 16/16 | 16/16 | 16/16 |
| mace_invariants | parameters | 3/3 | 3/3 | 3/3 |
| mace_invariants | reference_policies | 5/5 | 5/5 | 5/5 |
| mace_invariants | balanced_draws | 16/16 | 16/16 | 16/16 |

Excess is measured above the line between endpoint distances. On raw D₀ with the
baseline parameters, peak excess declined by 70.0% (mace_invariants) and 96.5% (soap_geometry).
Parameter settings
and reference draws are sensitivity cases. Independent material count remains three.

## Reference exclusions

| Path | Target sites removed | Exact-system sites removed | Anion-family sites removed |
| --- | ---: | ---: | ---: |
| mp-753143_5_13_0_0_0 | 0 | 0 | 1957 |
| mp-1177833_2_3_0_0_1 | 0 | 0 | 1327 |
| mp-768690_0_0_1_0_0 | 0 | 0 | 1327 |

Zero means the exclusion reproduces the raw bank. The anion-family knockout changes
the reference population. Equal-material thinning uses the same quota for every material,
alongside a matched-size unstratified control. Thinning draw summaries are descriptive.

## SOAP–MACE alignment

- The geometric SOAP grid on Cr–Ga–Li–O aligns with the combined MACE central peak at 10/10 settings on init. The tolerance is 0.10 in normalized coordinate.
- The geometric SOAP grid on Cr–Ga–Li–O aligns with the combined MACE central peak at 3/10 settings on relaxed. The tolerance is 0.10 in normalized coordinate.

A center-position match does not identify the same atomistic mechanism. Five to seven
native points limit peak-width resolution. Frequency across these descriptors cannot establish
prediction of training value or generalization to independent paths.
