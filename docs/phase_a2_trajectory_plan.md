# Phase A2 trajectory novelty robustness

## Evidence and scope

Use the three A0 geometry cases, A0.1 reference audit and A1 descriptors as a fixed baseline. Compare Init and Relaxed within each path and representation. Only Init is admissible for future DFT-free selection. Hold the archived seed-17 D₀ MACE checkpoint fixed. Keep the three paths as case studies. Do not tune on gains, refit MACE, or recalculate DFT.

## Parameters

| Representation | A1 baseline | A2 variants |
| --- | --- | --- |
| SOAP, element-blind and element-resolved | cutoff 4.0 Å, sigma 0.40 Å, n 5, l 3 | cutoff 3.5 / 4.5 Å, sigma 0.30 / 0.50 Å, n 4 / 6, l 2 / 4, plus cutoff 3.5 Å × sigma 0.30 Å |
| Local radial density, element-blind and element-resolved | grid 0.15 Å, Gaussian width 0.12 Å | grid 0.10 / 0.20 Å, width 0.08 / 0.18 Å |
| Hard and smooth coordination | 3.0 Å center; smooth transition ±0.4 Å | centers 2.8 / 3.2 Å |
| MACE interaction features | combined invariant output | layer 1 only, layer 2 only |

Each parameter variant uses the full unchanged reference bank. Comparison of reference policies takes place at the A1 parameter baseline. Keep distances in their original units. Cosine distances and coordination-count differences remain separate.

## Reference policies

- Use the raw and the A0.1 3.0 Å deduplicated D₀ bank.
- Draw 16 fixed-seed equal-material banks without replacement from the deduplicated pool. The quota is the smallest material size, capped at 8. Keep sample-wise metrics and summarize nearest distances by median and interquartile range. Draw equally sized unstratified banks from the same pool as a control for the smaller bank.
- Starting with raw D₀, exclude the target material, exact element system, or anion family in separate evaluations. Chemistry and anion family use the audited Init metadata. This is a family proxy based on chemistry. Record the number of removed environments and materials including no-op exclusions.

## Peaks and comparisons

Use piecewise-linear interpolation with original knots and a 0.005-step common normalized grid. Record the maximum raw distance and its coordinate. Compute excess relative to the straight line joining endpoint distances. Record its maximum, peak coordinate, full width at half excess for the connected peak, and integrated positive excess. Report ties. The diagnostic central-feature flag requires the excess peak at s = 0.20–0.80 and excess at least 10% of the raw profile maximum.

Pair stages by edge, representation and setting. Report both excess-peak change and the mean absolute distance-profile shift on s. Report Cr SOAP/MACE central-peak alignment within 0.10 normalized coordinate as descriptive, with each SOAP setting and matched reference bank. The baseline combined MACE is the comparison for SOAP parameter variants; the separate MACE layers are a sensitivity check. Parameter and reference repetitions do not increase the independent path count.

## Outputs

Produce the parameter and reference tables with image identities, peak and stage comparison tables, robustness summary, alignment table, and a report with observed runtime and input provenance. Compare A2 baseline profiles to A1 before interpreting the results.
