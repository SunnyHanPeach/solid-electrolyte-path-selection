# H3 evaluation panel qualification

The 12 pilot paths are designated the **Protocol Qualification Set**. The eight reserves retain their original priority. Any path used to tune the labeling protocol, acquisition strategy, or model is excluded from the final confirmatory set. The 20 geometries were drawn from the BVSE prescreened pool, and the audit did not reorder them. There are currently zero production-qualified paths and zero newly labeled primary evaluation paths.

## Pseudopotentials and electronic states

All 33 required elements are covered by the official SSSP 1.3.0 PBE Efficiency archive. The full archive MD5 and each selected reference file MD5 match the catalogue. The local inventory records SHA-256, UPF version, functional, valence, orbital occupations, relativity, and both reference cutoffs. The per-path register links the elemental checks to each stoichiometry. Combining the catalogue and header recommendations gives starting references of **40–90 Ry** for wavefunctions and **320–1,080 Ry** for density. The files were inspected without installing or substituting a production UPF. [Materials Cloud SSSP release](https://archive.materialscloud.org/records/rcyfm-68h65)

With the neutral-vacancy convention, all 20 seed cells have odd valence-electron counts. Six pilots and three reserves also contain elements flagged for explicit magnetic review. The elemental screen does not assign oxidation states, magnetic ordering, moments, or the defect charge state. Qualification must justify the physical charge model and check appropriate spin branches and cutoff and k-point convergence for the selected chemistry. A non-spin-polarized smeared solution alone does not establish the reference electronic state. [Quantum ESPRESSO pw.x input reference](https://www.quantum-espresso.org/Doc/INPUT_PW.html)

## Local overlap

The audit ran 1,810 path-pair comparisons across the 20 candidates, 24 H1 paths and 57 FP-DeErr reference paths. Five of the 62 FP-DeErr references had ambiguous migrating-Li assignments in the relaxed geometries and were logged locally. The moving Li must displace at least 0.75 Å and at least twice as far as any other atom between the relaxed endpoints to enter that comparison. The exclusion applies only to this local-overlap audit.

The screen uses periodic neighbor vectors at A, the image closest to the middle of the Li arc, and B. It compares sorted Li-neighbor and neighbor-pair distances at each anchor and requires compatible neighbor counts. It tests both path directions with and without elemental identities. Per-image neighbor counts and minimum non-Li contact distances are preserved locally. The contact distance is a geometric proxy. It does not define a physical bottleneck radius.

At 3.0 Å radius and 0.10 Å tolerance, no pair matched all three anchors. At 2.8 Å and 0.15 Å without elemental identities, one candidate matched one H1 path on all three anchors. This pair stays flagged for review. No candidate-to-candidate pairs matched all three anchors over the tested grid. Reversed, rotated, translated, and periodically wrapped copies of four paths passed all 24 positive-control comparisons. The results screen for geometric redundancy within the tested representations. Differences in occupancy, shell membership, and the use of relaxed rather than initial FP-DeErr geometries can hide related migration environments. [Sensitivity grid](overlap_sensitivity.csv)

## Release decision

The next useful DFT gate is a small fixed-geometry check on one or two pilot systems after the defect charge, spin branches, and referenced UPFs are reviewed. Use endpoints and an interior image to check relative energies, forces and convergence before a pilot CI-NEB. Estimate confirmatory sample size later from paired differences between the actual acquisition strategies across independent families. The 12 pilot slots and eight reserves carry no statistical power guarantee.

Only aggregated counts and code are published. Per-path details, nearest matches and hashes are retained locally. The four reserved labels remain sealed and the existing cloud job was undisturbed.
