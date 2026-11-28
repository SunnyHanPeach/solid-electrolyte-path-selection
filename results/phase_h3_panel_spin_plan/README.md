# H3 independent evaluation panel · fixed-geometry checks

The Protocol Qualification Set supplies one primary and one backup. Both stay outside confirmatory evaluation. The 12 existing pilot positions and eight reserves retain their order.

## Selection and geometry checks

Screen for complete UPF header checks, no magnetic-element flag, no flagged local overlap, and scalar or nonrelativistic reference UPFs with a resolved spin–orbit flag. For this initial collinear check, defer chemistries containing elements with atomic number 80 or higher. This is an engineering triage rule. It does not establish that spin–orbit effects are negligible. Pick the smallest remaining cell by atom count, using valence electron count and the frozen slot for ties. Pick the first eligible candidate in a different source stratum as the backup. The result is one sulfide and one halide. No model result or barrier enters the order.

Reconstruct both defect endpoints from the frozen parent and the public centroid-marked geometry. Verify the integer supercell transform, atom order, moving Li index, unique A and B sites and the absence of exactly one Li relative to the parent. Compare periodic coordinates to the frozen seeds. Keep the exact stoichiometries, slot assignment, coordinates, starting grids and file hashes in the local audit.

## Defect and spin choices

Use the neutral vacancy supercell, QE `tot_charge=0`, as the primary qualification convention because it matches QE-a. It has an odd valence-electron count in both materials. Formal ionic bookkeeping for the reconstructed parents yields a singly negative empty Li site. A neutral-electron-count calculation therefore contains a compensating hole relative to the electron-filled `tot_charge=-1` branch. The site or degree of localization of that hole remains a calculation result.

| Branch at the same A geometry | QE choices | What to inspect |
| --- | --- | --- |
| Neutral reference | `tot_charge=0`, `nspin=2`, `tot_magnetization=1.0` | Constrained collinear branch continuous with QE-a |
| Neutral independent initialization | `tot_charge=0`, `nspin=2`, unconstrained total magnetization, initialize spin on the anion nearest the empty Li site | Sensitivity to initialization and any distinct spin density, moment or lower-energy state |
| Negative comparison | `tot_charge=-1`, `nspin=1` initially | Even-electron branch with the compensating periodic background. Check for a stable, gapped reference. An ambiguous occupation calls for an explicit spin-polarized check. |

For the local initialization, use a separate atomic type for the selected anion with the same UPF and a fractional `starting_magnetization` of 0.15. Omit the fixed-magnetization field on that run. Compare the two neutral runs using the same smearing, numerical settings and geometry. Archive total and absolute magnetization and site or real-space spin diagnostics. Start each branch independently so that a previous charge density does not decide which solution is found. Charged results have their own energy zero. A comparison of migration profiles would use energy differences referenced to A separately within each charge sector. Ground-state charge assignment would also require the material's defect thermodynamics and finite-size treatment. [QE PW input reference](https://www.quantum-espresso.org/Doc/INPUT_PW.html) · [Charged-defect finite-size corrections](https://doi.org/10.1103/PhysRevLett.102.016402)

## Execution gates

1. **First allocation: three fixed-geometry SCFs at A on the primary only.** Use PBE, the hash-verified SSSP references, the per-material cutoff pair from the qualification table, Gaussian smearing 0.10 eV, `conv_thr=1e-8 Ry`, and full forces. Choose a Γ-centered grid from the defect supercell's reciprocal vector lengths at an initial spacing of 0.30 Å⁻¹. The backup remains staged. Keep the full timings and logs.
2. **Review the electronic states before adding images.** Check SCF convergence, finite force arrays, occupation near the Fermi level and reproducible spin character. If the neutral solutions differ materially, preserve both and assess branch consistency before choosing a production reference. If a charge sensitivity test is needed, calculate the midpoint and B in both charge sectors with internally consistent references. Delay any charged production label until the barrier's supercell-size and correction sensitivity are established.
3. **Qualify the fixed-geometry reference.** Extend the selected neutral branch to B and the interior image closest to half of the Li arc. This image is chosen without a barrier label. Compare the three images with a denser Γ-centered grid and separately with both cutoffs raised by 25%. Require relative-energy changes of at most 10 meV, all-atom force-component RMSE of at most 0.020 eV/Å and moving-Li force-vector difference of at most 0.020 eV/Å. Compare energy differences relative to both endpoints. Log the actual force and timing per image. Treat changing spin character, numerical failure or a discrepancy above a limit as a stop for review.
4. **Keep the release separate.** Only after this review can a pilot endpoint-relaxation and path-validation request be considered. No DFT was executed in this preparation. No additional complete NEB was started, and the current QE-a queue and four reserved labels were left untouched.
