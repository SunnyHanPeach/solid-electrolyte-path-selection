# Phase A0 geometry integrity and reference coverage

## Inputs and identity

Read three archived paths from nebDFT2k, the existing moving-Li index, and the canonical first 278 frames of the frozen Phase 1 bundle. Verify file and member hashes and the D₀-only training schedule. Keep each edge and stage separate. Keep the source energy and force provenance with the geometry. Treat Init energies as BVSE values and Relaxed energies and forces as DFT labels.

For Init, check the explicit site IDs and periodic offsets in the source. For Relaxed, retain the archived row identity and verify it against species-preserving minimum-image geometry assignments between adjacent frames, without relabeling atoms. Audit cross-stage framework correspondence separately with the migrating Li held fixed in the matching. Report the verification method and any ambiguity or alternative assignment.

## Geometry and coverage

- Compute minimum images with ASE for skew cells. Integrate consecutive moving-Li displacements and divide by the full arc length. Keep cumulative arc length and displacement from the start as distinct quantities.
- Define element-specific radial shells by splitting ordered distances at gaps. Report the first two groups within 5 Å using a 0.35 Å central gap and 0.25 and 0.45 Å sensitivity. Count each lattice site once using its closest image. The resulting shells describe these geometries. They do not assign bonds or transfer a fixed coordination boundary between elements.
- Count identities entering and leaving each shell between adjacent frames separately from changes in coordination number. Compare Init and Relaxed geometries on both the input image pairing and their own reaction coordinates.
- Match Li-centered environments against all Li in the 278 D₀ frames. Compare matching element counts within the local cutoff, sorted center distances by element, and sorted inter-neighbor distances by unordered element pair. Use the maximum of the radial and pair-distance RMS differences in Å. Keep separate counts for 2.8, 3.0 and 3.2 Å radii, and for 0.03, 0.10 and 0.20 Å tolerance at 3.0 Å. The 3.0 Å scale matches the previous contact analysis. The tolerances measure fingerprint sensitivity and do not certify atomic equivalence. Compare internal image pairs only within a path and stage. Allow an unmatched fingerprint.
- Keep normalized path coordinates, element-specific radial distances, shell changes, reference matches and visual comparisons as reproducible outputs. Keep path and stage as the reporting units.

## Verification

Run synthetic skew-cell and boundary-crossing checks, site-label and species-permutation checks, empty and split shell cases, coordination exchange and fingerprint symmetry checks. Verify the raw three-case output against input cardinality, periodic displacements, mapping status, D₀ provenance and Cartesian force completeness. Keep all outputs in a new directory. Run on CPU with Python, ASE, NumPy, SciPy and Matplotlib.

From the repository root, supply the original NEB ZIP, the audited Init feature index and the canonical Phase 1 bundle. The output directory must be new.

```text
python -m unittest discover -s tests -v
python src/phase_a0_geometry.py NEBDFT2K_ZIP INIT_FEATURES_CSV PHASE1_BUNDLE_ZIP NEW_OUTPUT_DIRECTORY
```
