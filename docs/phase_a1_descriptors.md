# Phase A1 Li-centered representation pilot

The pilot uses the 34 audited images alongside the unchanged 278-frame D₀ archive. Initial and relaxed images retain independent reaction coordinates and an explicit source-stage marker. Only the 17 initial geometries are available for DFT-free selection. The feature extractor passes positions, species, cell and periodicity to the descriptor models. Energies, forces and historical path gains are outside the feature inputs.

| Family | Representation | Distance to D₀ |
| --- | --- | --- |
| Geometry | Element-blind smoothed local radial density and element-blind SOAP | Cosine distance |
| Chemical environment | Per-element hard and smooth coordination, local radial density and element-resolved SOAP | Euclidean for counts, cosine for radial density and SOAP |
| Model | Invariant features from the interaction layers of one fixed seed-17 D₀ MACE checkpoint | Cosine distance |

Hard coordination counts minimum-image sites within 3.0 Å. Smooth coordination uses a cosine switch from 2.6 to 3.4 Å. The local radial profiles use Gaussian broadening and spherical shell area normalization, as specified in the feature manifest. The two SOAP representations share one fixed configuration. Element-resolved SOAP uses DScribe's [mu1nu1 compression](https://singroup.github.io/dscribe/latest/tutorials/descriptors/soap.html). These are pilot configurations. Parameter sensitivity and descriptor agreement belong to A2.

The NPZ contains raw feature arrays indexed by `feature_index.csv`. The novelty CSV records nearest distances to the raw bank and the A0.1 3.0 Å representatives, plus a median of per-material nearest distances in the raw bank. The representative comparison tests sensitivity to reference reduction. A0.1 deduplication uses a geometry fingerprint and does not imply equivalent SOAP or MACE embeddings. The distances remain separate by representation.

The invariance table records repeats, rotations, permutations, and translations followed by periodic wrapping of the central image of each stage and path. The descriptors use the archived seed-17 D₀ weights. Validation is conditional on the tested input cases and model.

## Reproduction

Use Python with ASE, NumPy, SciPy, Matplotlib, PyTorch, MACE and DScribe 2.1.2. Specify new output directories. The checkpoint archive supplies `D0_17.pt` and `models.json`. Model identity is verified before loading.

```text
python src/phase_a01_coverage.py NEB_ZIP INDEX_CSV PHASE1_BUNDLE RESULTS_A0 NEW_RESULTS_A01
python src/phase_a1_descriptors.py NEB_ZIP INDEX_CSV PHASE1_BUNDLE RESULTS_A0 RESULTS_A01 CHECKPOINTS_ZIP NEW_RESULTS_A1 --threads 4
```
