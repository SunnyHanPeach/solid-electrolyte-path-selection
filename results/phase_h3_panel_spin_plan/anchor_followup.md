# H3 fixed-geometry anchor checks

## Initial image

All three SCFs completed on 28 November 2026. The logs and XML agree on convergence and contain 105 finite force components per branch. Atom order, cell and coordinates match the frozen A geometry. The neutral fixed-moment and free-moment runs took 7,506 and 7,899 seconds. The negative branch took 3,169 seconds. Raw inputs, XML, logs, timings and per-atom forces are kept in the local archive with SHA-256 checksums.

The free-moment neutral solution converged to 0.96 μB, compared with the constrained 1.00 μB. Its total energy is 0.64 meV lower. The force-component RMSE between them is 0.00057 eV/Å and the largest atomic force-vector difference is 0.00203 eV/Å. These measurements apply to A at the tested grid, cutoff and smearing.

The qualification reference stays at neutral total charge and 1 μB to match the QE-a convention. Formal vacancy bookkeeping allows an electron-filled negative branch. Its convergence at A establishes numerical feasibility for that calculation. The stable physical charge state depends on defect formation energies, the electron chemical potential and finite-size terms. [QE charge convention](https://www.quantum-espresso.org/Doc/INPUT_PW.html) · [Point-defect thermodynamics](https://doi.org/10.1103/RevModPhys.86.253)

## Baseline continuation

The local queue started on 28 November at **19:37:02 (UTC+08:00)**. It evaluates the image closest to half of the Li arc, then endpoint B. A is reused. The three anchors use the same neutral branch, PBE, verified UPFs, 60/480 Ry, a Γ-centered 3×2×3 grid and 0.10 eV Gaussian smearing. The queue uses the same verified QE binary with eight MPI ranks and two k-point pools, one job at a time.

The denser 4×3×4 grid and the separate 75/600 Ry series are staged for all three anchors. Their relative energies should be compared with the baseline relative to both endpoints. The qualification limits are 10 meV for relative energies and 0.020 eV/Å for the full-atom force-component RMSE and the moving-Li force-vector difference. Changes to spin character require review. The interior anchor was selected by arc position before labels and does not determine the saddle point.
