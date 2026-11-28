# Public DFT costing baseline and project comparison

## Public source

Transcribed the rightmost relative-wall-time column of Fig. 2 from the versioned preprint. The values are aggregate QE ratios for the authors' 2D and 3D metallic benchmark. The study uses small cells of at most 14 atoms. The 116 KiB Materials Cloud plotting package was verified against the catalogue MD5. Its JSON files contain convergence fractions and do not provide per-job timing. Individual workflow records are in the separate AiiDA archive. [Figure source](https://arxiv.org/pdf/2504.03962v1#page=10) · [Journal article](https://doi.org/10.1038/s41524-026-02097-8) · [Materials Cloud release](https://archive.materialscloud.org/record/2025.62)

The Cu/TaN VASP archive provides a separate extraction example. The catalogue MD5 and all 11 per-image OUTCAR records across three paths were verified. Every record has its own elapsed time and parallel header. Some paths reuse endpoints, and internally parallel images can share a job. The per-image elapsed times therefore need workflow grouping and scheduler allocation before estimating a total acquisition cost. The extraction does not supply cost coefficients for lithium vacancy migration. [Raw Zenodo archive](https://zenodo.org/records/15604106) · [Extracted timing fields](public_neb_outcar_timings.csv)

## Independent project checks

The frozen `eval-06` cell gives maximum reciprocal grid spacings of 0.289139 and 0.216854 Å⁻¹. Log interpolation of the transcribed aggregate gives a prediction of **1.49765×** for the denser grid. The measured baseline seconds serve only as the time unit for the individual forecasts. The two resulting dense forecasts are 11,241 s for A and 11,268 s for the midpoint. Their measured dense times are pending at freeze. The 35-atom vacancy, spin treatment, PBE, executable and parallel layout differ from the public benchmark. [Prediction receipt](eval06_walltime_predictions.csv)

`mp-1191903` remains the end-to-end calibration trial. Its phase costs are grouped under the ledger before forming hardware-specific hours. Its electronic/ionic iteration counts, failure and restart costs should be compared with the public prior before fitting any adjustment. [QE-a ledger](../phase_h3_qe_a/ledger.json)

## Procurement sensitivity

Use the public 0.30 to 0.20 Å⁻¹ cost ratio and an isotropic cubic spacing-to-count approximation to derive a nominal grid exponent of 0.48322. Replace only the grid exponent in the existing H2 middle scenario. The local reference SCF time and all other stage assumptions are retained so the comparison isolates this one change. The 20 eligible candidates, six ranking rules and three nominal budgets are unchanged.

The overlay changes **4 of 18** decisions, all at the 267.33-hour planning budget. At **133.67 hours**, all six original choices are reproduced. SOAP selects two paths, SOAP+Activation one, with no overlap. There are five unique paths across all six strategies, with shared paths physically acquired once. These are provisional combinations under the assumed costs. [Candidate cost overlay](candidate_cost_overlay.csv) · [Strategy replay](procurement_replay.csv)
