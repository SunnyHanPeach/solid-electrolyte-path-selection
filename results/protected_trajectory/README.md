# Protected D₀ training trajectory

Seeds 17 and 29 each ran one eight-path augmentation and one exposure-matched D₀ replay. The two arms shared 710 frozen D₀ updates, 97 matched insertion slots, 807 total updates, and a single runtime. The saved steps are 8, 64, 200, 400, 600, and 807. Blind forward barriers were evaluated on fixed DFT-NEB images.

- Positive MAE differences indicate degradation relative to matched replay.
- Step 400 is the first saved main-panel checkpoint to clear the planned gate, followed by steps 600 and 807. The gate requires both seed means and the paired material bootstrap lower bound to exceed 0.25 meV. The observed main-panel endpoint difference is +30.09 meV.
- The historical 88-material panel had a −2.40 meV endpoint point estimate with a wide interval. Its material-wise distribution is retained.
- Checkpoint hashes document 24 successful local deserializations with model and Adam state. The checkpoint archive and original DFT input bundles are stored separately.
