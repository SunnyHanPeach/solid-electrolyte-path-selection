# Second-block force weighting replication

All eight frozen arms finished and passed local scoring at steps 400 and 807. The second eight-path training block was disjoint from the strength-selection block by material ID. Both evaluation panels were reused from earlier experiments. The model uses the same initialization and training configuration across all four arms.

| Step 807 comparison (treatment minus control, meV) | Primary 91 | Auxiliary 88 |
| --- | --- | --- |
| Full minus matched replay | +10.57 [−1.92, +21.93] | −12.88 [−49.34, +14.07] |
| Matched uniform minus full | −2.79 [−3.94, −1.63] | −2.12 [−4.25, +0.17] |
| Energy-dependent minus full | −2.79 [−4.06, −1.49] | −2.08 [−4.43, +0.53] |
| Energy-dependent minus matched uniform | −0.003 [−0.221, +0.245] | +0.036 [−0.264, +0.447] |
| Energy-dependent minus matched replay | +7.78 [−4.06, +18.51] | −14.97 [−49.91, +10.36] |

Both lower-force treatments reproducibly improved over the original augmentation on the primary panel. The energy-versus-uniform estimate was close to zero, and its seed contrasts had opposite signs. The full primary-panel material interval is within the 0.25 meV practical band, conditional on the two seeds. Energy-dependent weighting did not meet either predeclared superiority gate. The two primary-panel seed contrasts versus matched replay were +7.54 and +8.03 meV. The auxiliary replay contrast remains unresolved.

The primary mean signed barrier error shifted by +4.86 meV versus full and −24.53 meV versus replay under energy-dependent weighting. The evidence supports attenuation of the error increase from raw augmentation on this panel and does not establish net utility over matched replay or an advantage from targeting high-energy frames across blocks. The discount remains a recorded experimental setting rather than a general recipe choice.

The compact archive matched its cloud SHA-256. The scoring implementation checks exact recipe, schedule and frame identities, exact force multipliers at training float32 precision, and a 1e−12 absolute tolerance for derived double-precision exposure totals. The scorer correction addresses cross-Python reduction rounding and required no new training. Prediction checksums and the locked DFT reference checksum were also verified.
