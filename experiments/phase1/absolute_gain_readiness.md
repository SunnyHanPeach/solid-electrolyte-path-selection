# Absolute-gain experiment readiness

**Decision on 2026-11-25**

Keep the absolute endpoint and leave candidate selection and the confirmatory experiment pending.

- **Endpoint.** Compare forward barrier MAE against the canonical pure D₀ model. Keep matched replay as a secondary comparison. The difference between the two gains is exactly the replay-minus-pure error. Its interpretation depends on the chosen replay schedule.
- **Training budget.** The existing mp-753143 runs used 710 pure D₀ updates and 735 path or replay updates. The three controls answer the appropriate questions for that design. For a fixed-update-budget comparison, draft a 710-step protocol that substitutes a predetermined set of D₀ image slots with complete-path images or matched replay. Check per-batch image and atom counts in all three arms. The estimand then includes the replacement of some D₀ exposure.
- **Independence.** The earlier 38-path feature table contains 32 entries eligible for its original design. All 32 source materials appear in the historical training registry and Phase 1 training manifests. The merged project registry finds zero unused materials in the source Test and Validation splits after including the 26-material replication. One source Train material remains unmatched by material ID in this ledger. The four reserved Test materials stay sealed. These counts describe the current inputs and audited registry.
- **Precision and heterogeneity.** Preserve the FP32 scores and calibrate a predeclared FP64 confirmation rule before new outcomes are scored. Use the same checkpoints, geometries and whole-pair comparison. Treat path, material and path–material effects as descriptive for the seven previously tested paths. Use the full panel for the primary contrast and report family-cluster uncertainty and influence.
- **Next gate.** Obtain independent candidate materials and an adequate reference panel, check material and structure-family overlap, and check compatibility with the frozen DFT reference and training budget. Predeclare final interval, practical margin, and treatment of multiplicity before accepting any of several candidate paths. Three paths can be used to pilot the protocol.

Inputs include the [Phase 1 feature table](candidate_path_features.csv), the [seven-path candidate manifest](../../results/phase1/candidate_comparison/lock.json), the [26-material panel lock](../../results/phase1/material_replication/lock.json) and the original project training and evaluation manifests used by the [panel freezer](../../src/freeze_net_path_panel.py).
