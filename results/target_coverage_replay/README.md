# Retrospective 17-pair selection replay

These frozen results cover 17 candidate pairs from 16 connected chemical-system groups. The main fit uses the historical 96-material panel. Choices remain fixed when evaluated on the 112- and 106-material historical panels. Lower differences from the matching pure D₀ model indicate better performance.

- Random selection, BVSE alone, BVSE with D₀ coverage, and BVSE with target-domain coverage are recorded alongside structural novelty, force disagreement, and the zero-cost pure D₀ reference.
- BVSE alone is the reference for feature additions. Target coverage uses the reduction in four-nearest-anion environment distances to the development target set. It changed two of 17 choices and did not show a reliable gain over BVSE. Its status is exploratory.
- All panels were used historically and provide a retrospective screen. Path matching uses atom and image counts as exposure proxies. The resulting comparisons do not establish equal realized DFT core-hours.
- Estimated expected random outcomes average the two available actions. The pure D₀ reference has no acquisition, and the hindsight best action is used only to calculate regret.
- cross_panel_summary.json reports models fitted on the main panel and applied unchanged to the other two panels. summary.json is the separate leave-chemistry-out analysis fitted and evaluated on the historical 106-material panel.
