# A-endpoint fixed-geometry SCF launch

The primary pilot started on the local CPU at **2026-11-28 10:24:19 (UTC+08:00)**. The two remaining branches are in the same local queue. The backup pilot and later images have yet to run.

- Quantum ESPRESSO `6.7MaX`, local Ubuntu build `6.7-2build1`.
- Eight MPI ranks, two k-point pools, one thread per rank and one active branch at a time. The initial six-rank, three-pool probe estimated 7.53 GB of dynamic memory. The selected production layout reports 6.10 GB. The Linux environment has approximately 7.6 GiB of RAM.
- The same A geometry, hash-checked pseudopotentials, cutoff pair, Γ-centered grid, and 0.10 eV Gaussian smearing are fixed for all three branches. Charge and spin initialization vary according to the qualification plan. Each branch writes into its own scratch directory.
- The input-only probe was stopped after it had confirmed electron count, grid, and memory. The production process has started from scratch. Scientific acceptance still requires a normal exit, converged electronic state and recoverable energy and force outputs for each branch. No barrier was evaluated.

The inputs, source geometry, checksums, solver logs, XML and timing receipts stay local. The current cloud QE-a queue was left untouched.
