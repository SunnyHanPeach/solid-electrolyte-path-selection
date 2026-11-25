# mp-5012 climbing-image DFT cost pilot

The existing ordinary five-image NEB band had a forward barrier of 1.867424 eV and a maximum residual of 0.098151 eV/Å. The climbing-image phase used image 3 and the same DFT settings. It began with fresh SCF from the saved path geometry and used 9 assigned MPI ranks across 3 image groups.

Two climbing-image updates completed. The last saved forward barrier is 1.870730 eV and the maximum path residual is 0.102824 eV/Å. The target is 0.100 eV/Å. The third update was interrupted after the time-budget sentinel was sent while the second image was in SCF. The image log records deletion of the requested EXIT file. QE emitted an SCF warning and returned code 1 on this interruption. This run contributes a checkpoint and a measured cost. The barrier remains provisional.

The stage used 4,432 seconds of wall time and 39,888 assigned MPI rank-seconds, equivalent to 11.08 assigned rank-hours. This excludes prior endpoint and plain-NEB work. The first attempted launch requested 12 ranks and was rejected by Open MPI before SCF. That attempt is recorded separately with zero wall seconds and zero assigned rank-seconds.
