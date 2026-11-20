RADIUSFPS + SPHERICAL BOUND INTEGRATION -- FIRST REAL IMPLEMENTATION TEST

This is a native CPU C++ experiment, not a gem5/RVV benchmark.

RUN (Ubuntu/WSL):
  cd "/mnt/c/Users/lsadi/Documents/PCD Research"
  bash Polar/Exp/run_radiusfps_spherical_test.sh

RUN (Git Bash, if g++/clang++ is installed):
  cd "/c/Users/lsadi/Documents/PCD Research"
  bash Polar/Exp/run_radiusfps_spherical_test.sh

It uses datasets/kitti/0000000000.bin, the first N=100000 points, M=128,
voxel size 2.0, first FPS point = original index 0.

EXPERIMENT
- Baseline: original qsort-based scalar RadiusFPS voxel/update logic, no m5ops.
- Variant: same voxel/update logic plus a certified spherical-coordinate
  distance LOWER bound for voxels that original XYZ bound did not skip.
- Each voxel's spherical intervals cover its actual member points (in double precision).
- FPS indices from baseline and variant must match at ALL 128 positions.
- Report end-to-end times PREP_MS+SAMPLE_MS for both, independently.
- Reports additional voxel skip counts, point visits, and full distances.
- Benchmark run DOES NOT perform additional expensive bound verification scans.
- Separate validation run checks the lower bound against each active voxel's
  actual nearest-point distance for all 127 iterations.

IMPORTANT SCIENTIFIC DISTINCTION
Prior offline 50.93% region pruning analysis was an UPPER bound for identifying
possible NEXT SAMPLE WINNER regions. This integration uses a LOWER bound for
SKIPPING DISTANCE UPDATES within RadiusFPS. The original 50.93% number is NOT
expected to translate into the extra voxel skips here. It is a distinct test.
The baseline is qsort scalar RadiusFPS, not prior optimized radix/RVV code.
Host timings must NOT be compared to gem5/RVV simulated timings.

If spherical bounds save too few updates to offset trigonometry/prep cost,
then this integration is slower, regardless of correctness.
