SphericalFPS preparation experiments 05, 06, 07
===============================================

PLACE CONTENTS OF THIS ZIP INTO:
  <repo>/spherical_fps/experimental_variants/optimization_tests/

This pack DOES NOT overwrite existing 01 code or the main implementation.
The generator creates separate source files based on existing 01:
  spherical_fps_05_elevation_squared.cpp
  spherical_fps_06_radial_squared.cpp
  spherical_fps_07_region_radius_squared.cpp

Variants:
  05  Sign-aware squared elevation boundary tests, no per-point horizontal sqrt.
  06  Squared radius bin comparison, actual interval extrema recovered per occupied region.
  07  Max squared enclosing radius then one sqrt per occupied region.

IMPORTANT: 05/06 may shift float-boundary bin assignments; if so, the native test FAILS.
That is a research result, not permission to skip correctness. Do not use failing variants.

FROM THE REPOSITORY ROOT (WSL/Ubuntu):
  cd /mnt/c/Users/lsadi/Documents/GitHub/RISCV-pointnet-acceleration
  python3 spherical_fps/experimental_variants/optimization_tests/run_prep_tests.py --mode native

If native correctness passes, run gem5:
  python3 spherical_fps/experimental_variants/optimization_tests/run_prep_tests.py --mode gem5

Note --mode gem5 also runs native correctness FIRST before attempting simulation.
To run both:
  python3 spherical_fps/experimental_variants/optimization_tests/run_prep_tests.py --mode both

DEFAULTS: KITTI datasets/kitti/0000000000.bin, N=100000, M=128, grid=24x8x6.
Optional: --n 10000 --m 128 --repeats 5 --data /full/path/to/frame.bin

Generated binaries go to ROOT/bin/ and results to:
  spherical_fps/experimental_variants/optimization_tests/results/prep_sqrt/

Gem5 installation assumptions relative to repository root:
  gem5/build/RISCV/gem5.opt
  gem5/include/
  gem5/util/m5/build/riscv/out/libm5.a
  benchmark/cpu/riscv_fps_comparison.py

Dependencies: python3, g++, riscv64-linux-gnu-g++, working gem5 installation.

Measurements: native table has prep/sample/total. The generated gem5
runner uses a SINGLE ROI around preparation + sampling, so gem5 comparison
reports accurate total ROI only. Native stage times must not be mislabeled gem5.
For detailed gem5 prep/sample split use separate reset/dump intervals later.

Methodological cautions:
- original 01 stays unchanged.
- Native correctness compares every selected index to 01, and by default
  to independent BasicFPS, and checks point evaluation counts.
- All gem5 results are validated against native 01 indices.
- Match compiler flags (-O3 -ffp-contract=off) and matched configurations.
- Arithmetic transformations may affect edge-case float behavior. Test more
  datasets and grid settings before claiming exactness generally.
