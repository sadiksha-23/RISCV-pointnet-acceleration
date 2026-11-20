INDEPENDENT FPS HEAD-TO-HEAD BENCHMARK (no hybrid algorithm)

Requirements: Ubuntu/WSL, g++ with C++17 support.
The existing radiusfps_spherical_test.cpp must be present in Polar/Exp
(from RadiusFPS_Spherical_Test_v2.zip, which you have already extracted).

This experiment uses ORIGINAL scalar RadiusFPS via runOnce(..., false, ...)
from your previous test source, and INDEPENDENT spherical-coordinate
region FPS (sine/cosine upper bounds, representative lower bound, and lazy
per-point updates). It does not execute RadiusFPS+spherical hybrid.

Run in Git Bash:
  cd "/c/Users/lsadi/Documents/PCD Research"
  wsl bash -lc 'cd "/mnt/c/Users/lsadi/Documents/PCD Research" && bash Polar/Exp/run_independent_fps.sh'

Output logs:
  Polar/Exp/independent_fps_results/centroid_comparison.txt
  Polar/Exp/independent_fps_results/sensor_comparison.txt

Both timing results include PREP + SAMPLING. 5 independent runs each.
Both algorithms are checked against a standalone plain float32 BasicFPS
reference for all 128 indices in every repeat. Prints FAIL and exits if any
mismatch, instead of reporting a false performance win.

The test is scalar native C++ for apples-to-apples initial timing, NOT gem5,
NOT RVV. Spherical approach is a prototype, not a fully optimized
implementation. Any faster result should be confirmed with more datasets,
compiler settings, and controlled profiling.
