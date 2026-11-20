# Spherical FPS research

Research is organized by purpose, with source code next to its saved results.

| Folder | Purpose | Saved results |
| --- | --- | --- |
| `selection_patterns/` | Exact FPS point-selection geometry on synthetic shapes and KITTI | `results/synthetic_1k_32/`, `results/kitti_100k_128/` |
| `geometric_bounds/` | Spherical region prediction, cosine-law bounds, certified pruning | `results/region_prediction_127_checkpoints/`, `results/cosine_bounds/`, `results/initial_8_checkpoints/` |
| `initial_implementation/` | Independent SphericalFPS vs scalar RadiusFPS | `results/centroid_comparison.txt`, `results/sensor_comparison.txt` |
| `experimental_variants/radiusfps_spherical_hybrid/` | RadiusFPS with added spherical pruning | `results/` |

**KITTI N=100,000, M=128 findings:** centroid-origin spherical+XYZ bounds certified 50.93% average point pruning over 127 checkpoints (not runtime speedup). Independent SphericalFPS median host time was 76.849 ms versus RadiusFPS 27.654 ms (centroid); sensor 77.613 ms versus 27.487 ms. Exact 128/128 indices matched. The hybrid was also slower than RadiusFPS. Host results are not gem5 measurements.

Run from the repository root (KITTI file needed at `datasets/kitti/0000000000.bin`):

```bash
python3 spherical_fps/geometric_bounds/spherical_kitti_tests.py --checkpoints $(seq 1 127)
python3 spherical_fps/geometric_bounds/compare_spherical_bounds.py
bash spherical_fps/initial_implementation/run_independent_fps.sh
bash spherical_fps/experimental_variants/radiusfps_spherical_hybrid/run_radiusfps_spherical_test.sh
```

C++ scripts require `g++` and compile into the ignored root `bin/`. Python plotting scripts require NumPy and Matplotlib. Historical notes may contain old paths. The `initial_8_checkpoints` results are preserved for reference.
