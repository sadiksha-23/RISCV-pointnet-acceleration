# RISC-V PointNet++ Acceleration

Research on accelerating **3D point-cloud operations**, especially **exact farthest point sampling (FPS)**, using algorithm design and RISC-V Vector (RVV) instructions.

FPS repeatedly chooses the point farthest from its nearest already-selected point. Our goal is to reduce unnecessary distance calculations **without changing the selected indices**.

## Where to start

- [Project guide](PROJECT_GUIDE.md) — folders, setup, and how to read the benchmarks.
- [Research flow](RESEARCH_FLOW.md) — short chronological research log and explanation of the latest algorithm.
- [Complete algorithm benchmarks](benchmark/stats/experiments/fps/comprehensive_benchmarks/complete_six_way.csv) — six implementations, 26 configurations each.
- [Benchmark caveats](benchmark/stats/experiments/fps/comprehensive_benchmarks/README.md) — provenance, correctness, and partial LMUL tests.

## Main code

- [Current SphericalFPS scalar](spherical_fps/main_implementation/spherical_fps.cpp)
- [Current SphericalFPS RVV, LMUL 1](spherical_fps/main_implementation/spherical_fps_rvv.cpp)
- [Earlier F2 reference](spherical_fps/ref_implementation/)
- [Other FPS algorithms](benchmark/src/fps/) — BasicFPS, LazyFPS, RadiusFPS, and comparison drivers.

The current algorithm uses **spherical spatial regions, conservative geometric pruning, AABB-first checks, packed point arrays, and selective 128-point microgroups**. The RVV version vectorizes point-distance updates.

## Current result

On KITTI frame `0000000000.bin`, **N = 100,000, M = 4,096**, gem5 reported:

| Implementation | Simulated time |
| --- | ---: |
| Main SphericalFPS RVV (LMUL 1) | **31.615 ms** |
| Main SphericalFPS scalar | 55.742 ms |
| Reference SphericalFPS RVV | 66.544 ms |
| RadiusFPS RVV | 51.240 ms |

Times include algorithm preparation and sampling, not input loading. Results are from the saved gem5 campaign, not wall-clock simulation durations or multi-run medians. Performance varies with N and M; **there is no claim that Main SphericalFPS wins every configuration**.

**Status:** The six principal benchmark sweeps are complete (156 rows, including 22 reused historical scalar measurements). LMUL 2/4/8 investigations are incomplete because of correctness mismatches; they are excluded from the primary comparison.

This is an experimental research repository, not a production PointNet++ package.
