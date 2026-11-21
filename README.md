# RISC-V PointNet++ Acceleration

This repository explores how to make **3D point-cloud processing faster on RISC-V processors**. It contains C++ implementations, gem5 simulation configurations, saved benchmark results, and ongoing research on faster **farthest point sampling (FPS)**.

The main focus is on operations used by **PointNet++**, especially FPS. We compare straightforward implementations with vectorized and geometry-based approaches, then measure both **speed and correctness**.

> **Project status:** Active research. This is an experimental codebase, not a ready-to-install PointNet++ library. Some older experiments have fixed paths or need manual setup before they can be reproduced.

## Start here

If you are new to the project, read **[PROJECT_GUIDE.md](PROJECT_GUIDE.md)**. It explains the terms, folder structure, experiments, how to run the currently supported scripts, and how to interpret results.

In simple terms:

- **Point cloud:** a collection of 3D points, often captured by LiDAR.
- **FPS:** repeatedly selects the point farthest from the points already chosen, helping represent a large cloud with fewer points.
- **RISC-V:** a processor instruction-set architecture.
- **RVV:** RISC-V Vector instructions, which process multiple data values together.
- **gem5:** a computer-system simulator used here to measure RISC-V implementations without relying on the exact target hardware.

## What the repository investigates

1. **PointNet++ workloads:** scalar and RVV versions of individual operations and a representative C++ pipeline.
2. **FPS algorithms:** BasicFPS, LazyFPS, voxel-based RadiusFPS, and an external NS-FPS comparison.
3. **Performance:** where execution time goes, how much work gets skipped, and whether optimizations actually make total runtime shorter.
4. **SphericalFPS:** an experimental FPS approach using spherical regions and safe geometric bounds. The goal is to skip unnecessary work **without changing which points FPS selects**.

## Repository layout

| Path | What you will find |
| --- | --- |
| [`benchmark/src/`](benchmark/src/) | C++ source code for FPS variants, PointNet++ operations, and pipeline experiments |
| [`benchmark/cpu/`](benchmark/cpu/) | gem5 RISC-V CPU and memory configurations |
| [`benchmark/stats/`](benchmark/stats/) | Saved gem5 measurements, experiment logs, and supporting compiler reports |
| [`benchmark/performance_review/`](benchmark/performance_review/) | Investigations into FPS bottlenecks and algorithm comparisons |
| [`benchmark/external_research/`](benchmark/external_research/) | Original third-party NS-FPS code kept separate from this project's implementations |
| [`spherical_fps/`](spherical_fps/) | Spherical-coordinate research, geometric bounds, initial algorithm, and alternative experiments |
| [`datasets/`](datasets/) | KITTI LiDAR frames used in the experiments |
| [`documentation/`](documentation/) | Research PDFs, diagrams, and figures |
| `bin/` | Local compiled executables; generated when needed and ignored by Git |

**Source code and saved results are intentionally separate.** For example, `benchmark/src/` contains implementations; `benchmark/stats/` contains past gem5 runs; `benchmark/performance_review/` explains why some runs were faster or slower.

## Try the spherical experiments

These are the clearest starting points because their current launchers compile or run from the repository root.

**Requirements:** Python 3 with NumPy and Matplotlib, a KITTI frame at `datasets/kitti/0000000000.bin`, and (for C++ benchmarks) a C++17 compiler such as `g++`. WSL/Ubuntu is recommended for the shell-based C++ scripts.

From the repository root:

~~~bash
# Install Python packages in your chosen environment
python3 -m pip install numpy matplotlib

# Study exact FPS selection patterns on KITTI
bash spherical_fps/selection_patterns/run_kitti_experiment1.sh

# Compare rigorous XYZ and spherical geometric bounds
python3 spherical_fps/geometric_bounds/compare_spherical_bounds.py

# Native C++ comparison: RadiusFPS versus independent SphericalFPS
bash spherical_fps/initial_implementation/run_independent_fps.sh
~~~

The native C++ script builds its executable into the root `bin/` folder and writes text results under `spherical_fps/initial_implementation/results/`. It does **not** run gem5.

For the simulated RISC-V experiments, see the [project guide](PROJECT_GUIDE.md#running-risc-v-benchmarks-with-gem5). A gem5 installation and suitable RISC-V binaries are required; compiled binaries are not provided.

## Selected findings

The findings below come from saved experiments on KITTI with **N = 100,000 input points** and **M = 128 selected points**. They are **not all from the same execution environment**, so they should not be combined into a single ranking.

- **Spherical geometric bounds:** the combined XYZ and spherical-cosine method certified **50.93% average point pruning** across 127 selection stages for the tested centroid-origin shell configuration. This is a *geometric pruning rate*, **not** a 50.93% reduction in runtime.
- **Independent SphericalFPS, native host (centroid origin, five repeats):** RadiusFPS had a median total time of **27.654 ms**, while initial SphericalFPS took **76.849 ms**. Both produced the same 128 indices in the recorded tests. This early spherical implementation was **slower**; later optimized implementations substantially improved on it. See the current direction below.
- **RadiusFPS + spherical hybrid:** adding spherical bounds to RadiusFPS also increased runtime in the recorded native host tests despite doing some extra pruning.

Saved RISC-V/gem5 experiments are available under `benchmark/stats/` and `benchmark/performance_review/`. Some results are incomplete or have correctness issues; **check the relevant logs before citing a performance claim**.

### Recent SphericalFPS optimization measurements (local gem5 runs)

On KITTI frame `0000000000.bin`, N=100,000, M=128, the following *separate experimental variants* measured preparation + sampling with a single explicit gem5 statistics interval:

| Variant | Optimizations | Measured interval |
| --- | --- | ---: |
| F1 | Direct sectors + squared elevation + squared radial binning | 20.4420 ms |
| **F2** | **F1 + AABB bounds** | **18.4939 ms** |
| F3 | F2 + radial early-rejection short circuit | 18.6108 ms |

F2 is the preferred **scalar candidate**. In the first follow-up scalar-versus-RVV baseline, scalar measured 18.5302 ms and RVV 13.9127 ms. These figures are *preliminary research measurements*: correctness across multiple workloads and fairness against the exact RadiusFPS version are still under verification. The local final files and new results are not assumed to be present in the GitHub checkout until committed.

## How to interpret timings

A fair comparison should use the **same dataset, N and M, simulation or host environment, and timing boundaries**.

In particular:

- **Host times** measured on a real computer are not directly comparable with **gem5 simulated times**.
- **Preparation** (for example building or sorting voxels) must be included when comparing full algorithm runtimes.
- Skipping many points or regions does **not** necessarily save time if checking the bounds is expensive.
- Matching the output indices is important: a faster but incorrect FPS result is not an equivalent implementation.

See [PROJECT_GUIDE.md](PROJECT_GUIDE.md#reading-results-correctly) for examples and known limitations.

## External research and documentation

The NS-FPS source under [`benchmark/external_research/ns-fps/`](benchmark/external_research/ns-fps/) is an **external reference implementation**, not original code from this project. Its own [README](benchmark/external_research/ns-fps/README.md) contains the upstream project details and citation information.

The `documentation/` directory contains papers, notes, and figures relating to FPS, RVV, and the broader research.

## Current direction

The original independent SphericalFPS was a starting point, not the final design. A newer scalar optimization candidate, **F2 (direct sectors + squared elevation + squared radial binning + AABB pruning)**, achieved an **18.4939 ms measured preparation-plus-sampling interval** on gem5 for KITTI frame 0, N=100,000 and M=128. The follow-on scalar implementation measured **18.5302 ms**, while a preliminary RVV version measured **13.9127 ms** (1.332× scalar/RVV speedup). These last two timings are from local runs described in the research log; the corresponding final source and new sweep outputs may not yet be pushed to GitHub. These are single-run measurements, not multi-run medians.

**Current priority:** finish scalar/RVV N and M sweeps and verify selected indices, executed RVV instructions, and a matched RadiusFPS comparison before making a final algorithm-level performance claim. No further broad scalar-optimization search is planned unless profiling reveals a concrete issue.

**For folder-by-folder explanations, setup, commands, and a guide to results, continue to [PROJECT_GUIDE.md](PROJECT_GUIDE.md).**
