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
- **Independent SphericalFPS, native host (centroid origin, five repeats):** RadiusFPS had a median total time of **27.654 ms**, while initial SphericalFPS took **76.849 ms**. Both produced the same 128 indices in the recorded tests. The initial spherical implementation is **slower**, and reducing its sampling overhead remains a research goal.
- **RadiusFPS + spherical hybrid:** adding spherical bounds to RadiusFPS also increased runtime in the recorded native host tests despite doing some extra pruning.

Saved RISC-V/gem5 experiments are available under `benchmark/stats/` and `benchmark/performance_review/`. Some results are incomplete or have correctness issues; **check the relevant logs before citing a performance claim**.

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

The next major goal is to improve **SphericalFPS**: preserve exact FPS selections while reducing repeated point updates and the cost of managing spherical regions. Any new speedup should be confirmed with **end-to-end measurements and correctness checks**, followed by RISC-V/gem5 testing where appropriate.

**For folder-by-folder explanations, setup, commands, and a guide to results, continue to [PROJECT_GUIDE.md](PROJECT_GUIDE.md).**
