# Project guide

A practical introduction to the **RISC-V PointNet++ Acceleration** repository. This guide is written for students and researchers who may be new to PointNet++, FPS, gem5, or the project's folder names.

**[Back to the main README](README.md)**

## 1. What problem are we trying to solve?

A LiDAR scan can contain tens of thousands of points. Deep-learning systems such as **PointNet++** need to choose representative points, find nearby neighbors, and process features. Those steps require a lot of computation.

This project asks two related questions:

1. **Processor question:** How do straightforward (scalar) and RISC-V Vector (RVV) implementations of these operations behave on a simulated RISC-V system?
2. **Algorithm question:** Can we reduce the amount of work required by **farthest point sampling (FPS)** while still choosing the correct points?

Most of the recent algorithm research focuses on the second question.

### Important terms

| Term | Plain-language meaning |
| --- | --- |
| **Point cloud** | A set of 3D coordinates, often captured by a LiDAR sensor |
| **N** | Number of input points, such as 100,000 |
| **M** | Number of points FPS should select, such as 128 |
| **FPS** | Farthest point sampling: select the point farthest from the *already selected set*, then repeat |
| **Exact FPS** | An implementation intended to return the same selected indices as the agreed reference, including a consistent starting point and tie rule |
| **Voxel** | A small 3D box used to group nearby points |
| **Pruning** | Safely skipping work when we can prove it cannot change the next selected point |
| **Scalar** | An implementation using ordinary operations rather than explicit vector instructions |
| **RVV** | RISC-V Vector instructions, which operate on groups of values |
| **gem5** | A simulator that models a processor and memory system for performance studies |
| **Preparation time** | Time spent organizing data structures before the main sampling loop |
| **Sampling time** | Time spent carrying out FPS selections after preparation |

FPS normally maintains each candidate point's distance to its **nearest** selected point. The next selected point is the one with the **largest** of these nearest distances. This is why finding reliable ways to skip distance updates matters.

## 2. Find your way around the folders

~~~text
RISCV-pointnet-acceleration/
├── README.md                 Overview and entry point
├── PROJECT_GUIDE.md          This detailed guide
├── benchmark/
│   ├── src/                  C++ algorithms, kernels and pipeline
│   ├── cpu/                  gem5 RISC-V system configurations
│   ├── stats/                Stored gem5 results and compiler reports
│   ├── performance_review/   Investigations into performance
│   └── external_research/    Third-party NS-FPS baseline
├── spherical_fps/
│   ├── selection_patterns/   Where FPS selects points
│   ├── geometric_bounds/     Safe geometric pruning studies
│   ├── initial_implementation/
│   ├── main_implementation/    Optimized scalar, profiling and RVV work (local)
│   └── experimental_variants/
├── datasets/                 KITTI LiDAR point clouds
├── documentation/            Research PDFs and figures
└── bin/                      Generated executables (ignored by Git)
~~~

The `bin/` directory may not exist in a fresh checkout. The native spherical benchmark scripts create it when compiling. **Do not commit compiled binaries.**

### `benchmark/src/` — implementations

- **`fps/basic implementation/`:** regular FPS, including scalar and RVV versions, with synthetic and KITTI cases.
- **`fps/RadiusFPS/`:** voxel-based FPS with spatial pruning; scalar, sorting, radix, and vector variants.
- **`fps/lazyFPS/`:** lazy-update FPS exploration.
- **`fps/NSFPS/`:** this project's drivers and profiling wrappers for an external NS-FPS comparison.
- **`microbenchmarks/scalar/` and `microbenchmarks/rvv/`:** smaller PointNet++ operations tested individually, such as FPS, ball query, interpolation, and upsampling.
- **`pipeline/scalar/` and `pipeline/rvv/`:** representative C++ PointNet++ processing pipelines, including profiled variants.

The two kinds of performance work are different: **microbenchmarks** isolate one operation, whereas **pipeline tests** measure several operations together.

### `benchmark/cpu/` — simulated RISC-V machine

The two Python files configure a gem5 system with a single out-of-order RISC-V core, 2 GHz clock, 32 KiB instruction/data L1 caches, 512 KiB L2 cache, and DDR4 memory.

- `riscv_benchmark.py` accepts a RISC-V executable and KITTI path.
- `riscv_fps_comparison.py` also accepts the input point count N and selected point count M.

These files describe the **simulated machine**. They do not implement the FPS algorithm themselves.

### `benchmark/stats/` — saved measurements

- `experiments/fps/`: FPS run results.
- `experiments/pipeline/`: full and profiled pipeline experiments.
- `supporting/compiler_reports/`: compiler-generated information.
- `supporting/microbenchmarks/`: supporting scalar, auto-vectorized, and RVV kernel measurements.

Expect to see files such as `stats.txt`, program output, test summaries, and compiler diagnostics. This is research history, not a folder of executables.

### `benchmark/performance_review/` — explanations of bottlenecks

These investigations examine **why** an algorithm does or does not run faster:

- `BasicFPS/`: distance calculations and wasted work in the basic algorithm.
- `RadiusFPS/`: preparation versus sampling, voxel size, vector profiling, sorting, and how much work spatial pruning avoids.
- `ns_fps/`: native and gem5 comparisons and profiling of NS-FPS.

**Keep the original log context.** Some folders contain exploratory runs or failed commands rather than valid benchmarks. For example, the saved `RadiusFPS/fps_sort_comparison/std_sort/.../run_output.txt` ends in a command-usage error, so it is **not a valid timing result**.

### `benchmark/external_research/` — independent baseline

`external_research/ns-fps/` contains the original third-party NS-FPS project. We use it for comparison. Do not confuse the upstream code with our own C++ FPS implementations. Refer to its own README for attribution and citation.

### `spherical_fps/` — experimental spherical FPS

The four research areas reflect how the investigation developed:

| Folder | Research question | Main files |
| --- | --- | --- |
| `selection_patterns/` | How does exact FPS spread selected points across shapes and real LiDAR data? | `shape_experiment.py`, `kitti_fps_structure.py` |
| `geometric_bounds/` | Can angular/radial regions be safely removed from consideration? | `spherical_kitti_tests.py`, `compare_spherical_bounds.py` |
| `initial_implementation/` | Can an independent spherical-region FPS algorithm work correctly and efficiently? | `compare_independent_fps.cpp` |
| `experimental_variants/radiusfps_spherical_hybrid/` | What happens when spherical bounds are added to the existing RadiusFPS? | `radiusfps_spherical_test.cpp` |

The `results/` folders next to each experiment contain saved graphs, tables, and text output:

- `selection_patterns/results/synthetic_1k_32/`: synthetic 3D/2D shapes, N = 1,000, M = 32.
- `selection_patterns/results/kitti_100k_128/`: exact FPS selection behavior on KITTI, N = 100,000, M = 128.
- `geometric_bounds/results/initial_8_checkpoints/`: early region study at eight checkpoints.
- `geometric_bounds/results/region_prediction_127_checkpoints/`: expanded region study.
- `geometric_bounds/results/cosine_bounds/`: XYZ-versus-spherical-versus-combined bound comparison.
- `initial_implementation/results/`: host benchmarks of independent SphericalFPS and RadiusFPS.
- `experimental_variants/radiusfps_spherical_hybrid/results/`: timing and bound-validation results for the hybrid.

**A result folder is not executable code.** Start with the nearby `.py`, `.cpp`, or `.sh` file to understand how it was generated.

### `datasets/` and `documentation/`

`datasets/kitti/` holds KITTI LiDAR frames named like `0000000000.bin`. The files are little-endian float32 records of four values per point: **x, y, z, intensity**. Most FPS code uses x/y/z and ignores intensity.

The repository currently includes a small set of these frames. The ignore rule for `datasets/kitti/` prevents **new untracked additions** from appearing in Git; it does not automatically remove files that are already tracked. Do not delete data just to make an ignore rule apply.

`documentation/` contains research PDFs and diagrams for further reading.

## 3. Setup for a first experiment

You can inspect the results and source code on any computer. Running experiments requires different tools depending on what you want to test.

| Activity | What you need |
| --- | --- |
| View saved plots, tables, or text logs | A file viewer or editor |
| Run Python geometry experiments | Python 3, NumPy, Matplotlib |
| Run native SphericalFPS C++ comparisons | A C++17 compiler such as `g++`, plus a shell environment such as WSL/Ubuntu |
| Run simulated RISC-V benchmarks | A RISC-V-enabled gem5 build, a suitable RISC-V cross-compiler, gem5 m5ops support, and a compiled RISC-V program |

**Windows note:** Git Bash is useful for Git commands, but the C++ benchmark scripts are easiest to run in **WSL/Ubuntu**. The old RISC-V build commands were not collected into a single universal build script. You may need to adjust paths and toolchain flags for your own setup.

From the repository root in a Python environment, install the geometry dependencies:

~~~bash
python3 -m pip install numpy matplotlib
~~~

Make sure this file is available:

~~~text
datasets/kitti/0000000000.bin
~~~

### Option A: Understand where FPS chooses points

~~~bash
bash spherical_fps/selection_patterns/run_kitti_experiment1.sh
~~~

This executes the KITTI Python experiment and writes plots and CSV files to `spherical_fps/selection_patterns/results/kitti_100k_128/`. It is an analysis of **exact 3D FPS behavior**, not a speed test of an optimized C++ implementation.

### Option B: Investigate spherical pruning bounds

~~~bash
python3 spherical_fps/geometric_bounds/compare_spherical_bounds.py
~~~

This evaluates XYZ bounding-sphere bounds, spherical cosine-law bounds, and their combination on KITTI. Its output goes to `spherical_fps/geometric_bounds/results/cosine_bounds/`.

The other study is:

~~~bash
python3 spherical_fps/geometric_bounds/spherical_kitti_tests.py
~~~

It evaluates spherical region predictions and pruning at its configured checkpoints. Use its `--help` option to see choices for origins, checkpoints, N, and M.

These scripts produce **geometric evidence**, not reliable execution-time speedups.

### Option C: Compare native C++ FPS implementations

Use WSL/Ubuntu or another supported environment with `g++`:

~~~bash
bash spherical_fps/initial_implementation/run_independent_fps.sh
~~~

The script:

1. Checks for the KITTI input and compiler.
2. Compiles `compare_independent_fps.cpp` into root `bin/`.
3. Runs RadiusFPS and SphericalFPS with centroid and sensor origins.
4. Saves results under `spherical_fps/initial_implementation/results/`.

For the hybrid experiment:

~~~bash
bash spherical_fps/experimental_variants/radiusfps_spherical_hybrid/run_radiusfps_spherical_test.sh
~~~

Both are **native host benchmarks**, not RISC-V/gem5 simulations. They can overwrite the saved text result files when rerun; copy results elsewhere first if you need to preserve the original measurements.

## 4. Running RISC-V benchmarks with gem5

This part is more advanced. The RISC-V simulations **cannot run simply by executing a C++ file**, because the programs are compiled for a simulated target processor.

The general process is:

1. Choose a C++ implementation from `benchmark/src/`.
2. Compile it for the **RISC-V target** (RVV instructions require a compatible compiler and target).
3. Include and link the **gem5 m5ops** support used by the source.
4. Place the resulting executable under your local root `bin/` directory.
5. Run it with the appropriate gem5 CPU configuration.
6. Read the program output and gem5 statistics, then check correctness.

A **command template**, not a guaranteed one-step reproduction command:

~~~bash
./gem5/build/RISCV/gem5.opt \
  --outdir=m5out/fps_test \
  benchmark/cpu/riscv_benchmark.py \
  bin/YOUR_RISCV_PROGRAM.riscv \
  datasets/kitti/0000000000.bin
~~~

For a binary that accepts N and M as additional arguments, use:

~~~bash
./gem5/build/RISCV/gem5.opt \
  --outdir=m5out/fps_comparison \
  benchmark/cpu/riscv_fps_comparison.py \
  bin/YOUR_RISCV_PROGRAM.riscv \
  datasets/kitti/0000000000.bin \
  100000 128
~~~

The executable name is a **placeholder**. These commands require a separately installed/compiled gem5 build and compatible RISC-V binary. The exact binary arguments must match its `main()` function.

**Known portability issue:** Several older C++ files and saved run commands contain hard-coded paths from an earlier development environment (`PCD Research`). For example, the basic KITTI scalar FPS file has a fixed dataset path despite showing a command-line argument. Check and fix those paths before trying to reproduce that program. Do not assume all files in `benchmark/src/` will compile and run unchanged.

### What the gem5 files mean

The C++ programs use `m5_reset_stats` and `m5_dump_stats` to tell gem5 where a measurement begins and ends. The simulator writes results such as `stats.txt`, usually under the chosen `m5out/` directory.

The `benchmark/stats/` and `benchmark/performance_review/` folders contain **previously saved** outputs. Generated `m5out/` directories and the root `bin/` directory are ignored by Git.

## 5. Reading results correctly

The most important principle is **compare equivalent work, including the time needed to prepare the data**.

### Preparation versus sampling

A spatial FPS algorithm might:

1. Load a point cloud.
2. Build voxels or spherical regions.
3. Sort or reorganize points.
4. Run the FPS selection loop.

Some logs measure only step 4; others include steps 2–4. A sampling-only time is **not** the full cost of the algorithm. When reporting a result, identify exactly what was measured.

### Native versus gem5 timing

- **Native host** means the C++ program ran on the computer that launched it.
- **gem5 time** comes from a modeled RISC-V processor and memory system.

Even for the same algorithm, these numbers are **not interchangeable**. Never present a native result as a simulated RISC-V speedup.

### Example from the saved independent SphericalFPS study

KITTI frame `0000000000.bin`, N = 100,000, M = 128, centroid origin, five native repeats:

| Algorithm | Median preparation | Median sampling | Median total |
| --- | ---: | ---: | ---: |
| RadiusFPS | 22.888 ms | 4.671 ms | **27.654 ms** |
| Initial independent SphericalFPS | 10.387 ms | 66.461 ms | **76.849 ms** |

Source: `spherical_fps/initial_implementation/results/centroid_comparison.txt`.

The spherical approach prepared its regions faster, but its sampling loop was much slower. Both implementations matched the reference 128 selected indices in those tests. This is a **negative performance result worth keeping** because it identifies the problem to solve next.

### A pruning percentage is not a speedup

The geometric study reported **50.93% average safely pruned points** with the tested combined bounds. That means the bounds proved many points could be excluded from certain searches. It does **not** mean the algorithm ran 50.93% faster. Computing bounds, managing regions, and updating points still require time.

### Check correctness before quoting a speedup

The selected point indices should match the agreed reference for the same input, first point, and tie-breaking convention.

**Known issue to investigate:** One saved RadiusFPS RVV profiling run with N = 100,000 and M = 32 reports `FPS_INDEX[1]=30625`, whereas the basic FPS reference has `FPS_INDEX[1]=3052`. That run should **not** be described as a validated exact-FPS result until the cause is resolved. See `benchmark/performance_review/RadiusFPS/fps_rvv_profile/N100000_M32_V2/output.txt`.

Similarly, the saved standard-sort comparison contains a gem5 invocation error and is not usable as valid timing evidence.

### Minimum information to record for each new result

A useful benchmark record says:

- **Which algorithm and source revision** were tested.
- **Which dataset and parameters** were used (frame, N, M, voxel size, and region settings).
- **Where it ran** (native machine or gem5, CPU configuration, RVV settings, toolchain).
- **What the time includes** (loading, preparation, sampling, whole pipeline).
- **Whether the selected indices were correct** and which reference they were checked against.
- **How many repeats were run**, with the median or spread when relevant.

This makes future comparisons much easier to trust.

## 6. Research direction and open work

The spherical research developed in four steps:

1. **Selection patterns:** observe the order in which exact FPS chooses points on synthetic shapes and KITTI.
2. **Geometric bounds:** find safe ways to eliminate spherical regions that cannot change the next FPS selection.
3. **Initial independent implementation:** put the geometry into a complete C++ algorithm; verify the selected indices; measure its cost.
4. **Alternative hybrid:** add spherical bounds to RadiusFPS and check whether that helps.

The more efficient scalar candidate **has now been developed and measured** in local experiments. Direct-sector binning (01), squared elevation binning (05), squared radial binning (06), and AABB region bounds (A) form the leading **F2** variant. The separate early radial-short-circuit addition (C) did not improve the measured F2 total in the combined test. The current phase is final scalar/RVV correctness validation, scaling sweeps and hardware-oriented analysis, rather than another broad round of algorithmic variants.

For the RISC-V branch of the project, another priority is to make build/run commands more portable and verify scalar-versus-RVV correctness for all reported configurations.

## 7. Working with the repository

- Keep **source code** in its algorithm folder rather than mixing it into result folders.
- Keep important graphs, logs, and experiment summaries with the relevant study.
- Put **compiled executables** in root `bin/`; do not commit them.
- Keep gem5-generated `m5out/` output local unless a specific experiment is deliberately archived into `benchmark/stats/`.
- Do not confuse **third-party NS-FPS** with our own implementations.
- Before running a script that writes into `results/`, check whether you want to preserve the existing measurements.
- Before deleting or reorganizing anything, check `git status` and any code paths that refer to the old name.

For new readers: **start with `selection_patterns/`, then `geometric_bounds/`, then the initial implementation**. That order explains not only what we built but why we built it.

## 8. Current SphericalFPS optimization and RVV study (November 2026)

**Repository-sync note:** The measurements below were reported from local gem5 runs. Updated `main_implementation` sources, temporary sweep helpers, and new result folders may still be uncommitted; do not mistake a documentation update for a push of those files.

### Algorithm decisions and completed timings

The optimization sequence first replaced per-point angular calculations with direct sector assignment (01). Later experiments independently tested squared elevation (05), squared radial binning (06), and a cheaper region-radius construction (07). In a comparable set of gem5 preparation-plus-sampling runs at N=100,000, M=128, 05 measured **21.4085 ms**, 06 **21.8126 ms**, and 07 **24.0560 ms**, compared with the experiment's 01 reference of **24.1963 ms**. These are individual variant tests, not an additive claim.

A separate native algorithm study tested AABB bounds (A), adaptive region splitting (B), and a radial short circuit (C). A reduced measured point-distance evaluations from **1,247,077** to **1,023,935**, with identical selected indices in five tested rounds. B's preparation overhead outweighed the sampling reduction. C did not establish a robust sampling-time improvement.

Three **combined** gem5 experiments, all using KITTI frame `datasets/kitti/0000000000.bin`, N=100,000, M=128, had these first explicit `simTicks` intervals:

| Variant | Algorithm components | gem5 preparation + sampling |
| --- | --- | ---: |
| F1 | 01 + 05 + 06 | 20.442032 ms |
| **F2** | **F1 + AABB bounds** | **18.493858 ms** |
| F3 | F2 + early radial short circuit | 18.610774 ms |

The F2 reset occurs before centroid computation and region preparation; its explicit dump occurs after sampling and before output. Thus the measured interval includes both **preparation and sampling**, not loading or printing. F2 was selected as the scalar candidate rather than continuing to optimize every small variation.

The first subsequent **final-code** local measurements at N=100,000, M=128 were **18.5302 ms scalar** and **13.9127 ms RVV**, corresponding to **1.332×** speedup relative to that scalar run (24.92% less runtime). This is a preliminary, one-run comparison. Exact FPS indices, RVV instruction usage, compiler flags and matched RadiusFPS baselines must be verified before publishing an RVV or cross-algorithm conclusion.

### Intended final source files and result layout

The local project uses these three implementations under `spherical_fps/main_implementation/`:

- `spherical_fps.cpp` — optimized scalar FPS; measure the complete algorithm without internal profiling overhead.
- `spherical_fps_profile.cpp` — same algorithm, instrumented for major phases and bottleneck diagnosis; do not compare instrumented total time against uninstrumented timing.
- `spherical_fps_rvv.cpp` — vectorized candidate; validate its selected point indices against scalar and BasicFPS.

The planned local result directories are `results/spherical_fps/`, `results/spherical_fps_profile/`, and `results/spherical_fps_rvv/`. Within each, a baseline such as `N100000_M128/` is kept separate from `N_sweep/` and `M_sweep/`. RadiusFPS RVV sweep results are being saved separately under `benchmark/stats/experiments/fps/RadiusFPS/kitti/RVV_sweeps/`.

### Controlled sweep parameters

- **N sweep:** N = 1,024; 2,048; 4,096; 10,000; 20,000; 50,000; 100,000, with M = 128.
- **M sweep:** M = 32; 64; 128; 256; 512; 1,024, with N = 100,000.
- **Dataset:** KITTI frame 0 (`datasets/kitti/0000000000.bin`), using the first N points of the frame.
- **gem5 configuration:** `benchmark/cpu/riscv_fps_comparison.py`: one 2 GHz O3 RISC-V core, 32 KiB L1 instruction and data caches, shared 512 KiB L2 and DDR4-2400. Verify RVV VLEN and executed vector instructions from individual runs.
- **Measurement definition:** program-provided m5 reset and dump around **preparation + sampling**, excluding file loading and final index printing. With two `simTicks` sections, confirm that the first is the explicit interval, and the later section is the termination dump; do not sum them blindly.

**RadiusFPS caution:** The existing `radius_fps_vector.cpp` hardcodes N=100,000, M=128 and uses multiple profiling checkpoints. A separate locally generated sweep-oriented copy accepts runtime N and M and uses a single total-algorithm interval. This transformation is not yet a validated correctness-preserving RVV benchmark; its FPS indices must be checked. Earlier documentation also records a RadiusFPS RVV index mismatch at M=32, which remains unresolved. Never describe that result as confirmed exact FPS.

**Avoid redundant simulations.** Reuse a completed run only when its executable, build flags, input frame, simulator configuration, measurement boundary and output validity match the desired comparison. Do not rerun unchanged baselines just to reproduce the same number, but do not silently combine results from different RadiusFPS variants.

---

**[Return to README](README.md)** · **[SphericalFPS experiments](spherical_fps/)** · **[Benchmark source](benchmark/src/)** · **[Saved measurements](benchmark/stats/)**
