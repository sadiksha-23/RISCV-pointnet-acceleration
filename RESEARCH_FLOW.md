# Research flow — FPS on RISC-V

*Short working memory for the project. Main results are single-run gem5 measurements unless marked native; this is not a publication-ready performance analysis.*

## How we got here

- **Started:** Investigated PointNet++ operations on RISC-V, focusing on FPS as an expensive point-cloud step.
- **BasicFPS:** Measured scalar/RVV brute-force distance updates; too much repeated work across N points and M selections.
- **LazyFPS:** Explored reducing unnecessary updates; kept it as a comparison, not our final design.
- **RadiusFPS:** Tested voxel grouping, sorting/radix sorting and geometric bounds; pruning helped but setup and per-region overhead mattered.
- **NS-FPS:** Examined an external implementation as another comparison; kept its source separate.
- **FPS selection patterns:** Studied where exact FPS picks points on synthetic shapes and KITTI; motivated angular/radial grouping.
- **Spherical bounds:** Tested XYZ, radial and spherical-angle bounds; one combined study certified **50.93% point pruning** at N=100k/M=128, **not** a 50.93% runtime gain.
- **First independent SphericalFPS:** Returned matching indices in the recorded native test but took **76.849 ms vs 27.654 ms RadiusFPS** (N=100k/M=128); pruning alone was too expensive.
- **RadiusFPS+spherical hybrid:** Added spherical checks to RadiusFPS; extra bound work outweighed savings in the native experiments.
- **Optimized F1/F2/F3:** Replaced costly angular functions with direct azimuth/elevation classification and squared radial bins. **F2 added AABB** and won that gem5 variant comparison: **18.4939 ms** at N=100k/M=128; F1 20.4420 ms, F3 18.6108 ms.
- **F2 reference:** Preserved in `spherical_fps/ref_implementation/` as the older optimized algorithm; its performance degraded as M grew.
- **Current Main SphericalFPS:** Added cheaper region checks, packed arrays and selective microgroups; kept both scalar and RVV LMUL 1 implementations.
- **Cone-bound experiment:** Rejected it because its preparation overhead outweighed the sampling gain (e.g. ~20.996 ms preparation with cone vs ~10.417 ms without, N=100k/M=1024 in the tested variants).
- **Complete comparison:** Collected **26 (N, M) cases × 6 implementations = 156** rows; 22 historical scalar rows were reused, not newly rerun.
- **LMUL study:** LMUL 2/4/8 is **unfinished**: Radius RVV M2 diverged at step 1 (N=1024/M=128); Main RVV M2 diverged at step 350 (N=100k/M=512). No validated M2/4/8 conclusion yet.

## How the current Main SphericalFPS works

1. **Input:** Read the first N KITTI XYZ points; FPS starts at original index **0**.
2. **Partition:** Calculate the point-cloud centroid and group points into **24 azimuth × 8 elevation × 6 radial bins** (up to 1,152 spherical regions).
3. **Prepare each region:** Store its point indices, bounding box (AABB), center/radius and radial limits; initialize cached farthest candidates.
4. **Pack points:** Store coordinates in contiguous **structure-of-arrays (SoA)** form, so distance updates are cheaper and RVV can process several points together.
5. **Split large regions:** Only regions with **at least 512 points** get **128-point microgroups**, each with its own safe bounding information.
6. **Select next sample:** For each newly selected point, try the cheap **AABB-first** skip test; if inconclusive, combine AABB, bounding-sphere and radial lower bounds.
7. **Prune safely:** Skip a region or microgroup only if the bound proves its distances cannot improve the stored nearest distances; otherwise update each point's nearest squared distance.
8. **Choose farthest:** Keep region/group maxima, choose the globally largest nearest distance, and resolve equal values by **smallest original index**. Repeat until M points are selected.

**Key idea:** Spend a little work checking spatial bounds to avoid a lot of repeated point-distance calculations, while preserving the exact FPS choice.

## Optimizations kept, and why

- **Direct azimuth quadrant + elevation sign classification:** Avoid repeated expensive angles/trigonometric functions when assigning regions.
- **Squared radial comparisons:** Avoid per-point square roots during radial bin assignment.
- **AABB-first early rejection:** Skip many regions before calculating all three geometric bounds.
- **Conservative bounds and margins:** Favor correctness rather than risky pruning near floating-point boundaries.
- **Packed SoA coordinates:** Improve contiguous access for scalar loops and RVV vector loads.
- **Selective 128/512 microgroups:** Add finer pruning **only** for sufficiently large regions, limiting preparation overhead.
- **Cached maxima and original indices:** Avoid rescanning skipped groups and preserve deterministic selection.
- **RVV LMUL 1:** Vectorize point-distance updates and maximum searches while keeping the same selection rules.
- **Separate preparation and sampling measurements:** Report both; compare algorithms using the combined interval, excluding file I/O.

## Where we ended

At **N=100,000, M=4,096** on the recorded KITTI/gem5 experiment:

- **Main RVV LMUL 1:** **31.615 ms**; Main scalar **55.742 ms**.
- **Older F2 reference RVV:** **66.544 ms**; RadiusFPS RVV **51.240 ms**.
- At **N=100,000, M=128**, Main RVV was **12.290 ms**, while RadiusFPS RVV was **12.242 ms**: the new method does **not** win everywhere.

**Next:** Debug LMUL correctness, compare speedups by N/M and preparation/sampling, and validate beyond the current KITTI configuration.

**Files:** `spherical_fps/main_implementation/` (current code), `spherical_fps/ref_implementation/` (F2), `benchmark/stats/experiments/fps/comprehensive_benchmarks/` (six-way results + LMUL debug).
