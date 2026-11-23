# Project guide

[Overview](README.md) · [Research history and algorithm details](RESEARCH_FLOW.md)

## What we measure

- **N:** number of input 3D points; **M:** number sampled by FPS.
- **Scalar:** ordinary CPU operations; **RVV:** RISC-V vector instructions.
- **Exact FPS:** preserve the reference selection order (start at index 0; break ties by smallest original index).
- **Preparation:** build spatial data structures; **sampling:** perform repeated FPS selections.
- **gem5 simulated time:** modeled RISC-V execution time, **not** minutes spent running the simulator on a laptop.

## Repository map

| Location | Contents |
| --- | --- |
| `benchmark/src/fps/` | BasicFPS, LazyFPS, RadiusFPS and other FPS experiments |
| `benchmark/cpu/riscv_fps_comparison.py` | gem5 RISC-V FPS configuration |
| `benchmark/performance_review/` | Earlier bottleneck and profiling studies |
| `benchmark/stats/experiments/fps/comprehensive_benchmarks/` | Six-way results, logs and separate incomplete LMUL records |
| `spherical_fps/selection_patterns/` | FPS selection-pattern studies |
| `spherical_fps/geometric_bounds/` | Geometric pruning experiments |
| `spherical_fps/initial_implementation/` | First independent SphericalFPS implementation |
| `spherical_fps/ref_implementation/` | Earlier optimized F2 reference |
| `spherical_fps/main_implementation/` | Current scalar/RVV SphericalFPS |
| `spherical_fps/experimental_variants/` | Alternative optimizations and hybrids |
| `datasets/kitti/` | KITTI LiDAR point clouds |
| `benchmark/external_research/ns-fps/` | Third-party NS-FPS source; preserve its attribution |
| `documentation/` | Research references and figures |

## Read the current results

Open [`complete_six_way.csv`](benchmark/stats/experiments/fps/comprehensive_benchmarks/complete_six_way.csv). It has **26 matched (N, M) cases** for each of six implementations:

`main_scalar`, `main_rvv_m1`, `ref_scalar`, `ref_rvv_m1`, `radius_scalar`, `radius_rvv_m1`.

Compare the `sim_ms` field for the same N and M. Check `status` and `source`: **22 scalar rows were reused from earlier experiments** and were not revalidated for exact indices in this campaign. The raw `run_output.txt`, `indices.txt`, `stats.txt`, and `console.log` files document individual runs.

The separate [`incomplete_lmul_debug.csv`](benchmark/stats/experiments/fps/comprehensive_benchmarks/incomplete_lmul_debug.csv) contains partial LMUL results and correctness failures. **Do not include these in a validated speed comparison.**

## Building and running

The main source programs accept:

```text
<executable> <KITTI_XYZI.bin> <N> <M>
```

They require C++17 and, for RVV builds, a RISC-V cross-compiler with vector intrinsics. The gem5 study additionally needs the configured simulator, compatible binaries, and `m5` operations. Typical compiler settings used in the experiments are `-O3 -std=c++17 -ffp-contract=off`, with `-march=rv64gc` for scalar and `-march=rv64gcv` for RVV.

The saved benchmarks and their source files are the reproducibility record; older experimental scripts may have hardcoded paths. Build or run from the repository root and verify dataset and configuration before measuring.

## Rules for new results

Use the same dataset, N, M, gem5 machine and measured interval. Include **preparation + sampling** when comparing full algorithms. Confirm **exact selected indices** before describing a vector variant as correct. Keep raw statistics and report both performance gains and regressions.

For how the algorithms evolved and why optimizations were kept or discarded, see [RESEARCH_FLOW.md](RESEARCH_FLOW.md).
