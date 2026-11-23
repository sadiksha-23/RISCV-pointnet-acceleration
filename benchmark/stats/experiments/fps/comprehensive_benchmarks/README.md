# FPS benchmark comparison — November 22, 2026

This folder contains the RISC-V gem5 KITTI benchmarking campaign for FPS implementations.

## Primary comparison

**[complete_six_way.csv](complete_six_way.csv)** contains **156 entries:** six complete 26-configuration sweeps, grouped as:

| Implementation | Entries | New or previously completed PASS | Reused historical |
| --- | ---: | ---: | ---: |
| `main_scalar` | 26 | 26 | 0 |
| `main_rvv_m1` | 26 | 26 | 0 |
| `ref_scalar` | 26 | 15 | 11 |
| `ref_rvv_m1` | 26 | 26 | 0 |
| `radius_scalar` | 26 | 15 | 11 |
| `radius_rvv_m1` | 26 | 26 | 0 |

All six groups share the same 26 (N, M) configurations. The primary file deliberately excludes partial LMUL 2/4/8 experiments and failed correctness checks.

**Important measurement caveats:**
- `sim_ms` measures gem5 simulated time, **not** the real-world time taken to run gem5.
- A `PASS` means the overnight runner marked the run successful; the full logs and per-run files are retained for auditing.
- `REUSED_HISTORICAL` entries were imported from earlier measurements, not rerun in this campaign. Their `source` column identifies the original stats file. The runner notes that selected indices were **not validated here** for these imported rows.
- For reused historical rows, `prep_ms`, `sample_ms`, and `total_ms` may be blank. Do **not** substitute `sim_ms` as a measured phase split.
- Exact-index correctness statements should only be made where the comparison was actually checked, not inferred from timing results.

## Incomplete LMUL experiments

**[incomplete_lmul_debug.csv](incomplete_lmul_debug.csv)** isolates the 11 partial LMUL records. These are **not** included in the primary six-way comparison.

Known failures:
- `radius_rvv_m2`, N=1024, M=128: incorrect output from step 1 compared with RadiusFPS scalar.
- `main_rvv_m2`, N=100000, M=512: incorrect output from step 350 compared with Main scalar.

LMUL 4/8 were only completed at the initial preflight configuration. Investigate correctness before drawing conclusions from partial LMUL timing results.

## Original data

- **[summary.csv](summary.csv)** is the untouched combined record, including partial/failed experiments.
- The per-implementation `N*_M*` directories contain original run outputs and statistics.
- **[console.log](console.log)** holds execution chronology and error messages.

Do **not** delete the raw outputs or historical source data; they support verification and later debugging. The two curated CSV files are derived views, not substitutes for the underlying results.
