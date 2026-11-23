#!/usr/bin/env python3
"""Matched gem5 validation for three exact SphericalFPS experimental variants.

Put this file in:
  spherical_fps/experimental_variants/sampling_bottleneck/test_gem5_three_variants.py

Sources:
  - Original F2 from ref_implementation/spherical_fps.cpp (never modified)
  - Combined F2 from prior test_combined_vs_cartesian_radius.py output
  - AABB-first derived conservatively from original F2
  - Combined + AABB-first derived conservatively from combined F2

Default schedule: N=50000,100000 and M=128,512,1024; 3 variants = 18 gem5 runs.
Preparation and sampling are separate explicit gem5 intervals. Total is their sum.
The script verifies each variant's selected indices against original F2 on the
native host BEFORE starting gem5, and verifies all gem5 results against the
native F2 reference. A matching F2 reference is NOT itself a BasicFPS proof.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next((p for p in (HERE, *HERE.parents)
             if (p / "spherical_fps/ref_implementation/spherical_fps.cpp").is_file()), None)
if ROOT is None:
    raise SystemExit("Place this script inside the RISCV-pointnet-acceleration repository.")

F2 = ROOT / "spherical_fps/ref_implementation/spherical_fps.cpp"
COMBINED = HERE / "results/combined_vs_cartesian_radius/f2_combined_quadrant_sign_first.cpp"
RESULTS = HERE / "results/gem5_three_variants"
DATASET = ROOT / "datasets/kitti/0000000000.bin"
GEM5 = ROOT / "gem5/build/RISCV/gem5.opt"
CPU_CONFIG = ROOT / "benchmark/cpu/riscv_fps_comparison.py"
M5_LIB = ROOT / "gem5/util/m5/build/riscv/out/libm5.a"
M5_INCLUDE = ROOT / "gem5/include"

# The original source uses this exact F2 skip condition.
SKIP_ANCHOR = "const double lb=lower_bound(g,xyz,newest,selected_radius);"
SKIP_CALL = (
    "if (aabb_can_skip_early(g,xyz,newest,g.max_nearest_sq)) {\n"
    "                ++c.skipped_regions;\n"
    "                c.skipped_points+=g.members.size();\n"
    "                continue;\n"
    "            }\n"
    "            " + SKIP_ANCHOR
)
AABB_HELPER = r"""
// Strictly conservative early-out: matches the original lower_bound()
// safety reduction and comparison margin, but tests the AABB alone.
// If it cannot prove a skip, the original three-bound calculation runs.
static inline bool aabb_can_skip_early(const Region& g,
                                      const std::vector<float>& xyz,
                                      int newest, float current_max_sq) {
    const size_t p=static_cast<size_t>(newest)*3;
    const double x=static_cast<double>(xyz[p]);
    const double y=static_cast<double>(xyz[p+1]);
    const double z=static_cast<double>(xyz[p+2]);
    const double bx=std::max({g.xmin-x,0.0,x-g.xmax});
    const double by=std::max({g.ymin-y,0.0,y-g.ymax});
    const double bz=std::max({g.zmin-z,0.0,z-g.zmax});
    const double raw_lb=std::sqrt(bx*bx+by*by+bz*bz);
    const double lb=std::max(0.0,raw_lb-1e-6*(1.0+raw_lb));
    const double max_sq=static_cast<double>(current_max_sq);
    const double margin=1e-5*(1.0+std::abs(max_sq));
    return lb*lb > max_sq+margin;
}

"""


def replace_once(text: str, old: str, new: str, context: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{context}: expected exactly one anchor, found {count}: {old[:90]!r}")
    return text.replace(old, new, 1)


def instrument_intervals(text: str) -> str:
    # Both sources retain the original single ROI until this transformation.
    # We introduce one dump/reset at the separation between preparation
    # and sampling, giving two nonoverlapping simulated intervals.
    marker = "const auto end_prep=std::chrono::steady_clock::now();"
    return replace_once(
        text, marker,
        marker + "\n        m5_dump_stats(0,0);\n        m5_reset_stats(0,0);",
        "gem5 preparation/sampling split"
    )


def make_aabb_first(text: str) -> str:
    # Inserting the helper before lower_bound() preserves the existing
    # sampling logic, including the original conservative fallback.
    begin = "static double lower_bound("
    text = replace_once(text, begin, AABB_HELPER + begin, "AABB helper insertion")
    text = replace_once(text, SKIP_ANCHOR, SKIP_CALL, "AABB-first early-out")
    return text


def run(command: list[str], *, output: Path | None = None) -> str:
    print("Running:", " ".join(map(str, command)), flush=True)
    proc = subprocess.run([str(x) for x in command], cwd=ROOT,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, errors="replace")
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(proc.stdout, encoding="utf-8")
    if proc.returncode:
        raise RuntimeError(
            f"Command failed (exit {proc.returncode}): {' '.join(map(str,command))}\n"
            f"Last output:\n{proc.stdout[-5000:]}"
        )
    return proc.stdout


def get_indices(text: str, expected_m: int) -> list[int]:
    pairs = re.findall(r"^FPS_INDEX\[(\d+)\]=(\d+)\s*$", text, flags=re.M)
    if len(pairs) != expected_m:
        raise RuntimeError(f"Expected {expected_m} FPS_INDEX lines, got {len(pairs)}.")
    if [int(i) for i, _ in pairs] != list(range(expected_m)):
        raise RuntimeError("FPS_INDEX numbering is inconsistent.")
    return [int(value) for _, value in pairs]


def get_counters(text: str) -> tuple[str, str, str]:
    keys = ("REGION_TESTS", "REGIONS_SKIPPED", "POINT_DISTANCE_EVALS")
    found = dict(re.findall(r"^([A-Z][A-Z0-9_]*)=([^\r\n]+)", text, flags=re.M))
    try:
        return tuple(found[k] for k in keys)
    except KeyError as exc:
        raise RuntimeError(f"Missing F2 counter: {exc}") from exc


def simtick_snapshots(stats: Path) -> list[int]:
    if not stats.is_file():
        raise RuntimeError(f"Missing gem5 stats: {stats}")
    ticks = [int(x) for x in re.findall(r"^\s*simTicks\s+(\d+)\b",
                                       stats.read_text(encoding="utf-8", errors="replace"), re.M)]
    if len(ticks) < 2:
        raise RuntimeError(f"Expected two gem5 explicit stats intervals in {stats}, found {ticks}.")
    return ticks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, nargs="+", default=[50000, 100000])
    parser.add_argument("--m", type=int, nargs="+", default=[128, 512, 1024])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--gem5", type=Path, default=GEM5)
    parser.add_argument("--config", type=Path, default=CPU_CONFIG)
    parser.add_argument("--m5-lib", type=Path, default=M5_LIB)
    parser.add_argument("--m5-include", type=Path, default=M5_INCLUDE)
    parser.add_argument("--compiler", default="riscv64-linux-gnu-g++")
    parser.add_argument("--native-compiler", default="g++")
    parser.add_argument("--native-only", action="store_true",
                        help="Generate sources, compile natively, verify indices; do not run gem5.")
    parser.add_argument("--compile-only", action="store_true",
                        help="Native checks and compile RISC-V binaries, but do not simulate.")
    parser.add_argument("--include-f2", action="store_true",
                        help="Also rerun original F2 on gem5 (six more runs per repeat).")
    args = parser.parse_args()

    if args.repeats < 1 or min(args.n) < 1 or min(args.m) < 1 or max(args.m) > min(args.n):
        parser.error("Invalid N/M/repeats.")
    if not args.dataset.is_file():
        parser.error(f"KITTI dataset not found: {args.dataset}")
    if not COMBINED.is_file():
        parser.error(
            "Missing combined source: " + str(COMBINED) + "\n"
            "Run test_combined_vs_cartesian_radius.py first, or keep its generated C++ file."
        )
    if not shutil.which(args.native_compiler):
        parser.error(f"Native compiler not found: {args.native_compiler}")
    if not args.native_only:
        for name, p in [("gem5", args.gem5), ("CPU config", args.config),
                        ("m5 library", args.m5_lib), ("gem5 include", args.m5_include)]:
            if not p.exists():
                parser.error(f"Missing {name}: {p}")
        if not shutil.which(args.compiler):
            parser.error(f"RISC-V compiler not found: {args.compiler}")

    src_out = RESULTS / "sources"
    bin_out = RESULTS / "bin"
    native_out = RESULTS / "native_checks"
    for p in (src_out, bin_out, native_out):
        p.mkdir(parents=True, exist_ok=True)

    # Build from the two EXISTING implementations—never edit ref_implementation.
    original = F2.read_text(encoding="utf-8")
    combined = COMBINED.read_text(encoding="utf-8")
    if "f2_elevation_above(" not in combined:
        raise RuntimeError("Combined F2 source doesn't contain sign-first elevation.")
    if "static inline bool f2_elevation_above(" not in combined:
        raise RuntimeError("Combined F2 missing elevation helper; regenerate with corrected runner.")
    if "direct_azimuth_bin" not in combined:
        raise RuntimeError("Combined F2 missing azimuth classifier.")
    sources = {
        "f2": instrument_intervals(original),
        "combined": instrument_intervals(combined),
        "aabb_first": instrument_intervals(make_aabb_first(original)),
        "combined_aabb_first": instrument_intervals(make_aabb_first(combined))
    }
    for name, source in sources.items():
        (src_out / f"{name}.cpp").write_text(source, encoding="utf-8")

    # First validate selected indices and work counters using real binaries
    # (not just classifying angles independently).
    native_bins: dict[str, Path] = {}
    for name in sources:
        exe = native_out / f"{name}{'.exe' if sys.platform == 'win32' else ''}"
        run([args.native_compiler, "-O3", "-std=c++17", "-ffp-contract=off",
             str(src_out / f"{name}.cpp"), "-o", str(exe)],
            output=native_out / f"compile_{name}.txt")
        native_bins[name] = exe

    references: dict[tuple[int, int], list[int]] = {}
    for n in args.n:
        for m in args.m:
            observed: dict[str, list[int]] = {}
            counts: dict[str, tuple[str,str,str]] = {}
            for name, exe in native_bins.items():
                output = run([str(exe), str(args.dataset), str(n), str(m)],
                             output=native_out / f"N{n}_M{m}_{name}.txt")
                observed[name] = get_indices(output, m)
                counts[name] = get_counters(output)
            reference = observed["f2"]
            for name in ("combined", "aabb_first", "combined_aabb_first"):
                if observed[name] != reference:
                    idx = next(i for i, (x,y) in enumerate(zip(reference,observed[name])) if x != y)
                    raise RuntimeError(f"NATIVE INDEX MISMATCH: N={n}, M={m}, variant={name}, step={idx}")
                if counts[name] != counts["f2"]:
                    raise RuntimeError(f"NATIVE WORK COUNTER MISMATCH: N={n}, M={m}, variant={name}")
            references[(n,m)] = reference
            print(f"N={n}, M={m}: native FPS indices and work counters MATCH.", flush=True)

    if args.native_only:
        print("Native preflight passed. No gem5 runs requested.")
        return

    binaries: dict[str, Path] = {}
    for name in sources:
        elf = bin_out / f"{name}.riscv"
        run([args.compiler, "-O3", "-std=c++17", "-static", "-ffp-contract=off",
             "-DENABLE_GEM5", "-I", str(args.m5_include),
             str(src_out / f"{name}.cpp"), str(args.m5_lib), "-o", str(elf)],
            output=bin_out / f"compile_{name}.txt")
        binaries[name] = elf

    if args.compile_only:
        print("All four RISC-V executables compiled. No gem5 simulations requested.")
        return

    cases = ["combined", "aabb_first", "combined_aabb_first"]
    if args.include_f2:
        cases.insert(0, "f2")
    rows: list[dict] = []
    print(f"Launching {len(args.n)*len(args.m)*len(cases)*args.repeats} gem5 runs.", flush=True)

    for n in args.n:
        for m in args.m:
            for repeat in range(1, args.repeats+1):
                for name in cases:
                    out = RESULTS / f"N{n}_M{m}_{name}_r{repeat}"
                    out.mkdir(parents=True, exist_ok=True)
                    output = run([str(args.gem5), f"--outdir={out}", str(args.config),
                                  str(binaries[name]), str(args.dataset), str(n), str(m)],
                                 output=out / "output.txt")
                    selected = get_indices(output, m)
                    if selected != references[(n,m)]:
                        idx = next(i for i,(a,b) in enumerate(zip(references[(n,m)],selected)) if a!=b)
                        raise RuntimeError(f"GEM5 INDEX MISMATCH at N={n}, M={m}, variant={name}, step={idx}.")
                    prep_ticks, sample_ticks = simtick_snapshots(out / "stats.txt")[:2]
                    # Prior project convention: 2 GHz simulated CPU, 1e9 ticks/ms.
                    prep_ms = prep_ticks/1e9
                    sample_ms = sample_ticks/1e9
                    row = dict(n=n,m=m,variant=name,repeat=repeat,prep_ticks=prep_ticks,
                               sampling_ticks=sample_ticks,total_ticks=prep_ticks+sample_ticks,
                               preparation_ms=prep_ms,sampling_ms=sample_ms,
                               total_ms=prep_ms+sample_ms,indices_match="YES")
                    rows.append(row)
                    print(f"N={n} M={m} {name}: prep={prep_ms:.4f} ms, "
                          f"sampling={sample_ms:.4f} ms, total={prep_ms+sample_ms:.4f} ms; indices MATCH",
                          flush=True)
                    with (RESULTS / "summary.csv").open("w", newline="", encoding="utf-8") as file:
                        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                        writer.writeheader()
                        writer.writerows(rows)

    (RESULTS / "experiment_metadata.json").write_text(
        json.dumps({"n":args.n, "m":args.m, "repeats":args.repeats,
                    "variants":cases, "dataset":str(args.dataset),
                    "timing":"two explicit non-overlapping gem5 ROI intervals; total is sum; "
                             "not numerically identical to historical single-ROI runs",
                    "correctness":"variants compared to original F2; BasicFPS not independently checked"},
                   indent=2), encoding="utf-8")
    print("\nComplete. CSV:", RESULTS / "summary.csv")
    print("IMPORTANT: Two-ROI totals should not be called directly interchangeable with older "
          "single-ROI measurements. Use --include-f2 for a matched baseline.")


if __name__ == "__main__":
    main()
