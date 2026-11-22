#!/usr/bin/env python3
"""Compare original scalar F2 against an exact AABB-first short-circuit variant.

Run from VS Code or any terminal:
    python spherical_fps/experimental_variants/sampling_bottleneck/test_aabb_first.py

Does not modify spherical_fps.cpp. Generates a C++ experiment file, builds two
native executables, checks all selected FPS indices, and writes a CSV summary.
These are *native host* timings, NOT gem5 results.
"""

import argparse
import csv
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "spherical_fps/main_implementation/spherical_fps.cpp"
DEFAULT_DATA = ROOT / "datasets/kitti/0000000000.bin"
DEFAULT_RESULTS = Path(__file__).resolve().parent / "results/aabb_first"

# The original combined bound subtracts this margin from max(xyz, radial, AABB).
# For AABB alone we apply the same margin: if it can skip, the combined bound
# necessarily can skip, too. If not, the original combined function is called.
HELPER = r'''
// Diagnostic AABB-first early rejection, using exactly F2's AABB formula.
static double aabb_lower_bound_first(const Region& g, const std::vector<float>& xyz, int newest) {
    const size_t p = static_cast<size_t>(newest) * 3;
    const double x = static_cast<double>(xyz[p]);
    const double y = static_cast<double>(xyz[p + 1]);
    const double z = static_cast<double>(xyz[p + 2]);
    const double bx = std::max({g.xmin-x, 0.0, x-g.xmax});
    const double by = std::max({g.ymin-y, 0.0, y-g.ymax});
    const double bz = std::max({g.zmin-z, 0.0, z-g.zmax});
    const double aabb_lb = std::sqrt(bx*bx + by*by + bz*bz);
    return std::max(0.0, aabb_lb - 1e-6*(1.0 + aabb_lb));
}
'''

OLD = '''            const double lb=lower_bound(g,xyz,newest,selected_radius);
            const double max_sq=static_cast<double>(g.max_nearest_sq);
            const double margin=1e-5*(1.0+std::abs(max_sq));
            if (lb*lb > max_sq+margin) {'''

NEW = '''            const double max_sq=static_cast<double>(g.max_nearest_sq);
            const double margin=1e-5*(1.0+std::abs(max_sq));
            const double aabb_lb=aabb_lower_bound_first(g,xyz,newest);
            if (aabb_lb*aabb_lb > max_sq+margin) {
                ++c.skipped_regions;c.skipped_points+=g.members.size();continue;
            }
            // AABB was inconclusive: retain the original combined-bound decision.
            const double lb=lower_bound(g,xyz,newest,selected_radius);
            if (lb*lb > max_sq+margin) {'''


def variant_source(original: str) -> str:
    anchor = "static std::vector<int> spherical_fps("
    if original.count(anchor) != 1 or original.count(OLD) != 1:
        raise RuntimeError("F2 source structure changed: refusing an unsafe transformation. Check spherical_fps.cpp.")
    return original.replace(anchor, HELPER + "\n" + anchor, 1).replace(OLD, NEW, 1)


def run_command(command: list[str], *, cwd: Path = ROOT) -> str:
    completed = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed ({completed.returncode}): {' '.join(command)}\n{completed.stdout[-6000:]}")
    return completed.stdout


def parse_output(output: str) -> dict:
    values = dict(re.findall(r"^([A-Z_]+)=(.+)$", output, flags=re.MULTILINE))
    indices = [int(x) for x in re.findall(r"^FPS_INDEX\[\d+\]=(\d+)$", output, flags=re.MULTILINE)]
    required = ("PREPARATION_MS", "SAMPLING_MS", "TOTAL_MS", "POINT_DISTANCE_EVALS", "REGIONS_SKIPPED")
    if not all(k in values for k in required) or not indices:
        raise RuntimeError("Expected timing/counter/index output not found.\n" + output[-3000:])
    return {
        "indices": indices,
        "prep_ms": float(values["PREPARATION_MS"]),
        "sample_ms": float(values["SAMPLING_MS"]),
        "total_ms": float(values["TOTAL_MS"]),
        "point_evals": int(values["POINT_DISTANCE_EVALS"]),
        "skipped_regions": int(values["REGIONS_SKIPPED"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA, help="KITTI XYZI .bin frame")
    ap.add_argument("--n", type=int, default=100000, help="input points")
    ap.add_argument("--m", nargs="+", type=int, default=[128, 256, 512, 1024], help="sample counts")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--compiler", default="g++")
    ap.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = ap.parse_args()
    if not SOURCE.is_file():
        ap.error(f"Missing F2 source: {SOURCE}")
    if not args.data.is_file():
        ap.error(f"Missing input: {args.data}")
    if args.n < 1 or args.repeats < 1 or any(m < 1 or m > args.n for m in args.m):
        ap.error("Require n >= 1, repeats >= 1, and 1 <= m <= n")

    results = args.results.resolve()
    results.mkdir(parents=True, exist_ok=True)
    variant = results / "spherical_fps_aabb_first.cpp"
    variant.write_text(variant_source(SOURCE.read_text(encoding="utf-8")), encoding="utf-8")
    suffix = ".exe" if os.name == "nt" else ""
    baseline_exe = results / ("f2_original" + suffix)
    variant_exe = results / ("f2_aabb_first" + suffix)
    for source, target in ((SOURCE, baseline_exe), (variant, variant_exe)):
        cmd = [args.compiler, "-O3", "-std=c++17", "-ffp-contract=off", str(source), "-o", str(target)]
        print("Compiling:", " ".join(cmd), flush=True)
        run_command(cmd)

    raw_rows = []
    summary_rows = []
    for m in args.m:
        samples = {"original": [], "aabb_first": []}
        for repetition in range(1, args.repeats + 1):
            outputs = {}
            # Alternate execution order to reduce systematic warmed-cache effects.
            order = ("original", "aabb_first") if repetition % 2 else ("aabb_first", "original")
            for label in order:
                exe = baseline_exe if label == "original" else variant_exe
                output = run_command([str(exe), str(args.data), str(args.n), str(m)])
                (results / f"N{args.n}_M{m}_run{repetition}_{label}.txt").write_text(output, encoding="utf-8")
                outputs[label] = parse_output(output)
            base = outputs["original"]
            opt = outputs["aabb_first"]
            if base["indices"] != opt["indices"]:
                first = next((i for i, (a, b) in enumerate(zip(base["indices"], opt["indices"])) if a != b), None)
                raise RuntimeError(f"FPS INDEX MISMATCH at N={args.n} M={m} run={repetition}, step={first}; see saved outputs")
            if (base["point_evals"], base["skipped_regions"]) != (opt["point_evals"], opt["skipped_regions"]):
                raise RuntimeError(f"Work counters differ at N={args.n} M={m} run={repetition}; inspect source transformation")
            for label, record in outputs.items():
                samples[label].append(record)
                raw_rows.append({"N": args.n, "M": m, "run": repetition, "variant": label, "prep_ms": record["prep_ms"], "sampling_ms": record["sample_ms"], "total_ms": record["total_ms"], "point_evals": record["point_evals"], "skipped_regions": record["skipped_regions"]})
            print(f"N={args.n} M={m} run={repetition}: indices MATCH | original sampling {base['sample_ms']:.3f} ms | AABB-first {opt['sample_ms']:.3f} ms", flush=True)
        orig_sampling = statistics.median(r["sample_ms"] for r in samples["original"])
        opt_sampling = statistics.median(r["sample_ms"] for r in samples["aabb_first"])
        orig_total = statistics.median(r["total_ms"] for r in samples["original"])
        opt_total = statistics.median(r["total_ms"] for r in samples["aabb_first"])
        sampling_gain = 100 * (orig_sampling - opt_sampling) / orig_sampling if orig_sampling else 0
        total_gain = 100 * (orig_total - opt_total) / orig_total if orig_total else 0
        summary_rows.append({"N": args.n, "M": m, "runs": args.repeats, "original_sampling_median_ms": orig_sampling, "aabb_first_sampling_median_ms": opt_sampling, "sampling_reduction_pct": sampling_gain, "original_total_median_ms": orig_total, "aabb_first_total_median_ms": opt_total, "total_reduction_pct": total_gain, "correctness": "MATCH", "point_evals": samples["original"][0]["point_evals"], "skipped_regions": samples["original"][0]["skipped_regions"]})
        print(f"M={m}: sampling median {orig_sampling:.3f} -> {opt_sampling:.3f} ms ({sampling_gain:+.1f}% reduction); full total {orig_total:.3f} -> {opt_total:.3f} ms ({total_gain:+.1f}% reduction)\n", flush=True)

    for filename, rows in (("raw.csv", raw_rows), ("summary.csv", summary_rows)):
        with (results / filename).open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print("All indices and work counters MATCH. Saved results:", results)
    print("NOTE: Native CPU measurements only; don't compare these timings directly with gem5.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
