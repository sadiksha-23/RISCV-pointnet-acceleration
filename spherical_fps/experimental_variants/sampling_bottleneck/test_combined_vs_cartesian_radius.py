#!/usr/bin/env python3
"""Native controlled comparison: original F2, combined angular F2, Cartesian F2, RadiusFPS.

Put this script in spherical_fps/experimental_variants/sampling_bottleneck/.
It reuses locally generated source from prior successful azimuth/elevation/Cartesian tests.
Does not alter original repository source files.
"""
from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next((p for p in [HERE, *HERE.parents] if (p / "spherical_fps/ref_implementation/spherical_fps.cpp").exists()), None)
if ROOT is None:
    raise SystemExit("Cannot find repository root. Put script under spherical_fps/experimental_variants/sampling_bottleneck/.")
OUT = HERE / "results" / "combined_vs_cartesian_radius"
F2 = ROOT / "spherical_fps/ref_implementation/spherical_fps.cpp"
RAD = ROOT / "benchmark/src/fps/RadiusFPS/kitti/radius_fps_radix_full.cpp"
DATA = ROOT / "datasets/kitti/0000000000.bin"


def function_span(code: str, name: str) -> tuple[int,int]:
    pattern = re.compile(r"static\s+int\s+" + re.escape(name) + r"\s*\([^;]*?\)\s*\{", re.S)
    m = pattern.search(code)
    if not m:
        raise ValueError(f"Cannot find definition of {name}")
    level = 0
    for i in range(m.end()-1, len(code)):
        if code[i] == "{": level += 1
        elif code[i] == "}":
            level -= 1
            if level == 0:
                return m.start(), i+1
    raise ValueError(f"Unterminated function: {name}")


def find_variant(directory: Path, method: str, keyword: str) -> Path:
    candidates = [p for p in directory.rglob("*.cpp") if keyword in p.stem.lower() and "check" not in p.stem.lower() and "microbench" not in p.stem.lower()]
    # A matching full F2 program has main() and the target function.
    valid = []
    for p in candidates:
        body = p.read_text(encoding="utf-8")
        if "int main(" in body and f"{method}(" in body and "static std::vector<Region> build_regions(" in body:
            valid.append(p)
    if len(valid) != 1:
        available = "\n".join(str(p) for p in candidates[:20]) or "(none)"
        raise SystemExit(f"Expected exactly one generated {keyword} F2 source under {directory}; found {len(valid)}.\nCandidates:\n{available}\nRun the earlier {keyword} experiment first.")
    return valid[0]


def build_combined() -> tuple[Path, Path, Path]:
    az_dir = HERE / "results/azimuth_variants"
    el_dir = HERE / "results/elevation_variants"
    az_path = find_variant(az_dir, "direct_azimuth_bin", "quadrant")
    el_path = find_variant(el_dir, "direct_elevation_bin", "sign_first")
    cart_candidates = [p for p in (HERE / "results").rglob("spherical_fps_cartesian_partition.cpp") if "combined_vs_cartesian_radius" not in p.parts]
    if len(cart_candidates) != 1:
        raise SystemExit(f"Need exactly one Cartesian source 'spherical_fps_cartesian_partition.cpp' in {HERE / 'results'}. Found {len(cart_candidates)}. Run test_partitioning.py first.")
    cart_path = cart_candidates[0]
    f2 = F2.read_text(encoding="utf-8")
    az_source = az_path.read_text(encoding="utf-8")
    el_source = el_path.read_text(encoding="utf-8")
    # Replace the two classification functions. The sign-first elevation function
    # depends on f2_elevation_above(), which must be copied along with it.
    combined = f2
    for name, source in [("direct_azimuth_bin", az_source), ("direct_elevation_bin", el_source)]:
        ss, se = function_span(source, name)
        ds, de = function_span(combined, name)
        combined = combined[:ds] + source[ss:se] + combined[de:]

    helper_marker = "static inline bool f2_elevation_above("
    if combined.count("f2_elevation_above(") and helper_marker not in combined:
        hs = el_source.find(helper_marker)
        if hs < 0:
            raise RuntimeError("Sign-first source uses f2_elevation_above but does not define it")
        opening = el_source.find("{", hs)
        depth = 0
        closing = None
        for i in range(opening, len(el_source)):
            if el_source[i] == "{": depth += 1
            elif el_source[i] == "}":
                depth -= 1
                if depth == 0:
                    closing = i + 1
                    break
        if closing is None:
            raise RuntimeError("Unterminated f2_elevation_above helper")
        helper = el_source[hs:closing]
        ds, _ = function_span(combined, "direct_elevation_bin")
        combined = combined[:ds] + helper + "\n" + combined[ds:]
    if combined.count("f2_elevation_above(") and helper_marker not in combined:
        raise RuntimeError("Missing f2_elevation_above definition in combined C++ source")
    combined_path = OUT / "f2_combined_quadrant_sign_first.cpp"
    combined_path.write_text(combined, encoding="utf-8")
    return combined_path, cart_path, az_path


def make_native_radius() -> Path:
    source = RAD.read_text(encoding="utf-8")
    original_include = '#include <gem5/m5ops.h>'
    if source.count(original_include) != 1:
        raise RuntimeError("RadiusFPS m5ops include differs from expected code")
    source = source.replace(original_include, '#ifdef ENABLE_GEM5\n#include <gem5/m5ops.h>\n#else\nstatic inline void m5_reset_stats(uint64_t, uint64_t) {}\nstatic inline void m5_dump_stats(uint64_t, uint64_t) {}\n#endif', 1)
    source = '#include <chrono>\n' + source
    marker_start = '    m5_reset_stats(0, 0);\n'
    marker_sample = '    radiusFarthestPointSampling(\n        n, m,'
    marker_end = '    m5_dump_stats(0, 0);\n'
    for marker in [marker_start, marker_sample, marker_end]:
        if source.count(marker) != 1:
            raise RuntimeError(f"Radius source changed, cannot safely instrument: {marker!r}")
    source = source.replace(marker_start, marker_start + '    const auto prep_start = std::chrono::steady_clock::now();\n', 1)
    source = source.replace(marker_sample, '    const auto prep_end = std::chrono::steady_clock::now();\n    const auto sample_start = prep_end;\n' + marker_sample, 1)
    source = source.replace(marker_end, '    const auto sample_end = std::chrono::steady_clock::now();\n' + marker_end, 1)
    source = source.replace('    printf("ALGORITHM=RadiusFPS_Radix\\n");', '''    const double prep_ms = std::chrono::duration<double, std::milli>(prep_end - prep_start).count();
    const double sample_ms = std::chrono::duration<double, std::milli>(sample_end - sample_start).count();
    printf("PREPARATION_MS=%.6f\\nSAMPLING_MS=%.6f\\nTOTAL_MS=%.6f\\n", prep_ms, sample_ms, prep_ms + sample_ms);
    printf("ALGORITHM=RadiusFPS_Radix\\n");''', 1)
    path = OUT / "radius_fps_radix_full_native_timed.cpp"
    path.write_text(source, encoding="utf-8")
    return path


def run(cmd: list[str]) -> str:
    result = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}\nSTDOUT:\n{result.stdout[-4000:]}\nSTDERR:\n{result.stderr[-4000:]}")
    return result.stdout


def parse(text: str) -> dict:
    kv = dict(re.findall(r"^([A-Z][A-Z0-9_]*)=([^\r\n]+)$", text, flags=re.M))
    indices = [int(v) for v in re.findall(r"^FPS_INDEX\[\d+\]=(\d+)$", text, flags=re.M)]
    for name in ["PREPARATION_MS", "SAMPLING_MS", "TOTAL_MS"]:
        if name not in kv:
            raise ValueError(f"Missing {name} in program output")
    return {"indices": indices, "prep": float(kv["PREPARATION_MS"]), "sample": float(kv["SAMPLING_MS"]), "total": float(kv["TOTAL_MS"]), "counts": {k:kv.get(k) for k in ["REGION_TESTS", "REGIONS_SKIPPED", "POINT_DISTANCE_EVALS"]}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--m", type=int, nargs="+", default=[128,256,512,1024])
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--compiler", default="g++")
    ap.add_argument("--dataset", type=Path, default=DATA)
    args = ap.parse_args()
    if args.n < 1 or args.repeats < 1 or any(m < 1 or m > args.n for m in args.m):
        ap.error("Invalid N, M, or repeats")
    if not args.dataset.is_file():
        ap.error(f"Missing KITTI frame: {args.dataset}")
    if not shutil.which(args.compiler):
        ap.error(f"Compiler not found: {args.compiler}. Use the same VS Code environment as previous experiments.")
    OUT.mkdir(parents=True, exist_ok=True)
    combined, cart, az = build_combined()
    radius = make_native_radius()
    print(f"Using quadrant-first source: {az}")
    print(f"Using Cartesian source: {cart}")
    sources = {"f2": F2, "combined": combined, "cartesian": cart, "radiusfps": radius}
    exe_ext = ".exe" if sys.platform == "win32" else ""
    binaries = {}
    for name, src in sources.items():
        binary = OUT / (name + exe_ext)
        cmd = [args.compiler, "-O3", "-std=c++17", "-ffp-contract=off", str(src), "-o", str(binary)]
        print("Compiling", name)
        run(cmd)
        binaries[name] = binary

    rows = []
    rng = random.Random(20261121)
    for m in args.m:
        for trial in range(1,args.repeats+1):
            order = list(sources)
            rng.shuffle(order)
            observed = {}
            for name in order:
                result = run([str(binaries[name]), str(args.dataset), str(args.n), str(m)])
                (OUT / f"N{args.n}_M{m}_repeat{trial}_{name}.txt").write_text(result, encoding="utf-8")
                observed[name] = parse(result)
                if len(observed[name]["indices"]) != m:
                    raise RuntimeError(f"{name} returned {len(observed[name]['indices'])} indices; expected {m}")
            expected = observed["f2"]["indices"]
            for name in sources:
                if observed[name]["indices"] != expected:
                    diff = next(i for i,(x,y) in enumerate(zip(expected,observed[name]["indices"])) if x != y)
                    raise RuntimeError(f"INDEX MISMATCH at M={m}, trial={trial}, step={diff}: F2={expected[diff]}, {name}={observed[name]['indices'][diff]}. See saved outputs; no performance conclusion should be drawn.")
            if observed["f2"]["counts"] != observed["combined"]["counts"]:
                raise RuntimeError(f"Combined F2 changed pruning work counters at M={m}, trial={trial}")
            for name in sources:
                a = observed[name]
                rows.append({"n":args.n,"m":m,"trial":trial,"algorithm":name,"preparation_ms":a["prep"],"sampling_ms":a["sample"],"total_ms":a["total"]})
            print(f"M={m} trial={trial}: indices MATCH | " + " ".join(f"{k}={observed[k]['total']:.3f}ms" for k in sources))
        print(f"\nM={m} median total/prep/sample (native ms):")
        medians = {}
        for name in sources:
            group = [r for r in rows if r["m"]==m and r["algorithm"]==name]
            med = {k: statistics.median(r[k] for r in group) for k in ("preparation_ms","sampling_ms","total_ms")}
            medians[name] = med
            print(f"  {name:10s} prep={med['preparation_ms']:.3f} sample={med['sampling_ms']:.3f} total={med['total_ms']:.3f}")
        for name in ["combined","cartesian","radiusfps"]:
            delta = 100*(medians["f2"]["total_ms"]-medians[name]["total_ms"])/medians["f2"]["total_ms"]
            print(f"  {name} total reduction relative to F2: {delta:+.1f}%")
        print()
    csv_path = OUT / "summary.csv"
    with csv_path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    print("All FPS indices matched the original F2 for the tested frame/configurations.")
    print("Output:", OUT)
    print("Native only; simulated gem5 times are not interchangeable with these. Different implementations use different grouping and work patterns.")

if __name__ == "__main__":
    main()
