#!/usr/bin/env python3
"""Compare F2's azimuth classifier with quadrant-first and branchless variants.

Save in: spherical_fps/experimental_variants/sampling_bottleneck/
Run from VS Code: python spherical_fps/experimental_variants/sampling_bottleneck/test_azimuth_variants.py

Does not modify original F2. Native C++ only (not gem5).
"""
from __future__ import annotations

import csv
import re
import shutil
import statistics
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next((p for p in (HERE, *HERE.parents) if (p / "spherical_fps/ref_implementation/spherical_fps.cpp").is_file()), None)
if ROOT is None:
    raise SystemExit("ERROR: Could not locate repository root containing spherical_fps/ref_implementation/spherical_fps.cpp")
SOURCE = ROOT / "spherical_fps/ref_implementation/spherical_fps.cpp"
DATA = ROOT / "datasets/kitti/0000000000.bin"
OUT = HERE / "results/azimuth_variants"
N = 100000
MS = (128, 256, 512, 1024)
REPEATS = 7
FLAGS = ["-O3", "-std=c++17", "-ffp-contract=off"]

# The original F2 bins the NEGATED relative XY vector. Preserve that exactly.
# Do not replace it with the un-negated vector; that changes bin IDs.
QUADRANT_HEAD = r'''
static int direct_azimuth_bin(double x,double y) {
    if(x==0.0&&y==0.0)return 0;
    static const std::vector<Ray2> boundaries=[] {
        std::vector<Ray2> rays; rays.reserve(AZ_BINS);
        for(int i=0;i<AZ_BINS;++i) {
            const double angle=TWO_PI*static_cast<double>(i)/AZ_BINS;
            double xx=std::cos(angle),yy=std::sin(angle);
            if(std::abs(xx)<1e-14)xx=0;
            if(std::abs(yy)<1e-14)yy=0;
            rays.push_back({xx,yy});
        }
        return rays;
    }();
    Ray2 q{-x,-y};
    if (AZ_BINS != 24) { // Retain F2 for other build-time bin counts.
        int lo=0,hi=AZ_BINS;
        while(lo<hi){int mid=lo+(hi-lo)/2;if(polar_less(q,boundaries[mid]))hi=mid;else lo=mid+1;}
        return std::clamp(lo-1,0,AZ_BINS-1);
    }
    // Exact axes go to the sector that begins on that axis.
    // Handle signed zero consistently with F2's half-plane predicate.
    int quadrant;
    if (q.y > 0.0) quadrant=(q.x > 0.0 ? 0 : 1);
    else if (q.y < 0.0) quadrant=(q.x < 0.0 ? 2 : 3);
    else quadrant=(q.x >= 0.0 ? 0 : 2);
    const int base=6*quadrant;
'''
QUADRANT = QUADRANT_HEAD + r'''
    // Binary search within just six sector boundaries; same predicates as F2.
    int lo=base,hi=base+6;
    while(lo<hi){int mid=lo+(hi-lo)/2;if(polar_less(q,boundaries[mid]))hi=mid;else lo=mid+1;}
    return std::clamp(lo-1,0,AZ_BINS-1);
}
'''
BRANCHLESS = QUADRANT_HEAD + r'''
    // A fixed sequence of five direction comparisons; no inner search branches.
    // Each comparison contributes 1 when q is at/after that sector boundary.
    int sector=0;
    for(int j=1;j<6;++j) {
        const Ray2& b=boundaries[base+j];
        sector += static_cast<int>(!(q.x*b.y-q.y*b.x>0.0));
    }
    return base+sector;
}
'''


def substitute_function(source: str, replacement: str) -> str:
    marker = "static int direct_azimuth_bin(double x,double y) {"
    begin = source.find(marker)
    if begin < 0:
        raise ValueError("F2 azimuth function signature has changed; aborting rather than guessing")
    brace = source.find("{", begin)
    depth = 0
    end = None
    for i in range(brace, len(source)):
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        raise ValueError("Cannot find azimuth function end")
    if source.count(marker) != 1:
        raise ValueError("Expected one azimuth function definition")
    return source[:begin] + replacement + source[end:]


def run(cmd: list[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run([str(x) for x in cmd], cwd=cwd, capture_output=True, text=True)
    if result.returncode:
        print(result.stdout[-3500:])
        print(result.stderr[-3500:])
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(map(str,cmd))}")
    return result.stdout


def read_field(s: str, key: str) -> str:
    match = re.search(rf"^{re.escape(key)}=(.+)$", s, re.MULTILINE)
    if not match:
        raise ValueError(f"Missing {key} in executable output")
    return match.group(1).strip()


def indices(s: str) -> list[int]:
    return [int(i) for i in re.findall(r"^FPS_INDEX\[\d+\]=(\d+)$", s, re.MULTILINE)]


def compiler() -> str:
    c = shutil.which("g++")
    if not c:
        raise SystemExit("ERROR: g++ not found. Use a VS Code terminal with g++ on PATH.")
    return c


def main() -> None:
    if not DATA.is_file():
        raise SystemExit(f"ERROR: KITTI frame not found: {DATA}")
    OUT.mkdir(parents=True, exist_ok=True)
    original = SOURCE.read_text(encoding="utf-8")
    variants = {"f2": original, "quadrant": substitute_function(original, QUADRANT), "branchless": substitute_function(original, BRANCHLESS)}
    cpp = {}
    exe = {}
    cc = compiler()
    suffix = ".exe" if __import__("os").name == "nt" else ""
    for name, content in variants.items():
        cpp[name] = OUT / f"{name}.cpp"
        exe[name] = OUT / f"{name}{suffix}"
        cpp[name].write_text(content, encoding="utf-8")
        cmd = [cc, *FLAGS, str(cpp[name]), "-o", str(exe[name])]
        print("Compiling", name)
        run(cmd)

    # Test the classifier itself, including axes, boundaries, adjacent float values,
    # and deterministic random directions. This detects bin differences even when
    # FPS selections coincidentally match.
    reference = original[original.index("static int direct_azimuth_bin(double x,double y) {"):original.index("static int direct_elevation_bin(double x,double y,double z) {")]
    reference = reference.replace("static int direct_azimuth_bin(", "static int reference_azimuth_bin(", 1)
    checker_body = r'''
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <limits>
int main() {
    uint64_t count=0,mismatch=0;
    auto check=[&](double x,double y) {
        const int a=reference_azimuth_bin(x,y), b=direct_azimuth_bin(x,y);
        if(a!=b) { if(mismatch<10) std::printf("MISMATCH x=%.17g y=%.17g reference=%d candidate=%d\n",x,y,a,b);++mismatch; }
        ++count;
    };
    check(0,0);check(1,0);check(-1,0);check(0,1);check(0,-1);
    for(int k=0;k<24;++k) {
        const double a=TWO_PI*static_cast<double>(k)/24.0;
        const double x=std::cos(a),y=std::sin(a);
        for(double scale : {1.0, 1e-8, 1e8}) {
            const double xx=x*scale,yy=y*scale;
            check(xx,yy);
            check(std::nextafter(xx,std::numeric_limits<double>::infinity()),yy);
            check(std::nextafter(xx,-std::numeric_limits<double>::infinity()),yy);
            check(xx,std::nextafter(yy,std::numeric_limits<double>::infinity()));
            check(xx,std::nextafter(yy,-std::numeric_limits<double>::infinity()));
        }
    }
    uint64_t state=0x8e5a3b42d191ec75ULL;
    for(int j=0;j<250000;++j) {
        state=state*2862933555777941757ULL+3037000493ULL;
        const double x=static_cast<double>((state>>12)&0xFFFFF)-524288.0;
        state=state*2862933555777941757ULL+3037000493ULL;
        const double y=static_cast<double>((state>>12)&0xFFFFF)-524288.0;
        check(x,y);
    }
    std::printf("CLASSIFIER_CASES=%llu\nCLASSIFIER_MISMATCHES=%llu\n",(unsigned long long)count,(unsigned long long)mismatch);
    return mismatch ? 1 : 0;
}
'''
    for name in ("quadrant", "branchless"):
        # Rename the normal FPS main so the checker can include all F2 helpers.
        body = '#define main original_fps_entry\n#include "' + cpp[name].name + '"\n#undef main\n' + reference + checker_body
        path = OUT / f"check_{name}.cpp"
        path.write_text(body, encoding="utf-8")
        check_exe = OUT / f"check_{name}{suffix}"
        print("Checking angular boundaries for", name)
        run([cc, *FLAGS, str(path), "-o", str(check_exe)])
        result = run([str(check_exe)])
        print(result.strip())

    rows = []
    for m in MS:
        for rep in range(REPEATS):
            # Rotate first runner to reduce consistent order bias.
            order = list(variants)
            order = order[rep % len(order):] + order[:rep % len(order)]
            outputs = {}
            for name in order:
                outputs[name] = run([str(exe[name]), str(DATA), str(N), str(m)])
            reference_indices = indices(outputs["f2"])
            if len(reference_indices) != m:
                raise RuntimeError(f"F2 returned {len(reference_indices)} indices, expected {m}")
            for name in order:
                if indices(outputs[name]) != reference_indices:
                    raise RuntimeError(f"FPS INDEX MISMATCH: {name} at M={m}, repeat {rep+1}")
                for counter in ("REGION_TESTS", "REGIONS_SKIPPED", "POINT_DISTANCE_EVALS"):
                    if read_field(outputs[name], counter) != read_field(outputs["f2"], counter):
                        raise RuntimeError(f"{counter} mismatch for {name}, M={m}, repeat {rep+1}")
                row = {"M":m,"repeat":rep+1,"variant":name}
                for key in ("PREPARATION_MS", "SAMPLING_MS", "TOTAL_MS"):
                    row[key]=float(read_field(outputs[name],key))
                rows.append(row)
            summary = ", ".join(f"{name}={float(read_field(outputs[name], 'TOTAL_MS')):.3f} ms" for name in variants)
            print(f"M={m} repeat={rep+1}: indices and counters MATCH; {summary}")
        print(f"\nM={m} MEDIANS (native, milliseconds):")
        baseline_total = statistics.median(r["TOTAL_MS"] for r in rows if r["M"]==m and r["variant"]=="f2")
        for name in variants:
            matching = [r for r in rows if r["M"]==m and r["variant"]==name]
            prep = statistics.median(r["PREPARATION_MS"] for r in matching)
            sample = statistics.median(r["SAMPLING_MS"] for r in matching)
            total = statistics.median(r["TOTAL_MS"] for r in matching)
            change = 100*(baseline_total-total)/baseline_total
            print(f"  {name:10s} prep={prep:.3f}, sampling={sample:.3f}, total={total:.3f}, total reduction={change:+.1f}%")
        print()
    with (OUT/"results.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=["M","repeat","variant","PREPARATION_MS","SAMPLING_MS","TOTAL_MS"])
        writer.writeheader();writer.writerows(rows)
    print("All FPS indices and work counters MATCH across all tested runs.")
    print("Saved generated sources and results:",OUT)
    print("Native measurements only. Windows timing variability can be large; do not compare to gem5 directly.")
    print("Original F2 was NOT modified.")

if __name__=="__main__":
    main()
