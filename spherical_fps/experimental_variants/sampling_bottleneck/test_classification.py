#!/usr/bin/env python3
"""Diagnostic native microbenchmark of existing F2 classification routines.

Place at spherical_fps/experimental_variants/sampling_bottleneck/test_classification.py
Run from anywhere: python path/to/test_classification.py
No modifications to the original F2 source.
"""
from __future__ import annotations

import csv
import re
import shutil
import statistics
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next((p for p in HERE.parents if (p / "spherical_fps/main_implementation/spherical_fps.cpp").exists()), None)
if ROOT is None:
    raise SystemExit("Cannot locate repository root containing spherical_fps/main_implementation/spherical_fps.cpp")
SRC = ROOT / "spherical_fps/main_implementation/spherical_fps.cpp"
DATA = ROOT / "datasets/kitti/0000000000.bin"
OUT = HERE / "results" / "classification_profile"
N = 100000
REPEATS = 15
TRIALS = 9

CPP_EXTENSION = r'''

// Diagnostic only. This intentionally measures isolated classification loops,
// not exact substage shares of the original combined build_regions() pass.
#include <atomic>
#include <array>

static volatile uint64_t classification_sink = 0;
using BenchClock=std::chrono::steady_clock;
static inline double bench_ms(BenchClock::time_point a,BenchClock::time_point b) {
    return std::chrono::duration<double,std::milli>(b-a).count();
}

int main(int argc,char** argv) {
    if(argc!=2) {std::fprintf(stderr,"Usage: %s KITTI_XYZI.bin\n", argv[0]);return 2;}
    std::vector<float> xyz;
    if(!load_kitti_xyz(argv[1],xyz)) return 2;
    const int n=BENCH_N;
    if(xyz.size()/3 < static_cast<size_t>(n)) {std::fprintf(stderr,"Not enough KITTI points\n");return 2;}
    double origin[3]; calculate_centroid(xyz,n,origin);
    std::vector<double> x(n),y(n),z(n);
    for(int i=0;i<n;++i) {
        const size_t p=static_cast<size_t>(i)*3;
        x[i]=static_cast<double>(xyz[p])-origin[0];
        y[i]=static_cast<double>(xyz[p+1])-origin[1];
        z[i]=static_cast<double>(xyz[p+2])-origin[2];
    }
    std::vector<int> az(n),el(n),combined_az(n),combined_el(n);
    std::vector<double> rsq(n),combined_rsq(n);
    // Establish identical outputs to the original classification expressions.
    for(int i=0;i<n;++i) {
        combined_az[i]=direct_azimuth_bin(x[i],y[i]);
        combined_el[i]=direct_elevation_bin(x[i],y[i],z[i]);
        combined_rsq[i]=x[i]*x[i]+y[i]*y[i]+z[i]*z[i];
    }
    const char* kinds[]={"azimuth","elevation","radius_sq","combined"};
    for(int trial=0;trial<BENCH_TRIALS;++trial) {
        // Rotate which case runs first to reduce systematic ordering bias.
        for(int step=0;step<4;++step) {
            const int kind=(trial+step)%4;
            const auto begin=BenchClock::now();
            for(int r=0;r<BENCH_REPEATS;++r) {
                if(kind==0) {
                    for(int i=0;i<n;++i) az[i]=direct_azimuth_bin(x[i],y[i]);
                } else if(kind==1) {
                    for(int i=0;i<n;++i) el[i]=direct_elevation_bin(x[i],y[i],z[i]);
                } else if(kind==2) {
                    for(int i=0;i<n;++i) rsq[i]=x[i]*x[i]+y[i]*y[i]+z[i]*z[i];
                } else {
                    for(int i=0;i<n;++i) {
                        az[i]=direct_azimuth_bin(x[i],y[i]);
                        el[i]=direct_elevation_bin(x[i],y[i],z[i]);
                        rsq[i]=x[i]*x[i]+y[i]*y[i]+z[i]*z[i];
                    }
                }
                std::atomic_signal_fence(std::memory_order_seq_cst);
            }
            const auto end=BenchClock::now();
            // Ensure outputs were computed correctly and remain observable.
            uint64_t checksum=0;
            if(kind==0||kind==3) for(int i=0;i<n;++i) {
                if(az[i]!=combined_az[i]) {std::fprintf(stderr,"Azimuth mismatch at %d\n",i);return 3;}
                checksum+=static_cast<uint64_t>(az[i]);
            }
            if(kind==1||kind==3) for(int i=0;i<n;++i) {
                if(el[i]!=combined_el[i]) {std::fprintf(stderr,"Elevation mismatch at %d\n",i);return 3;}
                checksum+=static_cast<uint64_t>(el[i]);
            }
            if(kind==2||kind==3) for(int i=0;i<n;++i) {
                if(rsq[i]!=combined_rsq[i]) {std::fprintf(stderr,"Squared radius mismatch at %d\n",i);return 3;}
                checksum+=static_cast<uint64_t>(rsq[i]);
            }
            classification_sink=classification_sink+checksum;
            std::printf("RESULT,%d,%s,%.9f,%llu\n",trial,kinds[kind],
                bench_ms(begin,end)/BENCH_REPEATS,(unsigned long long)checksum);
        }
    }
    std::printf("VERIFIED=YES\n");
    return 0;
}
'''

def run(cmd: list[str]) -> str:
    print("Running:", " ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    if result.returncode:
        print(result.stdout)
        print(result.stderr)
        raise SystemExit(f"Command failed (exit {result.returncode})")
    return result.stdout

def main() -> None:
    if not DATA.exists():
        raise SystemExit(f"KITTI input not found: {DATA}")
    compiler = shutil.which("g++")
    if not compiler:
        raise SystemExit("g++ not found. Use a VS Code terminal with g++ available, or WSL.")
    OUT.mkdir(parents=True, exist_ok=True)
    original = SRC.read_text(encoding="utf-8")
    # Reuse the actual F2 routines verbatim; replace only main for the microbenchmark.
    pattern = r'\bint\s+main\s*\(\s*int\s+argc\s*,\s*char\s*\*\s*\*\s*argv\s*\)\s*\{'
    matches = list(re.finditer(pattern, original))
    if len(matches) != 1:
        raise SystemExit("Unexpected F2 main signature. Refusing to generate a potentially invalid experiment.")
    code = original[:matches[0].start()] + "#define BENCH_N " + str(N) + "\n#define BENCH_TRIALS " + str(TRIALS) + "\n#define BENCH_REPEATS " + str(REPEATS) + "\n" + CPP_EXTENSION
    cpp = OUT / "f2_classification_microbench.cpp"
    cpp.write_text(code, encoding="utf-8")
    binary = OUT / ("f2_classification_microbench.exe" if __import__("os").name == "nt" else "f2_classification_microbench")
    run([compiler, "-O3", "-std=c++17", "-ffp-contract=off", str(cpp), "-o", str(binary)])
    output = run([str(binary), str(DATA)])
    (OUT / "raw_output.txt").write_text(output, encoding="utf-8")
    if "VERIFIED=YES" not in output:
        raise SystemExit("Diagnostic did not complete correctness verification")
    measurements = {key: [] for key in ["azimuth", "elevation", "radius_sq", "combined"]}
    rows = []
    for line in output.splitlines():
        if line.startswith("RESULT,"):
            _, trial, kind, ms, checksum = line.split(",")
            row = {"trial": int(trial), "stage": kind, "ms_per_100k": float(ms), "checksum": checksum}
            measurements[kind].append(float(ms))
            rows.append(row)
    if any(len(a) != TRIALS for a in measurements.values()):
        raise SystemExit("Missing trial results")
    with (OUT / "trials.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["trial", "stage", "ms_per_100k", "checksum"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nN={N}, {TRIALS} trials x {REPEATS} repeats, native CPU only")
    print("All individual classification results MATCH the combined F2 expressions.")
    print("Median time per 100,000 points (relative coordinates precomputed):")
    for kind, values in measurements.items():
        print(f"  {kind:12s}: {statistics.median(values):.4f} ms  (range {min(values):.4f}–{max(values):.4f})")
    print(f"\nOutputs saved to {OUT}")
    print("Caution: isolated microbenchmarks exclude centroid, coordinate subtraction, allocation, and region assignment.")
    print("Do not interpret these as exact shares of F2 preparation or compare native times to gem5.")

if __name__ == "__main__":
    main()
