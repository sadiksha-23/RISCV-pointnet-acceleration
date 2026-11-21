#!/usr/bin/env python3
"""Native C++ exactness + timing comparison; does NOT launch gem5."""
from pathlib import Path
import argparse
import csv
import re
import statistics
import subprocess

folder = Path(__file__).resolve().parent
root = folder.parents[2]

sources = {
    "baseline": "spherical_fps_baseline.cpp",
    "01_direct_sectors": "spherical_fps_01_direct_sectors.cpp",
    "02_radial_first": "spherical_fps_02_radial_first.cpp",
    "03_coordinate_rejection": "spherical_fps_03_coordinate_rejection.cpp",
    "04_approx_atan2": "spherical_fps_04_approx_atan2.cpp",
}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--n", type=int, default=100000)
parser.add_argument("--m", type=int, default=128)
parser.add_argument("--repeats", type=int, default=5)
parser.add_argument("--data", type=Path, default=root / "datasets/kitti/0000000000.bin")
parser.add_argument("--no-reference", action="store_true", help="Skip independent brute-force reference (faster for large M)")
parser.add_argument("--az", type=int, default=24)
parser.add_argument("--el", type=int, default=8)
parser.add_argument("--rad", type=int, default=6)
args = parser.parse_args()
if not args.data.is_file():
    parser.error(f"KITTI file not found: {args.data}")
if args.m < 1 or args.n < args.m or args.repeats < 1:
    parser.error("Need N >= M >= 1 and repeats >= 1")

bin_folder = root / "bin"
bin_folder.mkdir(exist_ok=True)
result_folder = folder / "results"
result_folder.mkdir(exist_ok=True)

def compile_one(name, file):
    exe=bin_folder / f"sfps_test_{name}"
    command=["g++", "-O3", "-std=c++17", "-ffp-contract=off",
             f"-DAZ_BINS={args.az}", f"-DEL_BINS={args.el}", f"-DRAD_BINS={args.rad}",
             str(folder/file), "-o", str(exe)]
    subprocess.run(command, check=True)
    return exe

def run(exe):
    proc=subprocess.run([str(exe),str(args.data),str(args.n),str(args.m)],
                        check=True, capture_output=True, text=True)
    out=proc.stdout
    inds=[int(v) for v in re.findall(r"^FPS_INDEX\[\d+\]=(\d+)$",out,re.M)]
    if len(inds)!=args.m:
        raise RuntimeError(f"Unexpected number of selected indices ({len(inds)}) from {exe}")
    def time(key):
        value=re.search(rf"^{key}=([0-9.]+)$",out,re.M)
        if not value: raise RuntimeError(f"Missing {key} in {exe}")
        return float(value.group(1))
    return inds, (time("PREPARATION_MS"),time("SAMPLING_MS"),time("TOTAL_MS"))

print("Compiling native C++ variants (-O3, -ffp-contract=off)...",flush=True)
executables={name:compile_one(name,filename) for name,filename in sources.items()}
print(f"N={args.n}, M={args.m}, grid={args.az}x{args.el}x{args.rad}, repeats={args.repeats}",flush=True)

baseline_indices,_=run(executables["baseline"])
if not args.no_reference:
    ref=bin_folder / "sfps_test_reference_basic"
    subprocess.run(["g++","-O3","-std=c++17","-ffp-contract=off",
                   str(folder/"reference_basic_fps.cpp"),"-o",str(ref)],check=True)
    print("Verifying baseline against independent BasicFPS (native, not gem5)...",flush=True)
    ref_out=subprocess.run([str(ref),str(args.data),str(args.n),str(args.m)],
                           check=True,capture_output=True,text=True).stdout
    reference=[int(v) for v in re.findall(r"^FPS_INDEX\[\d+\]=(\d+)$",ref_out,re.M)]
    if baseline_indices!=reference:
        mismatch=next((i for i,(a,b) in enumerate(zip(baseline_indices,reference)) if a!=b),None)
        raise SystemExit(f"FAIL: baseline differs from independent BasicFPS at selection {mismatch}")
    print("BasicFPS reference: PASS",flush=True)

measurements={name:[] for name in sources}
order=list(sources)
for repeat in range(args.repeats):
    # Reverse order on alternate repeats to reduce run-order bias.
    sequence=order if repeat%2==0 else list(reversed(order))
    for name in sequence:
        indices,times=run(executables[name])
        if indices!=baseline_indices:
            i=next(i for i,(a,b) in enumerate(zip(indices,baseline_indices)) if a!=b)
            raise SystemExit(f"FAIL {name}: selection {i+1}: got {indices[i]}, baseline {baseline_indices[i]}")
        measurements[name].append(times)
    print(f"Completed round {repeat+1}/{args.repeats}; all indices match",flush=True)

rows=[]
for name in sources:
    values=measurements[name]
    medians=[statistics.median(t[i] for t in values) for i in range(3)]
    rows.append({"variant":name,"N":args.n,"M":args.m,"AZ":args.az,"EL":args.el,"RAD":args.rad,
                 "preparation_ms":round(medians[0],6),"sampling_ms":round(medians[1],6),
                 "total_ms":round(medians[2],6),"matches_baseline":True})

csv_path=result_folder / f"native_N{args.n}_M{args.m}_A{args.az}_E{args.el}_R{args.rad}.csv"
with csv_path.open("w",newline="") as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print("\n=== Native C++ benchmark (median, milliseconds) ===")
print(f"{'Variant':<27} {'Prep':>10} {'Sampling':>11} {'Total':>11} {'Relative':>10}")
base_total=rows[0]["total_ms"]
for row in rows:
    print(f"{row['variant']:<27} {row['preparation_ms']:>10.3f} {row['sampling_ms']:>11.3f} "
          f"{row['total_ms']:>11.3f} {base_total/row['total_ms']:>9.3f}x")
print("\nCorrectness: all variants match baseline" + (" and independent BasicFPS" if not args.no_reference else ""))
print("CSV:",csv_path)
