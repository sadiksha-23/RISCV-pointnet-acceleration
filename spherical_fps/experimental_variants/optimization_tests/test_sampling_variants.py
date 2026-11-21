#!/usr/bin/env python3
"""Native SphericalFPS 01 vs S1/S2/S3; exactness and stage timing. No gem5."""
import argparse
import csv
from pathlib import Path
import re
import statistics
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]  # optimization_tests -> experimental_variants -> spherical_fps -> repo
SOURCES = {
    "01_reference": "spherical_fps_01_reference.cpp",
    "S1_packed_regions": "spherical_fps_S1_packed_regions.cpp",
    "S2_local_updates": "spherical_fps_S2_local_updates.cpp",
    "S3_two_pass": "spherical_fps_S3_two_pass.cpp",
}
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--data', type=Path, default=ROOT/'datasets/kitti/0000000000.bin')
p.add_argument('--n', type=int, default=100000)
p.add_argument('--m', type=int, default=128)
p.add_argument('--repeats', type=int, default=5)
p.add_argument('--az', type=int, default=24)
p.add_argument('--el', type=int, default=8)
p.add_argument('--rad', type=int, default=6)
p.add_argument('--no-reference', action='store_true', help='Skip independent BasicFPS (large M can take longer)')
a=p.parse_args()
if not a.data.is_file(): p.error(f'Missing KITTI file: {a.data}; pass --data PATH')
if a.n < a.m or a.m < 1 or a.repeats < 1: p.error('Require N >= M >= 1, repeats >= 1')
BIN=ROOT/'bin'; BIN.mkdir(parents=True, exist_ok=True)
RESULTS=HERE/'results'; RESULTS.mkdir(exist_ok=True)


def compile_cpp(name, file):
    exe=BIN / ('sfps_sweep_'+name)
    cmd=['g++','-O3','-std=c++17','-ffp-contract=off',
         f'-DAZ_BINS={a.az}',f'-DEL_BINS={a.el}',f'-DRAD_BINS={a.rad}',
         str(HERE/file),'-o',str(exe)]
    subprocess.run(cmd,check=True)
    return exe

def execute(exe):
    output=subprocess.run([str(exe),str(a.data),str(a.n),str(a.m)],
                          check=True,text=True,capture_output=True).stdout
    ids=[int(v) for v in re.findall(r'^FPS_INDEX\[\d+\]=(\d+)$',output,re.M)]
    if len(ids)!=a.m: raise ValueError(f'Missing selected points in {exe}')
    def number(key):
        x=re.search(rf'^{re.escape(key)}=([0-9.]+)$',output,re.M)
        if not x: raise ValueError(f'Missing {key} in {exe}')
        return float(x.group(1))
    times=[number('PREPARATION_MS'),number('SAMPLING_MS'),number('TOTAL_MS')]
    counts=[number('POINT_DISTANCE_EVALS'),number('OCCUPIED_REGIONS')]
    return ids,times,counts

print('Compiling four native C++ implementations...', flush=True)
executables={name:compile_cpp(name, filename) for name, filename in SOURCES.items()}
print(f'N={a.n}, M={a.m}, grid={a.az}x{a.el}x{a.rad}, repeats={a.repeats}',flush=True)
expected,_,expected_counts=execute(executables['01_reference'])
if not a.no_reference:
    exe=BIN/'sfps_sweep_basic_reference'
    subprocess.run(['g++','-O3','-std=c++17','-ffp-contract=off',
                    str(HERE/'reference_basic_fps.cpp'),'-o',str(exe)],check=True)
    out=subprocess.run([str(exe),str(a.data),str(a.n),str(a.m)],
                       check=True,capture_output=True,text=True).stdout
    basic=[int(x) for x in re.findall(r'^FPS_INDEX\[\d+\]=(\d+)$',out,re.M)]
    if basic != expected:
        mismatch=next((i for i,(x,y) in enumerate(zip(basic,expected)) if x!=y),None)
        raise SystemExit(f'FAIL: 01 baseline differs from BasicFPS at zero-based step {mismatch}')
    print('Independent BasicFPS: PASS',flush=True)

runs={name:[] for name in SOURCES}
for round_i in range(a.repeats):
    names=list(SOURCES)
    if round_i%2: names.reverse()
    for name in names:
        ids, timing, counts=execute(executables[name])
        if ids!=expected:
            mismatch=next(i for i,(x,y) in enumerate(zip(ids,expected)) if x!=y)
            raise SystemExit(f'FAIL {name}: index[{mismatch}]={ids[mismatch]}, expected {expected[mismatch]}')
        if counts!=expected_counts:
            raise SystemExit(f'FAIL {name}: evaluated-distances/occupied-region counters changed: {counts} vs {expected_counts}')
        runs[name].append(timing)
    print(f'Round {round_i+1}/{a.repeats}: selection indices + evaluation counts MATCH',flush=True)

rows=[]
for name in SOURCES:
    medians=[statistics.median(sample[i] for sample in runs[name]) for i in range(3)]
    rows.append({'variant':name,'n':a.n,'m':a.m,'az':a.az,'el':a.el,'rad':a.rad,
                 'prep_ms':round(medians[0],6),'sampling_ms':round(medians[1],6),
                 'total_ms':round(medians[2],6),
                 'point_evals':int(expected_counts[0]),'indices_match':True})
path=RESULTS/f'sampling_native_N{a.n}_M{a.m}_A{a.az}_E{a.el}_R{a.rad}.csv'
with path.open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
base=rows[0]
print('\n=== NATIVE TEST (median, ms) ===')
print(f'{"Variant":<22} {"Prep":>9} {"Sampling":>10} {"Total":>10} {"Sample speedup":>16} {"Full speedup":>13}')
for row in rows:
    print(f'{row["variant"]:<22} {row["prep_ms"]:9.3f} {row["sampling_ms"]:10.3f} {row["total_ms"]:10.3f} '
          f'{base["sampling_ms"]/row["sampling_ms"]:15.3f}x {base["total_ms"]/row["total_ms"]:12.3f}x')
print('Selection correctness: PASS; evaluation count unchanged')
print('Results CSV:',path)
