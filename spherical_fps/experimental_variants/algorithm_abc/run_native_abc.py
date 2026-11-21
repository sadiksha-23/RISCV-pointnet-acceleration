#!/usr/bin/env python3
"""Native-only A/B/C experiments. NO gem5, NO RISC-V cross compiler."""
import argparse
import csv
from pathlib import Path
import re
import statistics
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OPT=ROOT/'spherical_fps/experimental_variants/optimization_tests'
SOURCES={
 '01_reference': OPT/'spherical_fps_01_direct_sectors.cpp',
 'A_aabb_bound': HERE/'spherical_fps_A_aabb_bound.cpp',
 'B_adaptive_regions': HERE/'spherical_fps_B_adaptive_regions.cpp',
 'C_radial_short_circuit': HERE/'spherical_fps_C_radial_short_circuit.cpp',
}

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--n',type=int,default=100000)
parser.add_argument('--m',type=int,default=128)
parser.add_argument('--repeats',type=int,default=5)
parser.add_argument('--data',type=Path,default=ROOT/'datasets/kitti/0000000000.bin')
parser.add_argument('--az',type=int,default=24)
parser.add_argument('--el',type=int,default=8)
parser.add_argument('--rad',type=int,default=6)
parser.add_argument('--no-basic-reference',action='store_true',help='Skip independent BasicFPS correctness check')
args=parser.parse_args()
if not args.data.is_file():parser.error(f'Dataset not found: {args.data}')
if not 1<=args.m<=args.n or args.repeats<1:parser.error('Need N >= M >= 1 and repeats >= 1')

# Generate A/B/C from the checked-in optimization 01 baseline (without altering it).
subprocess.run([sys.executable,str(HERE/'generate_abc.py')],check=True)
BIN=ROOT/'bin'; BIN.mkdir(exist_ok=True)
RESULT=HERE/'results'; RESULT.mkdir(exist_ok=True)
compiled={}
for name,source in SOURCES.items():
    target=BIN/f'sfps_abc_{name}'
    command=['g++','-O3','-std=c++17','-ffp-contract=off',
             f'-DAZ_BINS={args.az}',f'-DEL_BINS={args.el}',f'-DRAD_BINS={args.rad}',
             str(source),'-o',str(target)]
    print('Compiling',name,flush=True)
    subprocess.run(command,check=True)
    compiled[name]=target

def run(exe):
    p=subprocess.run([str(exe),str(args.data),str(args.n),str(args.m)],capture_output=True,text=True,check=True)
    output=p.stdout
    indices=[int(x) for x in re.findall(r'^FPS_INDEX\[\d+\]=(\d+)$',output,re.M)]
    if len(indices)!=args.m:raise RuntimeError(f'{exe}: FPS indices missing')
    def val(k):
        m=re.search(rf'^{re.escape(k)}=([0-9.]+)$',output,re.M)
        if not m:raise RuntimeError(f'{exe}: missing {k}')
        return float(m.group(1))
    return {'indices':indices,'prep':val('PREPARATION_MS'), 'sampling':val('SAMPLING_MS'),
            'total':val('TOTAL_MS'),'evals':int(val('POINT_DISTANCE_EVALS')),
            'regions':int(val('OCCUPIED_REGIONS')),'checks':int(val('REGION_TESTS')),
            'skips':int(val('REGIONS_SKIPPED'))}

expected=run(compiled['01_reference'])['indices']
if not args.no_basic_reference:
    basic=BIN/'sfps_abc_basic_reference'
    print('Compiling independent BasicFPS...',flush=True)
    subprocess.run(['g++','-O3','-std=c++17','-ffp-contract=off',str(OPT/'reference_basic_fps.cpp'),'-o',str(basic)],check=True)
    out=subprocess.run([str(basic),str(args.data),str(args.n),str(args.m)],check=True,capture_output=True,text=True).stdout
    basics=[int(x) for x in re.findall(r'^FPS_INDEX\[\d+\]=(\d+)$',out,re.M)]
    if basics!=expected:
        raise SystemExit('ERROR: 01 baseline does not match independent BasicFPS reference!')
    print('Independent BasicFPS: PASS',flush=True)

runs={name:[] for name in SOURCES}
order=list(SOURCES)
for repeat in range(args.repeats):
    for name in (order if repeat%2==0 else list(reversed(order))):
        v=run(compiled[name])
        if v['indices']!=expected:
            mismatch=next((i for i,(x,y) in enumerate(zip(v['indices'],expected)) if x!=y),None)
            raise SystemExit(f'FAIL: {name} selected index at step {mismatch} differs from reference')
        runs[name].append(v)
    print(f'Round {repeat+1}/{args.repeats}: FPS indices match in all variants',flush=True)

rows=[]
for name,values in runs.items():
    record={'variant':name,'n':args.n,'m':args.m,'az':args.az,'el':args.el,'rad':args.rad}
    for key in ['prep','sampling','total','evals','regions','checks','skips']:
        record[key]=statistics.median(v[key] for v in values)
    record['indices_match']=True
    rows.append(record)
baseline=rows[0]
print('\nNATIVE ONLY — median results, same workload / compiler:')
print(f'{"Variant":<25} {"Prep":>9} {"Sampling":>9} {"Total":>9} {"Speedup":>9} {"Dist evals":>12} {"Regions":>8}')
for r in rows:
    print(f'{r["variant"]:<25} {r["prep"]:9.3f} {r["sampling"]:9.3f} {r["total"]:9.3f} {baseline["total"]/r["total"]:8.3f}x {int(r["evals"]):12d} {int(r["regions"]):8d}')
print('\nNote: distance counts CAN differ for A/B/C; exact selected indices must match.')
path=RESULT/f'abc_native_N{args.n}_M{args.m}_A{args.az}_E{args.el}_R{args.rad}.csv'
with path.open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print('Saved:',path)
