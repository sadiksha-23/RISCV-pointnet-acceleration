#!/usr/bin/env python3
"""Native correctness & stage timing, optional gem5 evaluation after native pass."""
import argparse,csv,re,statistics,subprocess,sys
from pathlib import Path
here=Path(__file__).resolve().parent
root=here.parents[2]
P={
 '01_reference':'spherical_fps_01_direct_sectors.cpp',
 '05_elevation_squared':'spherical_fps_05_elevation_squared.cpp',
 '06_radial_squared':'spherical_fps_06_radial_squared.cpp',
 '07_region_radius_squared':'spherical_fps_07_region_radius_squared.cpp',
}
p=argparse.ArgumentParser()
p.add_argument('--mode',choices=['native','gem5','both'],default='native')
p.add_argument('--data',type=Path,default=root/'datasets/kitti/0000000000.bin')
p.add_argument('--n',type=int,default=100000)
p.add_argument('--m',type=int,default=128)
p.add_argument('--repeats',type=int,default=5)
p.add_argument('--az',type=int,default=24);p.add_argument('--el',type=int,default=8);p.add_argument('--rad',type=int,default=6)
p.add_argument('--skip-basic',action='store_true',help='Skip independent brute-force BasicFPS reference')
a=p.parse_args()
if not a.data.is_file():p.error(f'dataset missing: {a.data}')
if not 1<=a.m<=a.n or a.repeats<1:p.error('Require N >= M >= 1 and repeats >= 1')
a.data=a.data.resolve()
bin=root/'bin';bin.mkdir(exist_ok=True)
res=here/'results'/'prep_sqrt';res.mkdir(parents=True,exist_ok=True)
subprocess.run([sys.executable,str(here/'generate_variants.py')],check=True)
flags=['-O3','-std=c++17','-ffp-contract=off',f'-DAZ_BINS={a.az}',f'-DEL_BINS={a.el}',f'-DRAD_BINS={a.rad}']

def parse(out):
    ids=[int(x) for x in re.findall(r'^FPS_INDEX\[\d+\]=(\d+)$',out,re.M)]
    if len(ids)!=a.m:raise RuntimeError(f'Expected {a.m} indices, found {len(ids)}')
    def get(k):
        m=re.search(rf'^{re.escape(k)}=([0-9.]+)$',out,re.M)
        if not m:raise RuntimeError(f'Missing {k}')
        return float(m.group(1))
    return ids,[get(k) for k in ['PREPARATION_MS','SAMPLING_MS','TOTAL_MS']],int(get('POINT_DISTANCE_EVALS'))

def compile_native():
    d={}
    for name,f in P.items():
        exe=bin/f'sfps_prep_{name}'
        subprocess.run(['g++',*flags,str(here/f),'-o',str(exe)],check=True)
        d[name]=exe
    return d

def launch(exe):
    return subprocess.run([str(exe),str(a.data),str(a.n),str(a.m)],capture_output=True,text=True,check=True).stdout

def validate(exes):
    reference,_,ref_evals=parse(launch(exes['01_reference']))
    if not a.skip_basic:
        bexe=bin/'sfps_prep_basic_reference'
        subprocess.run(['g++','-O3','-std=c++17','-ffp-contract=off',str(here/'reference_basic_fps.cpp'),'-o',str(bexe)],check=True)
        basic=subprocess.run([str(bexe),str(a.data),str(a.n),str(a.m)],capture_output=True,text=True,check=True).stdout
        basic_ids=[int(x) for x in re.findall(r'^FPS_INDEX\[\d+\]=(\d+)$',basic,re.M)]
        if basic_ids!=reference:raise RuntimeError('01 differs from independent BasicFPS')
        print('Independent BasicFPS: PASS',flush=True)
    timings={k:[] for k in P}; counts={}
    for r in range(a.repeats):
        names=list(P) if r%2==0 else list(reversed(P))
        for name in names:
            ids,t,ev=parse(launch(exes[name]))
            if ids!=reference:
                i=next(i for i,(x,y) in enumerate(zip(ids,reference)) if x!=y)
                raise RuntimeError(f'{name} mismatch at FPS_INDEX[{i}] got {ids[i]} expected {reference[i]}')
            if ev!=ref_evals:
                raise RuntimeError(f'{name} changed pruning/evaluation count: {ev} != {ref_evals}')
            timings[name].append(t);counts[name]=ev
        print(f'Native round {r+1}/{a.repeats}: indices and evaluation counts MATCH',flush=True)
    rows=[]
    for name in P:
        t=[statistics.median(z[i] for z in timings[name]) for i in range(3)]
        rows.append(dict(variant=name,prep_ms=t[0],sampling_ms=t[1],total_ms=t[2],evals=counts[name],correctness='PASS'))
    outfile=res/f'native_N{a.n}_M{a.m}_A{a.az}_E{a.el}_R{a.rad}.csv'
    with outfile.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('\nNATIVE (median ms; NOT gem5)')
    for r in rows:print(f"{r['variant']:26s} prep={r['prep_ms']:9.4f} sample={r['sampling_ms']:9.4f} total={r['total_ms']:9.4f} speedup={rows[0]['total_ms']/r['total_ms']:.3f}x")
    print('Saved:',outfile)
    return reference,ref_evals

def simulate(reference,ref_evals):
    gem5=root/'gem5/build/RISCV/gem5.opt'
    m5lib=root/'gem5/util/m5/build/riscv/out/libm5.a'
    cpu=root/'benchmark/cpu/riscv_fps_comparison.py'
    for f in (gem5,m5lib,cpu):
        if not f.is_file():raise FileNotFoundError(f'Missing gem5 dependency: {f}')
    rows=[]
    for name,source in P.items():
        exe=bin/f'sfps_prep_{name}.riscv'
        subprocess.run(['riscv64-linux-gnu-g++',*flags,'-static','-DENABLE_GEM5',f'-I{root/"gem5/include"}',str(here/source),str(m5lib),'-o',str(exe)],check=True)
        outdir=res/f'gem5_{name}_N{a.n}_M{a.m}_A{a.az}_E{a.el}_R{a.rad}'
        outdir.mkdir(parents=True,exist_ok=True)
        print('gem5',name,flush=True)
        cmd=[str(gem5),f'--outdir={outdir}',str(cpu),str(exe),str(a.data),str(a.n),str(a.m)]
        proc=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        (outdir/'output.txt').write_text(proc.stdout)
        if proc.returncode:raise RuntimeError(f'gem5 failed for {name}: see {outdir}/output.txt')
        ids,_,evals=parse(proc.stdout)
        if ids!=reference or evals!=ref_evals:raise RuntimeError(f'gem5 correctness/counters mismatch for {name}')
        stats=(outdir/'stats.txt').read_text()
        # One gem5 reset/dump pair = a single ROI including prep + sample.
        ticks=[int(v) for v in re.findall(r'^simTicks\s+(\d+)',stats,re.M)]
        if not ticks:raise RuntimeError(f'No simTicks found in {outdir}/stats.txt')
        full_ms=ticks[0]/1e9
        rows.append(dict(variant=name,gem5_total_roi_ms=full_ms,point_evals=evals,indices_match=True))
    # This source uses ONE ROI (prep+sampling), so gem5 provides a valid TOTAL only.
    out=res/f'gem5_total_N{a.n}_M{a.m}_A{a.az}_E{a.el}_R{a.rad}.csv'
    with out.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('\ngem5 total ROI (ms; prep + sampling; no load/printing)')
    for r in rows:print(f"{r['variant']:26s} total={r['gem5_total_roi_ms']:.4f} speedup={rows[0]['gem5_total_roi_ms']/r['gem5_total_roi_ms']:.3f}x")
    print('Saved:',out)

try:
    exes=compile_native()
    refs,evals=validate(exes)
    if a.mode in ('gem5','both'):simulate(refs,evals)
except (Exception,subprocess.CalledProcessError) as e:
    print('FAIL:',e,file=sys.stderr);sys.exit(1)
