#!/usr/bin/env python3
"""Native tests: AABB-only, partial-distance early-out, and both.
Place next to test_gem5_three_variants.py; run: python test_sampling_next_ideas.py
Uses the generated combined_aabb_first.cpp. Does not modify existing sources.
"""
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
ROOT = next((p for p in [HERE, *HERE.parents] if (p / 'spherical_fps/main_implementation/spherical_fps.cpp').is_file()), None)
if ROOT is None:
    sys.exit('Put this script under spherical_fps/experimental_variants/sampling_bottleneck/.')
BASE = HERE / 'results/gem5_three_variants/sources/combined_aabb_first.cpp'
DATASET = ROOT / 'datasets/kitti/0000000000.bin'
OUT = HERE / 'results/sampling_next_ideas'

# Match the actual block written by the previous gem5 runner. Fail on mismatch.
AABB_PATTERN = re.compile(r'''if\s*\(\s*aabb_can_skip_early\s*\(\s*g\s*,\s*xyz\s*,\s*newest\s*,\s*g\.max_nearest_sq\s*\)\s*\)\s*\{\s*\+\+c\.skipped_regions\s*;\s*c\.skipped_points\s*\+=\s*g\.members\.size\(\)\s*;\s*continue\s*;\s*\}\s*const double lb\s*=\s*lower_bound\s*\(\s*g\s*,\s*xyz\s*,\s*newest\s*,\s*selected_radius\s*\)\s*;\s*const double max_sq\s*=\s*static_cast<double>\s*\(\s*g\.max_nearest_sq\s*\)\s*;\s*const double margin\s*=\s*1e-5\s*\*\s*\(\s*1\.0\s*\+\s*std::abs\s*\(\s*max_sq\s*\)\s*\)\s*;\s*if\s*\(\s*lb\s*\*\s*lb\s*>\s*max_sq\s*\+\s*margin\s*\)\s*\{\s*\+\+c\.skipped_regions\s*;\s*c\.skipped_points\s*\+=\s*g\.members\.size\(\)\s*;\s*continue\s*;\s*\}''', re.S)
UPDATE_PATTERN = re.compile(r'''for\s*\(\s*int i\s*:\s*g\.members\s*\)\s*\{\s*const float d\s*=\s*sqdist\s*\(\s*xyz\s*,\s*i\s*,\s*newest\s*\)\s*;\s*\+\+c\.evaluated_distances\s*;\s*if\s*\(\s*d\s*<\s*nearest\[i\]\s*\)\s*nearest\[i\]\s*=\s*d\s*;''', re.S)

AABB_ONLY = '''if (aabb_can_skip_early(g,xyz,newest,g.max_nearest_sq)) {
                ++c.skipped_regions;
                c.skipped_points+=g.members.size();
                continue;
            }'''
# IMPORTANT: unlike the tempting RadiusFPS dx/dy/dz standalone tests,
# each bound below is safe because squared components are nonnegative.
# Preserve float math order in the final d expression, and strict comparisons.
PARTIAL_UPDATE = '''for(int i:g.members){
                ++c.evaluated_distances;
                const size_t pi=static_cast<size_t>(i)*3;
                const size_t pn=static_cast<size_t>(newest)*3;
                const float dx=xyz[pi]-xyz[pn];
                const float dx2=dx*dx;
                if (dx2<nearest[i]) {
                    const float dy=xyz[pi+1]-xyz[pn+1];
                    const float dy2=dy*dy;
                    if (dy2<nearest[i]) {
                        const float dz=xyz[pi+2]-xyz[pn+2];
                        const float dz2=dz*dz;
                        if (dz2<nearest[i]) {
                            const float d=dx2+dy2+dz2;
                            if(d<nearest[i])nearest[i]=d;
                        }
                    }
                }'''

def change_once(source, pattern, repl, label):
    out, count = pattern.subn(lambda m: repl, source)
    if count != 1:
        raise RuntimeError(f'{label}: expected exactly one C++ match, found {count}. No source altered.')
    return out

def run(cmd):
    proc=subprocess.run([str(x) for x in cmd],capture_output=True,text=True,errors='replace')
    if proc.returncode:
        raise RuntimeError(f'Command failed: {" ".join(map(str,cmd))}\n{proc.stderr[-4000:]}\n{proc.stdout[-2000:]}')
    return proc.stdout

def parse(output,m):
    indices=[(int(i),int(v)) for i,v in re.findall(r'^FPS_INDEX\[(\d+)\]=(\d+)\s*$',output,re.M)]
    if len(indices)!=m or [i for i,_ in indices]!=list(range(m)):
        raise RuntimeError(f'Expected {m} ordered FPS indices, got {len(indices)}')
    def val(key):
        match=re.search(rf'^{key}=([0-9.]+)\s*$',output,re.M)
        if match is None: raise RuntimeError(f'Missing {key}')
        return float(match.group(1))
    return {'indices':[v for _,v in indices], 'prep_ms':val('PREPARATION_MS'),
            'sampling_ms':val('SAMPLING_MS'),'total_ms':val('TOTAL_MS'),
            'point_visits':int(val('POINT_DISTANCE_EVALS')),
            'regions_skipped':int(val('REGIONS_SKIPPED'))}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset',type=Path,default=DATASET)
    ap.add_argument('--n',type=int,nargs='+',default=[50000,100000])
    ap.add_argument('--m',type=int,nargs='+',default=[128,512,1024])
    ap.add_argument('--trials',type=int,default=7)
    ap.add_argument('--compiler',default='g++')
    args=ap.parse_args()
    if not BASE.is_file(): ap.error(f'Missing generated source: {BASE}\nRun the earlier gem5 runner first.')
    if not args.dataset.is_file(): ap.error(f'KITTI dataset missing: {args.dataset}')
    if not shutil.which(args.compiler): ap.error(f'Compiler missing: {args.compiler}')
    if args.trials<1 or min(args.n)<1 or min(args.m)<1 or max(args.m)>min(args.n): ap.error('Invalid N/M/trials')
    OUT.mkdir(parents=True,exist_ok=True)
    base=BASE.read_text(encoding='utf-8')
    aabb=change_once(base,AABB_PATTERN,AABB_ONLY,'AABB-only')
    partial=change_once(base,UPDATE_PATTERN,PARTIAL_UPDATE,'Partial-distance')
    both=change_once(aabb,UPDATE_PATTERN,PARTIAL_UPDATE,'AABB-only + partial')
    sources={'current_combined_aabb':base,'aabb_only':aabb,'partial_distance':partial,'aabb_only_partial':both}
    bins={}
    for name,src in sources.items():
        source_file=OUT/f'{name}.cpp';source_file.write_text(src,encoding='utf-8')
        binary=OUT/(name+('.exe' if sys.platform=='win32' else ''))
        print('Compiling:',name,flush=True)
        run([args.compiler,'-O3','-std=c++17','-ffp-contract=off',str(source_file),'-o',str(binary)])
        bins[name]=binary
    allrows=[];summaries=[]
    for n in args.n:
        for m in args.m:
            for trial in range(1,args.trials+1):
                order=list(sources);random.Random(100000*n+100*m+trial).shuffle(order)
                results={}
                for name in order:
                    results[name]=parse(run([str(bins[name]),str(args.dataset),str(n),str(m)]),m)
                reference=results['current_combined_aabb']['indices']
                for name,result in results.items():
                    if result['indices']!=reference:
                        pos=next(i for i,(a,b) in enumerate(zip(reference,result['indices'])) if a!=b)
                        raise RuntimeError(f'FPS INDICES DIFFER: N={n}, M={m}, trial={trial}, variant={name}, step={pos}')
                    if name=='partial_distance' and result['point_visits']!=results['current_combined_aabb']['point_visits']:
                        raise RuntimeError('Partial-distance changed point visitation counts.')
                    allrows.append({'N':n,'M':m,'trial':trial,'variant':name,**{k:v for k,v in result.items() if k!='indices'},'indices_match':'YES'})
                print(f'N={n} M={m} trial={trial}/{args.trials}: indices MATCH | '+ ' '.join(f'{name}={results[name]["sampling_ms"]:.3f}ms' for name in sources),flush=True)
            baseline=statistics.median(r['sampling_ms'] for r in allrows if r['N']==n and r['M']==m and r['variant']=='current_combined_aabb')
            print(f'\nN={n}, M={m}, native median sampling ms:',flush=True)
            for name in sources:
                relevant=[r for r in allrows if r['N']==n and r['M']==m and r['variant']==name]
                median=statistics.median(r['sampling_ms'] for r in relevant)
                visits=statistics.median(r['point_visits'] for r in relevant)
                gain=100*(baseline-median)/baseline
                summaries.append({'N':n,'M':m,'variant':name,'median_sampling_ms':median,'reduction_vs_current_pct':gain,'point_visits':visits})
                print(f'  {name:22s} {median:.4f} ms ({gain:+.1f}%), visits={visits:.0f}',flush=True)
            for filename,records in [('trials.csv',allrows),('summary.csv',summaries)]:
                with (OUT/filename).open('w',newline='',encoding='utf-8') as f:
                    writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
            print(flush=True)
    print('Done! Results:',OUT,'\nNative only; these are NOT gem5 timings.')

if __name__=='__main__':main()
