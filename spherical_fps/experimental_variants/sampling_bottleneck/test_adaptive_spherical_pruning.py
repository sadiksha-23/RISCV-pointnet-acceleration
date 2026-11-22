#!/usr/bin/env python3
"""Test static adaptive spherical-region subdivision with AABB-first pruning.

Copy into spherical_fps/experimental_variants/sampling_bottleneck/ and run:
  python test_adaptive_spherical_pruning.py

Uses the generated C++ source from the previous gem5_three_variants experiment:
  results/gem5_three_variants/sources/combined_aabb_first.cpp

Each large spherical region is recursively subdivided along its widest Cartesian
axis at the median point. Each leaf gets its own conservative AABB, sphere,
and radial bounds. The exact FPS sampling loop is unchanged. We compare
full native preparation, sampling, point visits, region checks, and indices.
This is NOT a gem5 benchmark and results do not establish a RISC-V speedup.
"""
import argparse
import csv
import os
import random
import re
import shutil
import statistics
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next((p for p in (HERE, *HERE.parents)
             if (p / 'spherical_fps/main_implementation/spherical_fps.cpp').is_file()), None)
if ROOT is None:
    raise SystemExit('Place this script under spherical_fps/experimental_variants/sampling_bottleneck in your repository.')
BASE_SOURCE = HERE / 'results/gem5_three_variants/sources/combined_aabb_first.cpp'
DEFAULT_DATASET = ROOT / 'datasets/kitti/0000000000.bin'
RESULTS = HERE / 'results/adaptive_spherical_pruning'

ADAPTIVE_CPP = r'''
// Experimental adaptive subdivision: does not alter exact FPS sampling.
// Every leaf covers a subset of one original spherical region.
static Region make_leaf_region(const std::vector<float>& xyz, const double origin[3], std::vector<int>&& members) {
    Region g;
    g.members = std::move(members);
    for (int i : g.members) {
        const size_t p=static_cast<size_t>(i)*3;
        const double x=static_cast<double>(xyz[p]);
        const double y=static_cast<double>(xyz[p+1]);
        const double z=static_cast<double>(xyz[p+2]);
        g.cx+=x; g.cy+=y; g.cz+=z;
        g.xmin=std::min(g.xmin,x); g.xmax=std::max(g.xmax,x);
        g.ymin=std::min(g.ymin,y); g.ymax=std::max(g.ymax,y);
        g.zmin=std::min(g.zmin,z); g.zmax=std::max(g.zmax,z);
        const double rx=x-origin[0], ry=y-origin[1], rz=z-origin[2];
        const double rsq=rx*rx+ry*ry+rz*rz;
        g.radial_min_sq=std::min(g.radial_min_sq,rsq);
        g.radial_max_sq=std::max(g.radial_max_sq,rsq);
    }
    const double count=static_cast<double>(g.members.size());
    g.cx/=count; g.cy/=count; g.cz/=count;
    g.radial_min=std::sqrt(g.radial_min_sq);
    g.radial_max=std::sqrt(g.radial_max_sq);
    for (int i : g.members) {
        const size_t p=static_cast<size_t>(i)*3;
        const double dx=static_cast<double>(xyz[p])-g.cx;
        const double dy=static_cast<double>(xyz[p+1])-g.cy;
        const double dz=static_cast<double>(xyz[p+2])-g.cz;
        g.radius=std::max(g.radius,std::sqrt(dx*dx+dy*dy+dz*dz));
    }
    // Same conservative padding as original F2.
    g.radius+=1e-7*(1.0+g.radius);
    g.radial_min=std::max(0.0,g.radial_min-1e-7*(1.0+g.radial_min));
    g.radial_max+=1e-7*(1.0+g.radial_max);
    g.best_index=g.members.front();
    return g;
}
static void split_spherical_region(const std::vector<float>& xyz, const double origin[3],
                                   std::vector<int>&& members, int max_points,
                                   std::vector<Region>& result) {
    if (static_cast<int>(members.size()) <= max_points) {
        result.push_back(make_leaf_region(xyz,origin,std::move(members)));
        return;
    }
    double lo[3]={std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity()};
    double hi[3]={-std::numeric_limits<double>::infinity(),-std::numeric_limits<double>::infinity(),-std::numeric_limits<double>::infinity()};
    for (int i:members) for (int d=0;d<3;++d) {
        const double v=static_cast<double>(xyz[static_cast<size_t>(i)*3+d]);
        lo[d]=std::min(lo[d],v); hi[d]=std::max(hi[d],v);
    }
    int axis=0;
    if (hi[1]-lo[1]>hi[axis]-lo[axis]) axis=1;
    if (hi[2]-lo[2]>hi[axis]-lo[axis]) axis=2;
    auto mid=members.begin()+members.size()/2;
    std::nth_element(members.begin(),mid,members.end(),[&](int a,int b) {
        const float av=xyz[static_cast<size_t>(a)*3+axis];
        const float bv=xyz[static_cast<size_t>(b)*3+axis];
        return av<bv || (av==bv && a<b);
    });
    std::vector<int> left(members.begin(),mid);
    std::vector<int> right(mid,members.end());
    split_spherical_region(xyz,origin,std::move(left),max_points,result);
    split_spherical_region(xyz,origin,std::move(right),max_points,result);
}
static std::vector<Region> subdivide_spherical_regions(const std::vector<float>& xyz,
                                                        const double origin[3],
                                                        std::vector<Region>&& original,
                                                        int max_points) {
    std::vector<Region> out;
    out.reserve(original.size());
    for (Region& g:original) {
        if (static_cast<int>(g.members.size())<=max_points) {
            out.push_back(std::move(g));
        } else {
            split_spherical_region(xyz,origin,std::move(g.members),max_points,out);
        }
    }
    return out;
}
'''

ANCHOR = 'std::vector<Region> regions=build_regions(xyz,n,origin);'
MAIN_ANCHOR = 'int main(int argc,char** argv){'


def generate(source: str, threshold: int) -> str:
    if source.count(ANCHOR)!=1 or source.count(MAIN_ANCHOR)!=1:
        raise RuntimeError('Source structure changed. Expected exactly one region-build and main() anchor.')
    if 'aabb_can_skip_early' not in source or 'f2_elevation_above' not in source:
        raise RuntimeError('Source is not the combined+AABB-first implementation expected.')
    source=source.replace(MAIN_ANCHOR,ADAPTIVE_CPP+'\n'+MAIN_ANCHOR,1)
    source=source.replace(ANCHOR,ANCHOR+'\n        regions=subdivide_spherical_regions(xyz,origin,std::move(regions),'+str(threshold)+');',1)
    return source


def execute(cmd):
    p=subprocess.run([str(x) for x in cmd],text=True,capture_output=True,errors='replace')
    if p.returncode:
        raise RuntimeError(f"Command failed: {' '.join(map(str,cmd))}\n{p.stderr[-4000:]}\n{p.stdout[-4000:]}")
    return p.stdout


def parse(output, m):
    def val(key):
        a=re.search(r'^'+re.escape(key)+r'=([^\r\n]+)',output,re.M)
        if not a: raise RuntimeError('Missing '+key)
        return a.group(1)
    indices=[(int(a),int(b)) for a,b in re.findall(r'^FPS_INDEX\[(\d+)\]=(\d+)',output,re.M)]
    if [i for i,_ in indices] != list(range(m)):
        raise RuntimeError(f'Expected {m} consecutive FPS_INDEX lines; got {len(indices)}')
    return {'indices':[v for _,v in indices],
            'prep_ms':float(val('PREPARATION_MS')),
            'sampling_ms':float(val('SAMPLING_MS')),
            'total_ms':float(val('TOTAL_MS')),
            'regions':int(val('OCCUPIED_REGIONS')),
            'region_tests':int(val('REGION_TESTS')),
            'regions_skipped':int(val('REGIONS_SKIPPED')),
            'point_visits':int(val('POINT_DISTANCE_EVALS'))}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,default=DEFAULT_DATASET)
    p.add_argument('--n',type=int,nargs='+',default=[50000,100000])
    p.add_argument('--m',type=int,nargs='+',default=[128,512,1024])
    p.add_argument('--max-points',type=int,nargs='+',default=[256,512,1024,2048],help='Maximum points in an adaptively split region')
    p.add_argument('--trials',type=int,default=5)
    p.add_argument('--compiler',default='g++')
    args=p.parse_args()
    if not BASE_SOURCE.is_file(): p.error('Missing previous combined+AABB-first source: '+str(BASE_SOURCE))
    if not args.dataset.is_file(): p.error('Missing dataset: '+str(args.dataset))
    if not shutil.which(args.compiler): p.error('Compiler not found: '+args.compiler)
    if args.trials<1 or min(args.n)<1 or min(args.m)<1 or max(args.m)>min(args.n) or min(args.max_points)<2:
        p.error('Invalid trial, N, M, or max-points values.')

    RESULTS.mkdir(parents=True,exist_ok=True)
    baseline=BASE_SOURCE.read_text(encoding='utf-8')
    generated={'current_combined_aabb':baseline}
    generated.update({f'split_{k}':generate(baseline,k) for k in sorted(set(args.max_points))})
    binaries={}
    for name,source in generated.items():
        cpp=RESULTS/(name+'.cpp')
        exe=RESULTS/(name+('.exe' if os.name=='nt' else ''))
        cpp.write_text(source,encoding='utf-8')
        print('Compiling',name,flush=True)
        execute([args.compiler,'-O3','-std=c++17','-ffp-contract=off',str(cpp),'-o',str(exe)])
        binaries[name]=exe

    rows=[];summaries=[]
    for n in args.n:
        for m in args.m:
            for trial in range(1,args.trials+1):
                order=list(binaries)
                random.Random(n*10000+m*10+trial).shuffle(order)
                results={}
                for name in order:
                    out=execute([binaries[name],args.dataset,n,m])
                    results[name]=parse(out,m)
                ref=results['current_combined_aabb']['indices']
                for name,r in results.items():
                    if r['indices']!=ref:
                        idx=next(i for i,(x,y) in enumerate(zip(ref,r['indices'])) if x!=y)
                        raise RuntimeError(f'INDEX MISMATCH at N={n} M={m} trial={trial} {name} step={idx}: {ref[idx]} vs {r["indices"][idx]}')
                    rows.append({'N':n,'M':m,'trial':trial,'variant':name,
                                 **{key:r[key] for key in ('prep_ms','sampling_ms','total_ms','regions','region_tests','regions_skipped','point_visits')},
                                 'indices_match':'YES'})
                print(f'N={n} M={m} trial={trial}/{args.trials}: indices MATCH | '+
                      ' '.join(f'{name}={results[name]["sampling_ms"]:.3f}ms' for name in generated),flush=True)
            print(f'\nN={n}, M={m} — median native times (ms):',flush=True)
            reference=statistics.median(r['sampling_ms'] for r in rows if r['N']==n and r['M']==m and r['variant']=='current_combined_aabb')
            for name in generated:
                a=[r for r in rows if r['N']==n and r['M']==m and r['variant']==name]
                def median(key): return statistics.median(x[key] for x in a)
                s={'N':n,'M':m,'variant':name,'median_prep_ms':median('prep_ms'),
                   'median_sampling_ms':median('sampling_ms'),'median_total_ms':median('total_ms'),
                   'sampling_reduction_pct':100*(reference-median('sampling_ms'))/reference,
                   'occupied_regions':median('regions'),'region_tests':median('region_tests'),
                   'point_visits':median('point_visits')}
                summaries.append(s)
                print(f'  {name:22s} prep={s["median_prep_ms"]:7.3f} '
                      f'sampling={s["median_sampling_ms"]:7.3f} total={s["median_total_ms"]:7.3f} '
                      f'sampling gain={s["sampling_reduction_pct"]:+6.1f}% '
                      f'leaves={s["occupied_regions"]:.0f} point visits={s["point_visits"]:.0f}',flush=True)
            for name,data in [('trials.csv',rows),('summary.csv',summaries)]:
                with (RESULTS/name).open('w',newline='',encoding='utf-8') as f:
                    w=csv.DictWriter(f,fieldnames=data[0].keys());w.writeheader();w.writerows(data)
    print('\nComplete:',RESULTS,flush=True)
    print('These are native timings only; successful FPS index matches are relative to current combined+AABB-first, not an independent BasicFPS check.')


if __name__=='__main__':main()
