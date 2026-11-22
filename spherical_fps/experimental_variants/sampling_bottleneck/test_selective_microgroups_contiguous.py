#!/usr/bin/env python3
"""Native exact FPS benchmark: selective microgroups and contiguous region storage.
Replace your existing test_adaptive_spherical_pruning.py with this file.
Original C++ remains unchanged. Default paired cases: 16k/2048, 50k/1024,
100k/128, 100k/512, 100k/1024, 100k/4096. Five shuffled trials each.
"""
from __future__ import annotations
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
    raise SystemExit('Put this script in spherical_fps/experimental_variants/sampling_bottleneck/')
BASE = HERE / 'results/gem5_three_variants/sources/combined_aabb_first.cpp'
DATA = ROOT / 'datasets/kitti/0000000000.bin'
OUT = HERE / 'results/selective_microgroups_contiguous'

MICRO_STRUCT = r'''
struct MicroRegion {
    int begin=0,end=0;
    double xmin=std::numeric_limits<double>::infinity();
    double ymin=std::numeric_limits<double>::infinity();
    double zmin=std::numeric_limits<double>::infinity();
    double xmax=-std::numeric_limits<double>::infinity();
    double ymax=-std::numeric_limits<double>::infinity();
    double zmax=-std::numeric_limits<double>::infinity();
    float max_nearest_sq=1e38f;
    int best_index=-1;
};
static std::vector<std::vector<MicroRegion>> make_microregions(
    const std::vector<float>& xyz,const std::vector<Region>& regions,int block_size,int threshold) {
    std::vector<std::vector<MicroRegion>> result;
    result.reserve(regions.size());
    for(const Region& g:regions) {
        std::vector<MicroRegion> parts;
        const int count=static_cast<int>(g.members.size());
        if(count<threshold || block_size==0) { result.push_back(std::move(parts)); continue; }
        parts.reserve(static_cast<size_t>((count+block_size-1)/block_size));
        for(int begin=0;begin<count;begin+=block_size) {
            MicroRegion c;
            c.begin=begin;
            c.end=std::min(count,begin+block_size);
            c.best_index=g.members[begin];
            for(int k=c.begin;k<c.end;++k) {
                const size_t p=static_cast<size_t>(g.members[k])*3;
                const double x=xyz[p],y=xyz[p+1],z=xyz[p+2];
                c.xmin=std::min(c.xmin,x);c.xmax=std::max(c.xmax,x);
                c.ymin=std::min(c.ymin,y);c.ymax=std::max(c.ymax,y);
                c.zmin=std::min(c.zmin,z);c.zmax=std::max(c.zmax,z);
            }
            parts.push_back(c);
        }
        result.push_back(std::move(parts));
    }
    return result;
}
static bool skip_microregion(const MicroRegion& g,const std::vector<float>& xyz,
                             int newest,float current_max_sq) {
    const size_t p=static_cast<size_t>(newest)*3;
    const double x=xyz[p],y=xyz[p+1],z=xyz[p+2];
    const double bx=std::max({g.xmin-x,0.0,x-g.xmax});
    const double by=std::max({g.ymin-y,0.0,y-g.ymax});
    const double bz=std::max({g.zmin-z,0.0,z-g.zmax});
    const double raw_lb=std::sqrt(bx*bx+by*by+bz*bz);
    const double lb=std::max(0.0,raw_lb-1e-6*(1.0+raw_lb));
    const double max_sq=static_cast<double>(current_max_sq);
    const double margin=1e-5*(1.0+std::abs(max_sq));
    return lb*lb > max_sq+margin;
}
'''

# The original parent-region pruning is kept *identical* (including its
# AABB-first fast path and sphere/radial fallback). Only the inner point loop
# changes: first check the already available block score against a block AABB.
PACK_STRUCT = r'''
struct PackedPoints {
    std::vector<float> x,y,z;
    std::vector<size_t> offsets;
};
static PackedPoints pack_regions(const std::vector<float>& xyz,const std::vector<Region>& regions,int n) {
    PackedPoints out;
    out.x.reserve(n);out.y.reserve(n);out.z.reserve(n);
    for(const Region& g:regions) {
        out.offsets.push_back(out.x.size());
        for(int i:g.members) {
            size_t p=static_cast<size_t>(i)*3;
            out.x.push_back(xyz[p]);out.y.push_back(xyz[p+1]);out.z.push_back(xyz[p+2]);
        }
    }
    return out;
}
'''
MICRO_SAMPLING = r''' 
static inline float packed_dist(const PackedPoints& a,size_t ni,const std::vector<float>& xyz,size_t p) {
    const float dx=a.x[ni]-xyz[p],dy=a.y[ni]-xyz[p+1],dz=a.z[ni]-xyz[p+2];
    return dx*dx+dy*dy+dz*dz;
}
static std::vector<int> spherical_fps(const std::vector<float>& xyz,int n,int m,
    const double origin[3],std::vector<Region>& regions,
    std::vector<std::vector<MicroRegion>>& microregions,const PackedPoints& packed,Counters& c) {
    std::vector<float> nearest(static_cast<size_t>(n),1e38f);
    std::vector<int> selected(static_cast<size_t>(m));selected[0]=0;
    c.occupied_regions=static_cast<int>(regions.size());
    for(int step=1;step<m;++step) {
        const int newest=selected[step-1];
        const size_t p=static_cast<size_t>(newest)*3;
        const double ox=static_cast<double>(xyz[p])-origin[0];
        const double oy=static_cast<double>(xyz[p+1])-origin[1];
        const double oz=static_cast<double>(xyz[p+2])-origin[2];
        const double selected_radius=std::sqrt(ox*ox+oy*oy+oz*oz);
        for(size_t r=0;r<regions.size();++r) {
            Region& g=regions[r];++c.checked_regions;
            if(aabb_can_skip_early(g,xyz,newest,g.max_nearest_sq)) {
                ++c.skipped_regions;c.skipped_points+=g.members.size();continue;
            }
            const double lb=lower_bound(g,xyz,newest,selected_radius);
            const double max_sq=static_cast<double>(g.max_nearest_sq);
            const double margin=1e-5*(1.0+std::abs(max_sq));
            if(lb*lb>max_sq+margin) {
                ++c.skipped_regions;c.skipped_points+=g.members.size();continue;
            }
            float best=-1.0f;int best_i=-1;
            if(microregions[r].empty()) {
                for(int k=0;k<static_cast<int>(g.members.size());++k) {
                    const int i=g.members[k];
                    const size_t ni=PACKED ? packed.offsets[r]+k : static_cast<size_t>(i);
                    const float d=PACKED ? packed_dist(packed,ni,xyz,p) : sqdist(xyz,i,newest);
                    ++c.evaluated_distances;
                    if(d<nearest[ni])nearest[ni]=d;
                    if(nearest[ni]>best || (nearest[ni]==best && (best_i<0 || i<best_i))) {best=nearest[ni];best_i=i;}
                }
            }
            for(MicroRegion& part:microregions[r]) {
                ++c.micro_checks;
                if(skip_microregion(part,xyz,newest,part.max_nearest_sq)) {
                    ++c.micro_skips;
                    c.micro_skipped_points+=static_cast<uint64_t>(part.end-part.begin);
                } else {
                    float local_best=-1.0f;int local_i=-1;
                    for(int k=part.begin;k<part.end;++k) {
                        const int i=g.members[k];
                        const size_t ni=PACKED ? packed.offsets[r]+k : static_cast<size_t>(i);
                        const float d=PACKED ? packed_dist(packed,ni,xyz,p) : sqdist(xyz,i,newest);
                        ++c.evaluated_distances;
                        if(d<nearest[ni])nearest[ni]=d;
                        if(nearest[ni]>local_best ||
                          (nearest[ni]==local_best && (local_i<0 || i<local_i))) {
                            local_best=nearest[ni];local_i=i;
                        }
                    }
                    part.max_nearest_sq=local_best;
                    part.best_index=local_i;
                }
                // Cached block maximum remains valid when skipped:
                // adding a selected point did not decrease ANY point's distance.
                if(part.max_nearest_sq>best ||
                  (part.max_nearest_sq==best && (best_i<0 || part.best_index<best_i))) {
                    best=part.max_nearest_sq;best_i=part.best_index;
                }
            }
            g.max_nearest_sq=best;g.best_index=best_i;
        }
        float global_best=-1.0f;int global_i=-1;
        for(const Region& g:regions) {
            if(g.max_nearest_sq>global_best ||
              (g.max_nearest_sq==global_best && (global_i<0 || g.best_index<global_i))) {
                global_best=g.max_nearest_sq;global_i=g.best_index;
            }
        }
        if(global_i<0)throw std::runtime_error("No FPS winner");
        selected[step]=global_i;
    }
    return selected;
}
'''


def replace_once(text: str, old: str, new: str, description: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f'{description}: expected one exact anchor, found {count}: {old[:90]}')
    return text.replace(old, new, 1)


def make_microvariant(base: str, block_size: int, threshold: int, packed: bool) -> str:
    if 'aabb_can_skip_early' not in base:
        raise RuntimeError('Source must be the generated Combined + AABB-first C++.')
    if base.count('static std::vector<int> spherical_fps(') != 1:
        raise RuntimeError('Unexpected spherical_fps source layout.')
    cpp = replace_once(base, 'struct Counters {',
                       'struct Counters {uint64_t micro_checks=0,micro_skips=0,micro_skipped_points=0;',
                       'Counters extension')
    start=cpp.index('static std::vector<int> spherical_fps(')
    end=cpp.index('\nint main(',start)
    cpp=cpp[:start]+("static constexpr bool PACKED="+str(packed).lower()+";\n")+MICRO_STRUCT+PACK_STRUCT+MICRO_SAMPLING+cpp[end:]
    cpp=replace_once(cpp, 'const auto end_prep=std::chrono::steady_clock::now();',
        f'auto microregions=make_microregions(xyz,regions,{block_size},{threshold});\n'
        '        auto packed=PACKED ? pack_regions(xyz,regions,n) : PackedPoints{};\n'
        '        const auto end_prep=std::chrono::steady_clock::now();',
        'Move microregion construction into PREPARATION')
    old='spherical_fps(xyz,n,m,origin,regions,counters)'
    new='spherical_fps(xyz,n,m,origin,regions,microregions,packed,counters)'
    cpp=replace_once(cpp,old,new,'Pass microregion metadata')
    marker='std::printf("CORRECTNESS=NOT_CHECKED (performance-only run)\\n");'
    metrics=('std::printf("MICRO_CHECKS=%llu\\nMICRO_SKIPS=%llu\\nMICRO_SKIPPED_POINTS=%llu\\n",'
             '(unsigned long long)counters.micro_checks,'
             '(unsigned long long)counters.micro_skips,'
             '(unsigned long long)counters.micro_skipped_points);\n        ')
    cpp=replace_once(cpp,marker,metrics+marker,'Print microregion metrics')
    return cpp


def run(cmd: list[str]) -> str:
    p=subprocess.run([str(x) for x in cmd],capture_output=True,text=True,errors='replace')
    if p.returncode:
        raise RuntimeError('Command failed: '+ ' '.join(map(str,cmd))+'\n'+p.stderr[-4500:]+'\n'+p.stdout[-1200:])
    return p.stdout


def parse(output: str,m: int) -> dict:
    fields=dict(re.findall(r'^([A-Z][A-Z0-9_]+)=([^\r\n]+)',output,re.M))
    ids=[(int(i),int(v)) for i,v in re.findall(r'^FPS_INDEX\[(\d+)\]=(\d+)',output,re.M)]
    if len(ids)!=m or [i for i,_ in ids]!=list(range(m)):
        raise RuntimeError('Missing or misnumbered FPS_INDEX entries')
    return dict(indices=[v for _,v in ids],prep=float(fields['PREPARATION_MS']),
                sampling=float(fields['SAMPLING_MS']),total=float(fields['TOTAL_MS']),
                visits=int(fields['POINT_DISTANCE_EVALS']),
                micro_checks=int(fields.get('MICRO_CHECKS','0')),
                micro_skips=int(fields.get('MICRO_SKIPS','0')),
                micro_skipped_points=int(fields.get('MICRO_SKIPPED_POINTS','0')))


def available_kitti_points(path: Path) -> int:
    if not path.is_file():
        return 0
    size=path.stat().st_size
    if size % 16:
        raise RuntimeError(f'Invalid XYZI binary size for {path}: {size} bytes is not divisible by 16')
    return size // 16


def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset',type=Path,default=DATA,
                    help='KITTI XYZI binary for the 16k and 100k cases')
    ap.add_argument('--trials',type=int,default=None,
                    help='Override trials for all cases (default: 5)')
    ap.add_argument('--compiler',default='g++')
    ap.add_argument('--strict',action='store_true',
                    help='Fail if a case lacks enough input points, instead of skipping it')
    args=ap.parse_args()
    if not BASE.is_file():
        ap.error(f'Missing {BASE}. Run the earlier gem5-three-variants generator first.')
    if not args.dataset.is_file():
        ap.error(f'Missing KITTI file: {args.dataset}')
    if not shutil.which(args.compiler):
        ap.error(f'Compiler unavailable: {args.compiler}')
    if args.trials is not None and args.trials<1:
        ap.error('Invalid block sizes or trials')

    requested_cases=[(16000,2048,5),(50000,1024,5),(100000,128,5),(100000,512,5),(100000,1024,5),(100000,4096,5)]
    cases=[]
    for n,m,default_trials in requested_cases:
        dataset=args.dataset
        available=available_kitti_points(dataset)
        if available<n:
            msg=(f'N={n}, M={m}: input {dataset} contains only {available:,} points '
                 f'(needs {n:,}).')
            if args.strict:
                ap.error(msg)
            print('SKIPPING '+msg,flush=True)
            continue
        cases.append((n,m,args.trials or default_trials,dataset))
    if not cases:
        ap.error('No runnable cases: all requested N values exceed input point counts')

    OUT.mkdir(parents=True,exist_ok=True)
    base=BASE.read_text(encoding='utf-8')
    sources={'current_combined_aabb':base}
    sources['contiguous_only']=make_microvariant(base,0,0,True)
    for threshold in (512,1024,2048):
        sources[f'selective128_min{threshold}']=make_microvariant(base,128,threshold,False)
        sources[f'packed128_min{threshold}']=make_microvariant(base,128,threshold,True)
    sources['micro128_all']=make_microvariant(base,128,0,False)
    binaries={}
    for name,source in sources.items():
        cpp=OUT/f'{name}.cpp'
        exe=OUT/(name+('.exe' if os.name=='nt' else ''))
        cpp.write_text(source,encoding='utf-8')
        print('Compiling',name,flush=True)
        run([args.compiler,'-O3','-std=c++17','-ffp-contract=off',str(cpp),'-o',str(exe)])
        binaries[name]=exe

    rows=[];summary=[]
    for n,m,trials,dataset in cases:
        print(f'\nSTART N={n:,} M={m:,}; {trials} trial(s); dataset={dataset}',flush=True)
        for trial in range(1,trials+1):
            order=list(sources)
            random.Random(100000*n+100*m+trial).shuffle(order)
            results={name:parse(run([binaries[name],dataset,n,m]),m) for name in order}
            baseline=results['current_combined_aabb']['indices']
            for name,result in results.items():
                if result['indices']!=baseline:
                    i=next(k for k,(a,b) in enumerate(zip(result['indices'],baseline)) if a!=b)
                    raise RuntimeError(f'INDEX MISMATCH N={n} M={m} trial={trial} '
                                       f'{name} at position {i}: {result["indices"][i]} vs {baseline[i]}')
                if name=='contiguous_only' and result['visits']!=results['current_combined_aabb']['visits']:
                    raise RuntimeError('Contiguous layout changed point visits')
                rows.append(dict(N=n,M=m,trial=trial,variant=name,dataset=str(dataset),
                                 prep_ms=result['prep'],sampling_ms=result['sampling'],
                                 total_ms=result['total'],point_visits=result['visits'],
                                 micro_checks=result['micro_checks'],micro_skips=result['micro_skips'],
                                 micro_skipped_points=result['micro_skipped_points'],indices_match='YES'))
            print(f'N={n} M={m} trial={trial}/{trials}: indices MATCH | '+
                  ' '.join(f'{name}={results[name]["sampling"]:.3f}ms' for name in sources),flush=True)
            # Save every trial, including large runs, before moving to the next one.
            with (OUT/'trials.csv').open('w',newline='',encoding='utf-8') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]))
                writer.writeheader();writer.writerows(rows)

        base_med=statistics.median(r['sampling_ms'] for r in rows if r['N']==n and r['M']==m
                                   and r['variant']=='current_combined_aabb')
        print(f'\nN={n} M={m} — native medians (ms):',flush=True)
        for name in sources:
            relevant=[r for r in rows if r['N']==n and r['M']==m and r['variant']==name]
            med=lambda key:statistics.median(r[key] for r in relevant)
            speed=(base_med-med('sampling_ms'))/base_med*100
            summary.append(dict(N=n,M=m,variant=name,dataset=str(dataset),trials=trials,
                                median_prep_ms=med('prep_ms'),median_sampling_ms=med('sampling_ms'),
                                median_total_ms=med('total_ms'),sampling_improvement_pct=speed,
                                point_visits=med('point_visits'),micro_checks=med('micro_checks'),
                                micro_skips=med('micro_skips'),
                                micro_skipped_points=med('micro_skipped_points')))
            print(f'  {name:22s} prep={med("prep_ms"):.3f} '
                  f'sampling={med("sampling_ms"):.3f} total={med("total_ms"):.3f} '
                  f'gain={speed:+.1f}% visits={med("point_visits"):.0f} '
                  f'micro_skips={med("micro_skips"):.0f}',flush=True)
        with (OUT/'summary.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(summary[0]))
            writer.writeheader();writer.writerows(summary)

    print('\nComplete:',OUT,flush=True)
    print('Native only. Correctness is checked against Combined + AABB-first.',flush=True)
    if len(cases)!=len(requested_cases):
        print('Some cases were skipped due to insufficient points; see messages above.',flush=True)

if __name__=='__main__':
    main()
