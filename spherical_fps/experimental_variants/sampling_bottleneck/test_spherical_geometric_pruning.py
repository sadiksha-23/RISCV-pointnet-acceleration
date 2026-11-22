#!/usr/bin/env python3
"""Native FPS benchmark. Standard library only. Run in VS Code."""
from pathlib import Path
import argparse, csv, random, re, statistics, subprocess

HERE = Path(__file__).resolve().parent
META = r'''
#ifndef GEOM_MODE
#define GEOM_MODE 0
#endif
// 1=radial, 2=cone, 3=radial+cone, 4=angular point bound.
struct Cone {double x=0,y=0,z=1,half=0,cos_half=1,sin_half=0;};
struct Geometry {std::vector<double> r; std::vector<Cone> cones;};
static Geometry prepare_geometry(const std::vector<float>& xyz,int n,const double origin[3],const std::vector<Region>& regions){
    Geometry h;
    if constexpr(GEOM_MODE!=0){
        h.r.resize(n);
        for(int i=0;i<n;++i){size_t p=3ULL*i;double x=xyz[p]-origin[0],y=xyz[p+1]-origin[1],z=xyz[p+2]-origin[2];h.r[i]=std::sqrt(x*x+y*y+z*z);}
    }
    if constexpr(GEOM_MODE>=2){
        h.cones.resize(regions.size());
        for(size_t j=0;j<regions.size();++j){const auto& g=regions[j];auto& a=h.cones[j];
            double x=g.cx-origin[0],y=g.cy-origin[1],z=g.cz-origin[2];double norm=std::sqrt(x*x+y*y+z*z);
            if(norm>0){a.x=x/norm;a.y=y/norm;a.z=z/norm;}
            for(int i:g.members){size_t p=3ULL*i;double r=h.r[i];if(r==0){a.half=PI;break;}
                double dot=(a.x*(xyz[p]-origin[0])+a.y*(xyz[p+1]-origin[1])+a.z*(xyz[p+2]-origin[2]))/r;
                a.half=std::max(a.half,std::acos(std::clamp(dot,-1.0,1.0)));}
            a.half=std::min(PI,a.half+1e-7);a.cos_half=std::cos(a.half);a.sin_half=std::sin(a.half);
        }
    }
    return h;
}
// cos(max(0, separation-half_angle)), with conservative padding.
// No acos, sin or cos during sampling.
static inline double cone_cos_upper(const Cone& a,double x,double y,double z,double sr){
    if(sr==0)return 1;
    double q=std::clamp((a.x*x+a.y*y+a.z*z)/sr+1e-12,-1.0,1.0);
    if(q>=a.cos_half)return 1;
    return std::min(1.0,q*a.cos_half+std::sqrt(std::max(0.0,1-q*q))*a.sin_half+1e-7);
}
static inline bool geo_reject(double lb,double old){
    return std::max(0.0,lb-1e-5*(1+std::abs(lb)))>old+1e-5*(1+std::abs(old));
}
static inline double cone_lb(double lo,double hi,double sr,double cu){
    double r=std::clamp(sr*cu,lo,hi),dr=r-sr;
    return dr*dr+2*r*sr*(1-cu);
}
static uint64_t point_visits=0,point_rejections=0,cone_skips=0,cone_points=0;
'''

def once(s,a,b):
    if s.count(a)!=1: raise RuntimeError(f'Expected one source anchor: {a!r}; found {s.count(a)}')
    return s.replace(a,b,1)

def generate(s):
    anchor='static std::vector<int> spherical_fps('
    s=once(s,anchor,META+'\n'+anchor)
    s=once(s,'std::vector<Region>& regions,Counters& c){','std::vector<Region>& regions,Counters& c,const Geometry& geom){')
    s=once(s,'const double selected_radius=std::sqrt(ox*ox+oy*oy+oz*oz);','const double selected_radius=GEOM_MODE ? geom.r[newest] : std::sqrt(ox*ox+oy*oy+oz*oz);')
    s=once(s,'float best=-1;int best_i=-1;',r'''double cu=1;
            if constexpr(GEOM_MODE>=2){
                cu=cone_cos_upper(geom.cones[&g-regions.data()],ox,oy,oz,selected_radius);
                if constexpr(GEOM_MODE==2 || GEOM_MODE==3){
                    if(geo_reject(cone_lb(g.radial_min,g.radial_max,selected_radius,cu),g.max_nearest_sq)){
                        ++c.skipped_regions;c.skipped_points+=g.members.size();++cone_skips;cone_points+=g.members.size();continue;
                    }
                }
            }
            point_visits+=g.members.size();
            float best=-1;int best_i=-1;''')
    s=once(s,'const float d=sqdist(xyz,i,newest);++c.evaluated_distances;\n                if(d<nearest[i])nearest[i]=d;',r'''bool reject=false;
                if constexpr(GEOM_MODE==1 || GEOM_MODE==3 || GEOM_MODE==4){
                    double r=geom.r[i],dr=r-selected_radius,lb=dr*dr;
                    if constexpr(GEOM_MODE==4)lb+=2*r*selected_radius*(1-cu);
                    reject=geo_reject(lb,nearest[i]);
                }
                if(!reject){const float d=sqdist(xyz,i,newest);++c.evaluated_distances;if(d<nearest[i])nearest[i]=d;}
                else ++point_rejections;''')
    s=once(s,'const auto end_prep=std::chrono::steady_clock::now();','const Geometry geom=prepare_geometry(xyz,n,origin,regions);\n        const auto end_prep=std::chrono::steady_clock::now();')
    s=once(s,'spherical_fps(xyz,n,m,origin,regions,counters)','spherical_fps(xyz,n,m,origin,regions,counters,geom)')
    s=once(s,'for(int i=0;i<m;++i)std::printf',r'''std::printf("POINT_VISITS=%llu\nPOINT_REJECTIONS=%llu\nCONE_SKIPS=%llu\nCONE_POINTS=%llu\n",(unsigned long long)point_visits,(unsigned long long)point_rejections,(unsigned long long)cone_skips,(unsigned long long)cone_points);
        for(int i=0;i<m;++i)std::printf''')
    return s

def main():
    root=next((p for p in (HERE,*HERE.parents) if (p/'datasets/kitti/0000000000.bin').exists()),None)
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,default=HERE/'results/gem5_three_variants/sources/combined_aabb_first.cpp');ap.add_argument('--dataset',type=Path,default=root/'datasets/kitti/0000000000.bin' if root else None);ap.add_argument('--out',type=Path,default=HERE/'results/spherical_geometric_pruning');ap.add_argument('--compiler',default='g++');ap.add_argument('--trials',type=int,default=5);ap.add_argument('--case',action='append',help='N,M; repeatable');a=ap.parse_args()
    if not a.source.is_file() or not a.dataset or not a.dataset.is_file():ap.error('Missing source or dataset; supply --source and --dataset.')
    if a.trials<1:ap.error('Trials must be positive')
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    source=a.source.read_text(encoding='utf-8');cpp=out/'geometric_pruning.cpp';cpp.write_text(generate(source),encoding='utf-8')
    modes=['current_combined_aabb','radial_only','cone_only','radial_cone','radial_angular'];bins={}
    for mode,name in enumerate(modes):
        exe=out/(name+'.exe');print('Compiling',name,flush=True);subprocess.run([a.compiler,'-O3','-std=c++17','-ffp-contract=off',f'-DGEOM_MODE={mode}',str(cpp),'-o',str(exe)],check=True);bins[name]=exe
    cases=[tuple(map(int,x.split(','))) for x in a.case] if a.case else [(16000,2048),(50000,1024),(100000,128),(100000,512),(100000,1024),(100000,4096)]
    rows=[];summary=[]
    def save(path,records):
        with path.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    def run(name,n,m):
        t=subprocess.check_output([str(bins[name]),str(a.dataset.resolve()),str(n),str(m)],text=True)
        idx=re.findall(r'^FPS_INDEX\[\d+\]=(\d+)$',t,re.M); vals=dict(re.findall(r'^([A-Z_]+)=([0-9.eE+\-]+)$',t,re.M))
        if len(idx)!=m:raise RuntimeError('Missing FPS indices')
        return idx,{k:float(vals[k]) for k in ['PREPARATION_MS','SAMPLING_MS','TOTAL_MS','POINT_VISITS','POINT_DISTANCE_EVALS','POINT_REJECTIONS','CONE_SKIPS','CONE_POINTS','REGIONS_SKIPPED']}
    for n,m in cases:
        if not 1<=m<=n<=a.dataset.stat().st_size//16:print('Skipping invalid case',n,m);continue
        print(f'\nSTART N={n:,} M={m:,}',flush=True)
        for trial in range(1,a.trials+1):
            order=modes.copy();random.Random(n*10000+m+trial).shuffle(order);results={name:run(name,n,m) for name in order};ref=results[modes[0]][0]
            for name in modes:
                idx,v=results[name]
                if idx!=ref:raise RuntimeError(f'FPS mismatch {name} N={n} M={m}')
                if v['POINT_VISITS']!=v['POINT_DISTANCE_EVALS']+v['POINT_REJECTIONS']:raise RuntimeError('Point accounting mismatch')
                rows.append(dict(N=n,M=m,trial=trial,variant=name,**v))
            print(f'Trial {trial}/{a.trials}: indices MATCH | '+' '.join(f'{name}={results[name][1]["SAMPLING_MS"]:.3f}ms' for name in modes),flush=True);save(out/'trials.csv',rows)
        for name in modes:
            subset=[r for r in rows if r['N']==n and r['M']==m and r['variant']==name];med={k:statistics.median(r[k] for r in subset) for k in results[name][1]}
            if name==modes[0]:base=med
            gain=100*(1-med['SAMPLING_MS']/base['SAMPLING_MS']) if base['SAMPLING_MS'] else 0
            summary.append(dict(N=n,M=m,variant=name,**med,sampling_gain_pct=gain))
            print(f'{name:23s} prep={med["PREPARATION_MS"]:.3f} sampling={med["SAMPLING_MS"]:.3f} total={med["TOTAL_MS"]:.3f} gain={gain:+.1f}% visits={med["POINT_VISITS"]:.0f} full_dist={med["POINT_DISTANCE_EVALS"]:.0f} point_reject={med["POINT_REJECTIONS"]:.0f} cone_skips={med["CONE_SKIPS"]:.0f}',flush=True)
        save(out/'summary.csv',summary)
    print('\nResults:',out,'\nNative only; matching indices against baseline.')
if __name__=='__main__':main()
