#!/usr/bin/env python3
"""Diagnostic, not a speed benchmark. Run directly in VS Code (stdlib only)."""
from pathlib import Path
import argparse, subprocess, shutil, re, csv

HERE = Path(__file__).resolve().parent
CODE = r'''
struct AngularMeta {
    double ux=0,uy=0,uz=1,half=0,rlo=1e300,rhi=0;
};
static double acos_safe(double x){return std::acos(std::clamp(x,-1.0,1.0));}
static double angular_lb_sq(double rlo,double rhi,double s,double cos_upper){
    double r=std::clamp(s*cos_upper,rlo,rhi);
    return (r-s)*(r-s)+2*r*s*(1-cos_upper);
}
static bool reject_sq(double lb,double old){
    double safe=std::max(0.0,lb-1e-5*(1+std::abs(lb)));
    return safe > old+1e-5*(1+std::abs(old));
}
static std::vector<int> spherical_fps(const std::vector<float>& xyz,int n,int m,const double origin[3],std::vector<Region>& regions,Counters& c){
    std::vector<double> r(n),ux(n),uy(n),uz(n);
    for(int i=0;i<n;++i){size_t p=3ULL*i;double x=xyz[p]-origin[0],y=xyz[p+1]-origin[1],z=xyz[p+2]-origin[2];r[i]=std::sqrt(x*x+y*y+z*z);if(r[i]>0){ux[i]=x/r[i];uy[i]=y/r[i];uz[i]=z/r[i];}}
    std::vector<AngularMeta> meta(regions.size());
    double weighted_radial=0,weighted_angle=0,max_angle=0;
    for(size_t j=0;j<regions.size();++j){auto& g=regions[j];auto& a=meta[j];
        double x=g.cx-origin[0],y=g.cy-origin[1],z=g.cz-origin[2];double norm=std::sqrt(x*x+y*y+z*z);
        if(norm>0){a.ux=x/norm;a.uy=y/norm;a.uz=z/norm;}
        for(int i:g.members){a.rlo=std::min(a.rlo,r[i]);a.rhi=std::max(a.rhi,r[i]);if(r[i]==0){a.half=3.141592653589793;continue;}a.half=std::max(a.half,acos_safe(a.ux*ux[i]+a.uy*uy[i]+a.uz*uz[i]));}
        a.half=std::min(3.141592653589793,a.half+1e-7);
        weighted_radial+=(a.rhi-a.rlo)*g.members.size();weighted_angle+=a.half*g.members.size();max_angle=std::max(max_angle,a.half);
        std::printf("ANG_REGION=%zu,%zu,%.9g,%.9g,%.9g\n",j,g.members.size(),a.rlo,a.rhi,a.half*180/3.141592653589793);
    }
    std::vector<float> nearest(n,1e38f);std::vector<int> selected(m);selected[0]=0;c.occupied_regions=regions.size();
    uint64_t active=0,extra_regions=0,extra_region_points=0,point_extra=0,radial_extra=0,angular_only_extra=0,updates=0,violations=0,tighter_aabb=0,tighter_combined=0,angular_dominant=0,nonzero=0,angular_variation_larger=0,variation_tests=0;
    double angular_sum=0,radial_sum=0,angle_span_sum=0,radius_span_sum=0;
    for(int step=1;step<m;++step){int s=selected[step-1];double sr=r[s];size_t p=3ULL*s;
        for(size_t j=0;j<regions.size();++j){auto& g=regions[j];const auto& a=meta[j];++c.checked_regions;
            double lb=lower_bound(g,xyz,s,sr),old=g.max_nearest_sq;
            if(lb*lb>old+1e-5*(1+std::abs(old))){++c.skipped_regions;c.skipped_points+=g.members.size();continue;}
            ++active;
            double sep=sr>0?acos_safe(a.ux*ux[s]+a.uy*uy[s]+a.uz*uz[s]):0;
            double gamma_lo=std::max(0.0,sep-a.half-1e-7),gamma_hi=std::min(3.141592653589793,sep+a.half+1e-7);
            double cu=std::cos(gamma_lo),cl=std::cos(gamma_hi);
            double cone=angular_lb_sq(a.rlo,a.rhi,sr,cu);
            double bx=std::max({g.xmin-xyz[p],0.0,xyz[p]-g.xmax}),by=std::max({g.ymin-xyz[p+1],0.0,xyz[p+1]-g.ymax}),bz=std::max({g.zmin-xyz[p+2],0.0,xyz[p+2]-g.zmax});
            if(cone>bx*bx+by*by+bz*bz+1e-5)++tighter_aabb;
            if(cone>lb*lb+1e-5*(1+lb*lb))++tighter_combined;
            bool skip_region=reject_sq(cone,old);if(skip_region){++extra_regions;extra_region_points+=g.members.size();}
            // Controlled comparison: vary one coordinate over its interval, hold the other at midpoint.
            double rm=(a.rlo+a.rhi)/2,cm=std::cos((gamma_lo+gamma_hi)/2);
            double angspan=2*rm*sr*(cu-cl);
            auto f=[&](double rr){return (rr-sr)*(rr-sr)+2*rr*sr*(1-cm);};
            double radspan=std::max(f(a.rlo),f(a.rhi))-angular_lb_sq(a.rlo,a.rhi,sr,cm);
            angle_span_sum+=angspan;radius_span_sum+=radspan;++variation_tests;if(angspan>radspan)++angular_variation_larger;
            float best=-1;int best_i=-1;
            for(int i:g.members){float d=sqdist(xyz,i,s);++c.evaluated_distances;double prev=nearest[i];
                double rad=(r[i]-sr)*(r[i]-sr);double point_bound=rad+2*r[i]*sr*(1-cu);
                bool radial_reject=reject_sq(rad,prev),point_reject=reject_sq(point_bound,prev);
                if(radial_reject)++radial_extra;
                if(point_reject){++point_extra;if(!radial_reject)++angular_only_extra;}
                if((skip_region||point_reject||radial_reject)&&d<prev)++violations;
                double angular=std::max(0.0,static_cast<double>(d)-rad);angular_sum+=angular;radial_sum+=rad;
                if(d>0){++nonzero;if(angular>rad)++angular_dominant;}
                if(d<nearest[i]){nearest[i]=d;++updates;}
                if(nearest[i]>best||(nearest[i]==best&&(best_i<0||i<best_i))){best=nearest[i];best_i=i;}
            }
            g.max_nearest_sq=best;g.best_index=best_i;
        }
        float best=-1;int idx=-1;for(const auto& g:regions)if(g.max_nearest_sq>best||(g.max_nearest_sq==best&&(idx<0||g.best_index<idx))){best=g.max_nearest_sq;idx=g.best_index;}
        if(idx<0)throw std::runtime_error("No FPS winner");selected[step]=idx;
    }
    #define COUNT(name,v) std::printf(name "=%llu\n",(unsigned long long)(v))
    #define VALUE(name,v) std::printf(name "=%.9g\n",double(v))
    COUNT("ACTIVE_REGION_TESTS",active);COUNT("EXTRA_REGION_SKIPS",extra_regions);COUNT("EXTRA_REGION_POINT_VISITS_AVOIDABLE",extra_region_points);
    COUNT("POINT_BOUND_REJECTIONS",point_extra);COUNT("RADIAL_ONLY_REJECTIONS",radial_extra);COUNT("ADDITIONAL_ANGULAR_POINT_REJECTIONS",angular_only_extra);
    COUNT("ACTUAL_UPDATES",updates);COUNT("UNSAFE_REJECTIONS",violations);COUNT("CONE_TIGHTER_THAN_AABB",tighter_aabb);COUNT("CONE_TIGHTER_THAN_COMBINED",tighter_combined);
    VALUE("WEIGHTED_RADIAL_WIDTH",weighted_radial/n);VALUE("WEIGHTED_CONE_HALF_ANGLE_DEG",weighted_angle/n*180/3.141592653589793);VALUE("MAX_CONE_HALF_ANGLE_DEG",max_angle*180/3.141592653589793);
    VALUE("ANGULAR_SHARE_OF_SQUARED_DISTANCE_PCT",100*angular_sum/std::max(1e-30,angular_sum+radial_sum));VALUE("ANGULAR_DOMINANT_PAIRS_PCT",100.0*angular_dominant/std::max<uint64_t>(1,nonzero));
    VALUE("ANGULAR_VARIATION_DOMINANT_REGION_TESTS_PCT",100.0*angular_variation_larger/std::max<uint64_t>(1,variation_tests));VALUE("MEAN_ANGULAR_VARIATION_SQ",angle_span_sum/std::max<uint64_t>(1,variation_tests));VALUE("MEAN_RADIAL_VARIATION_SQ",radius_span_sum/std::max<uint64_t>(1,variation_tests));
    return selected;
}
'''

def main():
    root=next((p for p in (HERE,*HERE.parents) if (p/'spherical_fps/main_implementation/spherical_fps.cpp').exists()),None)
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path);ap.add_argument('--dataset',type=Path);ap.add_argument('--compiler',default='g++');ap.add_argument('--case',action='append',help='N,M (repeatable)');args=ap.parse_args()
    if root is None and (args.source is None or args.dataset is None): raise SystemExit('Place in sampling_bottleneck or supply --source and --dataset.')
    source=args.source or root/'spherical_fps/main_implementation/spherical_fps.cpp'
    data=args.dataset or root/'datasets/kitti/0000000000.bin'
    out=HERE/'results/spherical_angular_bounds';out.mkdir(parents=True,exist_ok=True)
    original=source.read_text(encoding='utf-8');start=original.index('static std::vector<int> spherical_fps(');end=original.index('int main(',start)
    generated=out/'angular_diagnostic.cpp';generated.write_text(original[:start]+CODE+'\n'+original[end:],encoding='utf-8')
    exe=out/'angular_diagnostic.exe';base=out/'baseline.exe'
    for src,binary in [(source,base),(generated,exe)]:
        print('Compiling',src.name,flush=True);subprocess.run([args.compiler,'-O3','-std=c++17','-ffp-contract=off',str(src),'-o',str(binary)],check=True)
    cases=[tuple(map(int,x.split(','))) for x in args.case] if args.case else [(16000,2048),(50000,1024),(100000,128),(100000,1024),(100000,4096)]
    rows=[]
    for n,m in cases:
        if not 1<=m<=n<=data.stat().st_size//16: print('Skipping invalid case',n,m);continue
        print(f'\nN={n:,}, M={m:,}',flush=True)
        def run(binary): return subprocess.check_output([str(binary),str(data),str(n),str(m)],text=True)
        b=run(base);s=run(exe);(out/f'N{n}_M{m}.txt').write_text(s)
        idx=lambda t: re.findall(r'^FPS_INDEX\[\d+\]=(\d+)$',t,re.M)
        assert len(idx(b))==m and idx(b)==idx(s),'FPS index mismatch'
        values=dict(re.findall(r'^([A-Z_]+)=([0-9.eE+\-]+)$',s,re.M));assert int(values['UNSAFE_REJECTIONS'])==0,'Unsafe rejection detected'
        assert re.search(r'^POINT_DISTANCE_EVALS=(\d+)$',b,re.M).group(1)==values['POINT_DISTANCE_EVALS'],'Baseline visit mismatch'
        keys=['ACTIVE_REGION_TESTS','EXTRA_REGION_SKIPS','EXTRA_REGION_POINT_VISITS_AVOIDABLE','POINT_BOUND_REJECTIONS','RADIAL_ONLY_REJECTIONS','ADDITIONAL_ANGULAR_POINT_REJECTIONS','ACTUAL_UPDATES','UNSAFE_REJECTIONS','CONE_TIGHTER_THAN_AABB','CONE_TIGHTER_THAN_COMBINED','WEIGHTED_RADIAL_WIDTH','WEIGHTED_CONE_HALF_ANGLE_DEG','MAX_CONE_HALF_ANGLE_DEG','ANGULAR_SHARE_OF_SQUARED_DISTANCE_PCT','ANGULAR_DOMINANT_PAIRS_PCT','ANGULAR_VARIATION_DOMINANT_REGION_TESTS_PCT','MEAN_ANGULAR_VARIATION_SQ','MEAN_RADIAL_VARIATION_SQ']
        row={'N':n,'M':m,'BASELINE_POINT_VISITS':values['POINT_DISTANCE_EVALS'],**{k:values[k] for k in keys}}
        rows.append(row);print('Indices MATCH; no unsafe rejections.');
        for k,v in row.items(): print(f'  {k}: {v}')
        with (out/f'N{n}_M{m}_regions.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['region','points','r_min','r_max','cone_half_angle_deg']);w.writerows(x.split(',') for x in re.findall(r'^ANG_REGION=(.*)$',s,re.M))
    if rows:
        with (out/'summary.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('\nResults:',out,'\nDiagnostic only: timings include instrumentation; do not compare runtime.')
if __name__=='__main__': main()
