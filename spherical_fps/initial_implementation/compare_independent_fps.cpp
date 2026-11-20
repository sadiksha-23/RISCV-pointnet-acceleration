// Independent head-to-head benchmark: scalar RadiusFPS vs spherical-region FPS.
// Includes RadiusFPS implementation as source to preserve its exact baseline.
#define main included_radius_benchmark_main
#include "radiusfps_spherical_test.cpp"
#undef main

#include <numeric>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <cstdlib>

using Clk = std::chrono::steady_clock;
static double ms(Clk::time_point a, Clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b-a).count();
}

struct SRegion {
    std::vector<int> members;
    int rep=-1;
    double rlo=1e300,rhi=0, alo=1e300,ahi=-1e300, elo=1e300,ehi=-1e300;
    float rep_nearest=1e38f;
    double upper=1e300;
};
struct SphericalStats {
    double prep=0, sample=0, total=0;
    uint64_t region_prunes=0, points_pruned=0, distance_evals=0, points_evaluated=0, upper_evals=0;
    int regions=0;
    std::vector<int> indices;
};

static double cos_min_interval(double lo, double hi, double psi) {
    double x=std::min(std::cos(lo-psi),std::cos(hi-psi));
    // cos(x) reaches -1 when x = PI + 2*pi*k.
    double k=std::ceil((lo-psi-PI)/TWO_PI);
    double angle=psi+PI+TWO_PI*k;
    if(angle <= hi+1e-12) x=-1;
    return x;
}
static double spherical_upper(const SRegion &g, double ox, double oy, double oz,
                              float qx, float qy, float qz) {
    double vx=(double)qx-ox, vy=(double)qy-oy, vz=(double)qz-oz;
    double xy=std::hypot(vx,vy), psi=std::atan2(vy,vx), vsq=vx*vx+vy*vy+vz*vz;
    double A=xy*cos_min_interval(g.alo,g.ahi,psi), B=vz;
    double low=A*std::cos(g.elo)+B*std::sin(g.elo);
    double high=A*std::cos(g.ehi)+B*std::sin(g.ehi);
    double dmin=std::min(low,high);
    double crit=std::atan2(B,A)+PI;
    if(crit>PI) crit-=TWO_PI;
    if(crit>=g.elo && crit<=g.ehi) dmin=std::min(dmin,-std::hypot(A,B));
    double d0=g.rlo*g.rlo+vsq-2*g.rlo*dmin;
    double d1=g.rhi*g.rhi+vsq-2*g.rhi*dmin;
    double ub=std::sqrt(std::max(0.,std::max(d0,d1)));
    return ub+1e-6*(1+ub); // conservative roundoff padding
}
static inline float float_sqdist(const std::vector<float>& xyz,int i,int k) {
    const size_t a=(size_t)i*3,b=(size_t)k*3;
    float dx=xyz[a]-xyz[b],dy=xyz[a+1]-xyz[b+1],dz=xyz[a+2]-xyz[b+2];
    return dx*dx+dy*dy+dz*dz;
}
static SphericalStats run_spherical(const std::vector<float>& xyz,int n,int m,
                                  const std::string& origin_type,int az_bins=24,int el_bins=8,int r_bins=6) {
    SphericalStats out;
    auto start=Clk::now();
    double ox=0,oy=0,oz=0;
    if(origin_type=="centroid"){
        for(int i=0;i<n;i++){
            ox+=(double)xyz[(size_t)i*3];oy+=(double)xyz[(size_t)i*3+1];oz+=(double)xyz[(size_t)i*3+2];
        }
        ox/=n;oy/=n;oz/=n;
    }else if(origin_type=="first"){
        ox=xyz[0];oy=xyz[1];oz=xyz[2];
    }else if(origin_type!="sensor"){
        fprintf(stderr,"Unsupported origin %s\n",origin_type.c_str());std::exit(2);
    }
    std::vector<double> rs(n),az(n),el(n);
    double rmax=1e-12;
    for(int i=0;i<n;i++){
        double dx=(double)xyz[(size_t)i*3]-ox,dy=(double)xyz[(size_t)i*3+1]-oy,dz=(double)xyz[(size_t)i*3+2]-oz;
        rs[i]=std::sqrt(dx*dx+dy*dy+dz*dz);
        az[i]=std::atan2(dy,dx);el[i]=std::atan2(dz,std::hypot(dx,dy));
        if(rs[i]>rmax)rmax=rs[i];
    }
    const int region_total=az_bins*el_bins*r_bins;
    std::vector<SRegion> groups((size_t)region_total);
    for(int i=0;i<n;i++){
        int ai=std::clamp((int)((az[i]+PI)/TWO_PI*az_bins),0,az_bins-1);
        int ei=std::clamp((int)((el[i]+PI/2)/PI*el_bins),0,el_bins-1);
        int ri=std::clamp((int)(rs[i]/rmax*r_bins),0,r_bins-1);
        SRegion &g=groups[(ri*el_bins+ei)*az_bins+ai];
        if(g.rep<0)g.rep=i;
        g.members.push_back(i);
        g.rlo=std::min(g.rlo,rs[i]);g.rhi=std::max(g.rhi,rs[i]);
        g.alo=std::min(g.alo,az[i]);g.ahi=std::max(g.ahi,az[i]);
        g.elo=std::min(g.elo,el[i]);g.ehi=std::max(g.ehi,el[i]);
    }
    std::vector<SRegion> occupied;
    occupied.reserve(region_total);
    for(auto &g:groups)if(g.rep>=0){
        g.rlo=std::max(0.,g.rlo-1e-8);g.rhi+=1e-8;
        g.alo-=1e-10;g.ahi+=1e-10;g.elo-=1e-10;g.ehi+=1e-10;
        occupied.push_back(std::move(g));
    }
    out.regions=(int)occupied.size();
    std::vector<float> nearest((size_t)n,1e38f);
    std::vector<int> last((size_t)n,0);
    out.indices.resize((size_t)m);out.indices[0]=0;
    auto prepped=Clk::now();
    for(int j=1;j<m;j++){
        const int newest=out.indices[j-1];
        float lb=-1; int best=-1;
        for(SRegion& g:occupied){
            ++out.upper_evals;
            float d=float_sqdist(xyz,g.rep,newest);
            if(d<g.rep_nearest)g.rep_nearest=d;
            g.upper=std::min(g.upper,spherical_upper(g,ox,oy,oz,
                         xyz[(size_t)newest*3],xyz[(size_t)newest*3+1],xyz[(size_t)newest*3+2]));
            if(g.rep_nearest>lb||(g.rep_nearest==lb&&(best<0||g.rep<best))){
                lb=g.rep_nearest;best=g.rep;
            }
        }
        // Every representative is an actual point; maximum of their exact
        // nearest distances is an attainable lower bound for global max.
        float best_d2=lb;
        int best_i=best;
        for(SRegion& g:occupied){
            // Conservative floating-point margin; strict inequality respects ties.
            if(g.upper*g.upper+1e-4*(1+g.upper*g.upper)<(double)lb){
                ++out.region_prunes;out.points_pruned+=g.members.size();
                continue;
            }
            for(int point:g.members){
                ++out.points_evaluated;
                for(int t=last[(size_t)point];t<j;t++){
                    ++out.distance_evals;
                    float d2=float_sqdist(xyz,point,out.indices[(size_t)t]);
                    if(d2<nearest[(size_t)point])nearest[(size_t)point]=d2;
                }
                last[(size_t)point]=j;
                float value=nearest[(size_t)point];
                if(value>best_d2||(value==best_d2&&point<best_i)){
                    best_d2=value;best_i=point;
                }
            }
        }
        if(best_i<0){fprintf(stderr,"No FPS winner at step %d\n",j);std::exit(3);}
        out.indices[(size_t)j]=best_i;
    }
    auto end=Clk::now();
    out.prep=ms(start,prepped);out.sample=ms(prepped,end);out.total=ms(start,end);
    return out;
}
static std::vector<int> basic_fps(const std::vector<float>& xyz,int n,int m){
    std::vector<float> nearest((size_t)n,1e38f);
    std::vector<int> chosen((size_t)m);chosen[0]=0;
    for(int j=1;j<m;j++){
        float best=-1;int bi=-1;
        for(int i=0;i<n;i++){
            float d=float_sqdist(xyz,i,chosen[(size_t)j-1]);
            if(d<nearest[(size_t)i])nearest[(size_t)i]=d;
            if(nearest[(size_t)i]>best||(nearest[(size_t)i]==best&&(bi<0||i<bi))){
                best=nearest[(size_t)i];bi=i;
            }
        }
        chosen[(size_t)j]=bi;
    }
    return chosen;
}
static double median(std::vector<double> values){
    std::sort(values.begin(),values.end());return values[values.size()/2];
}
int main(int argc,char** argv){
    std::string dataset="datasets/kitti/0000000000.bin",org="centroid";
    int n=100000,m=128,repeats=5;
    for(int i=1;i<argc;i++){
        auto get=[&](){if(i+1>=argc){fprintf(stderr,"Missing argument after %s\n",argv[i]);std::exit(2);}return std::string(argv[++i]);};
        if(!strcmp(argv[i],"--dataset"))dataset=get();
        else if(!strcmp(argv[i],"--n"))n=std::stoi(get());
        else if(!strcmp(argv[i],"--m"))m=std::stoi(get());
        else if(!strcmp(argv[i],"--repeats"))repeats=std::stoi(get());
        else if(!strcmp(argv[i],"--origin"))org=get();
        else {fprintf(stderr,"Unknown option: %s\n",argv[i]);return 2;}
    }
    if(n<1||m<1||m>n||repeats<1){fprintf(stderr,"Invalid N/M/repeats\n");return 2;}
    std::vector<float> xyz;
    if(!loadKittiFrame(dataset.c_str(),xyz))return 1;
    if((size_t)n*3>xyz.size()){
        fprintf(stderr,"Dataset smaller than N=%d\n",n);return 1;
    }
    printf("INDEPENDENT ALGORITHMS: RadiusFPS versus SphericalFPS (NO hybrid)\n");
    printf("DATASET=%s N=%d M=%d REPEATS=%d ORIGIN=%s\n",dataset.c_str(),n,m,repeats,org.c_str());
    std::vector<int> reference=basic_fps(xyz,n,m);
    printf("BASICFPS_REFERENCE_FIRST10=");
    for(int i=0;i<std::min(10,m);i++)printf("%s%d",i?",":"",reference[(size_t)i]);
    puts("");
    double origin[3]={0,0,0};
    std::vector<double> radius_prep,radius_sample,radius_total,sph_prep,sph_sample,sph_total;
    for(int rep=0;rep<repeats;rep++){
        // Alternate execution order to minimize systematic warm-cache bias.
        Result r;SphericalStats s;
        if(rep%2==0){
            r=runOnce(xyz,n,m,2.f,false,origin,false);
            s=run_spherical(xyz,n,m,org);
        }else{
            s=run_spherical(xyz,n,m,org);
            r=runOnce(xyz,n,m,2.f,false,origin,false);
        }
        for(int i=0;i<m;i++){
            if(r.indices[(size_t)i]!=reference[(size_t)i]||s.indices[(size_t)i]!=reference[(size_t)i]){
                fprintf(stderr,"FAIL EXACT FPS AT STEP %d: BasicFPS=%d RadiusFPS=%d SphericalFPS=%d\n",
                        i+1,reference[(size_t)i],r.indices[(size_t)i],s.indices[(size_t)i]);
                return 4;
            }
        }
        radius_prep.push_back(r.prep_ms);radius_sample.push_back(r.sample_ms);radius_total.push_back(r.total_ms);
        sph_prep.push_back(s.prep);sph_sample.push_back(s.sample);sph_total.push_back(s.total);
        printf("REPEAT=%d RadiusFPS: prep=%.3fms sample=%.3fms total=%.3fms\n",rep+1,r.prep_ms,r.sample_ms,r.total_ms);
        printf("REPEAT=%d SphericalFPS: prep=%.3fms sample=%.3fms total=%.3fms regions=%d points_pruned=%llu distance_evals=%llu\n",
               rep+1,s.prep,s.sample,s.total,s.regions,(unsigned long long)s.points_pruned,
               (unsigned long long)s.distance_evals);
        printf("EXACT_128_CHECK=PASS (%d/%d)\n",m,m);
    }
    double br=median(radius_total),bs=median(sph_total);
    printf("\nMEDIAN RadiusFPS: prep=%.3fms sample=%.3fms TOTAL=%.3fms\n",median(radius_prep),median(radius_sample),br);
    printf("MEDIAN SphericalFPS: prep=%.3fms sample=%.3fms TOTAL=%.3fms\n",median(sph_prep),median(sph_sample),bs);
    printf("SPEED_RATIO_Radius_over_Spherical=%.4fx (>1 means spherical faster)\n",br/bs);
    printf("WINNER=%s\n", bs<br ? "SphericalFPS" : br<bs ? "RadiusFPS" : "Tie");
    printf("CORRECTNESS=PASS (%d/%d exact matches for both algorithms, every repeat)\n",m,m);
    return 0;
}
