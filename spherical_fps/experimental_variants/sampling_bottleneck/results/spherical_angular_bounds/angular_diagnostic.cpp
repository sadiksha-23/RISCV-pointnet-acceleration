// FINAL SCALAR SphericalFPS: exact F2 (01 + 05 + 06 + A) implementation.
// Source preserved from the winning combined gem5 benchmark.
// Performance ROI: centroid + region preparation + sampling; excludes load and output.
// Runtime: <KITTI_XYZI.bin> <N> <M>. Compile with -O3 -std=c++17 -ffp-contract=off.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#ifdef ENABLE_GEM5
extern "C" {
#include <gem5/m5ops.h>
}
#else
static inline void m5_reset_stats(uint64_t, uint64_t) {}
static inline void m5_dump_stats(uint64_t, uint64_t) {}
#endif
#ifndef AZ_BINS
#define AZ_BINS 24
#endif
#ifndef EL_BINS
#define EL_BINS 8
#endif
#ifndef RAD_BINS
#define RAD_BINS 6
#endif
static constexpr double PI=3.1415926535897932384626433832795;
static constexpr double TWO_PI=6.283185307179586476925286766559;
struct Ray2 {double x,y;};
static bool polar_half(const Ray2& r) {return r.y>0.0 || (r.y==0.0 && r.x>=0.0);}
static bool polar_less(const Ray2& a,const Ray2& b) {const bool ha=polar_half(a),hb=polar_half(b);if(ha!=hb)return ha;return a.x*b.y-a.y*b.x>0.0;}
static int direct_azimuth_bin(double x,double y) {
    if(x==0.0&&y==0.0)return 0;
    static const std::vector<Ray2> boundaries=[] {
        std::vector<Ray2> rays; rays.reserve(AZ_BINS);
        for(int i=0;i<AZ_BINS;++i) {const double angle=TWO_PI*static_cast<double>(i)/AZ_BINS;
            double xx=std::cos(angle),yy=std::sin(angle);
            if(std::abs(xx)<1e-14)xx=0;if(std::abs(yy)<1e-14)yy=0;
            rays.push_back({xx,yy});}return rays;}();
    Ray2 q{-x,-y};int lo=0,hi=AZ_BINS;
    while(lo<hi){int mid=lo+(hi-lo)/2;if(polar_less(q,boundaries[mid]))hi=mid;else lo=mid+1;}
    return std::clamp(lo-1,0,AZ_BINS-1);
}
static int direct_elevation_bin(double x,double y,double z) {
    const double horizontal_sq = x*x+y*y;
    if (horizontal_sq==0.0) return z>=0 ? EL_BINS-1 : 0;
    static const std::vector<std::pair<double,double>> boundaries=[] {
        std::vector<std::pair<double,double>> edges;
        for(int k=0;k<=EL_BINS;++k){const double theta=-PI/2.0+(PI*k)/EL_BINS;edges.emplace_back(std::sin(theta),std::cos(theta));}
        return edges;}();
    int lo=0,hi=EL_BINS;
    while(lo+1<hi){int mid=lo+(hi-lo)/2;const auto& edge=boundaries[mid];
        // cos(theta)>0 for internal boundaries. Preserve signs before squaring.
        const double lhs = z*edge.second;
        const double sin_theta = edge.first;
        bool above;
        if (sin_theta == 0.0) above = lhs >= 0.0;
        else if (sin_theta > 0.0) {
            above = lhs >= 0.0 && lhs*lhs >= horizontal_sq*sin_theta*sin_theta;
        } else {
            above = lhs >= 0.0 || lhs*lhs <= horizontal_sq*sin_theta*sin_theta;
        }
        if (above) lo=mid;
        else hi=mid;}
    return lo;
}
struct Region {
    std::vector<int> members;
    double cx=0,cy=0,cz=0,radius=0;
    double radial_min=std::numeric_limits<double>::infinity(),radial_max=0;
    double radial_min_sq=std::numeric_limits<double>::infinity();
    double radial_max_sq=0.0;
    double xmin=std::numeric_limits<double>::infinity(), ymin=std::numeric_limits<double>::infinity(), zmin=std::numeric_limits<double>::infinity();
    double xmax=-std::numeric_limits<double>::infinity(), ymax=-std::numeric_limits<double>::infinity(), zmax=-std::numeric_limits<double>::infinity();
    float max_nearest_sq=1e38f;
    int best_index=-1;
};
struct Counters {uint64_t checked_regions=0,skipped_regions=0,skipped_points=0,evaluated_distances=0;int occupied_regions=0;};
static bool load_kitti_xyz(const char* filename,std::vector<float>& xyz) {
    std::ifstream file(filename,std::ios::binary|std::ios::ate);
    if(!file){std::fprintf(stderr,"ERROR: cannot open KITTI file: %s\n",filename);return false;}
    const std::streamoff bytes=file.tellg();
    if(bytes<=0||bytes%(4*static_cast<int>(sizeof(float)))!=0){std::fprintf(stderr,"ERROR: invalid KITTI XYZI data\n");return false;}
    file.seekg(0,std::ios::beg);std::vector<float> raw(static_cast<size_t>(bytes)/sizeof(float));
    if(!file.read(reinterpret_cast<char*>(raw.data()),static_cast<std::streamsize>(bytes))){std::fprintf(stderr,"ERROR: reading KITTI\n");return false;}
    xyz.resize((raw.size()/4)*3);
    for(size_t i=0;i<raw.size()/4;++i)for(size_t d=0;d<3;++d){const float v=raw[4*i+d];if(!std::isfinite(v))return false;xyz[3*i+d]=v;}
    return true;
}
static inline float sqdist(const std::vector<float>& xyz,int a,int b){size_t i=static_cast<size_t>(a)*3,j=static_cast<size_t>(b)*3;
    float dx=xyz[i]-xyz[j],dy=xyz[i+1]-xyz[j+1],dz=xyz[i+2]-xyz[j+2];return dx*dx+dy*dy+dz*dz;}
static void calculate_centroid(const std::vector<float>& xyz,int n,double out[3]){
    out[0]=out[1]=out[2]=0;for(int i=0;i<n;++i)for(int d=0;d<3;++d)out[d]+=static_cast<double>(xyz[static_cast<size_t>(i)*3+d]);
    for(int d=0;d<3;++d)out[d]/=static_cast<double>(n);
}
static std::vector<Region> build_regions(const std::vector<float>& xyz,int n,const double origin[3]){
    const size_t bins=static_cast<size_t>(AZ_BINS)*EL_BINS*RAD_BINS;
    std::vector<Region> grid(bins);
    std::vector<double> radii_sq(static_cast<size_t>(n));
    std::vector<int> az_bin(static_cast<size_t>(n)),el_bin(static_cast<size_t>(n));
    double global_rmax_sq=1e-24;
    for(int i=0;i<n;++i){const size_t p=static_cast<size_t>(i)*3;
        const double x=static_cast<double>(xyz[p])-origin[0],y=static_cast<double>(xyz[p+1])-origin[1],z=static_cast<double>(xyz[p+2])-origin[2];
        const double r_sq=x*x+y*y+z*z;
        radii_sq[i]=r_sq;
        az_bin[i]=direct_azimuth_bin(x,y);el_bin[i]=direct_elevation_bin(x,y,z);
        if (r_sq>global_rmax_sq) global_rmax_sq=r_sq;
    }
    // Precompute squared radial boundaries once, not one square root per point.
    const double global_rmax=std::sqrt(global_rmax_sq);
    std::vector<double> radial_edges_sq(static_cast<size_t>(RAD_BINS)+1);
    for (int k=1;k<RAD_BINS;++k) {
        const double boundary=global_rmax*static_cast<double>(k)/RAD_BINS;
        radial_edges_sq[k]=boundary*boundary;
    }
    for (int i=0; i<n; ++i) {
        const int ai=az_bin[i];
        const int ei=el_bin[i];
        const double rsq=radii_sq[i];
        // upper_bound behavior: equality at a boundary goes into the upper bin.
        const int ri=static_cast<int>(std::upper_bound(radial_edges_sq.begin()+1,
            radial_edges_sq.begin()+RAD_BINS,rsq)- (radial_edges_sq.begin()+1));
        const size_t key=(static_cast<size_t>(ri)*EL_BINS+ei)*AZ_BINS+ai;
        Region& g=grid[key];g.members.push_back(i);
        const size_t p=static_cast<size_t>(i)*3;
        g.cx+=xyz[p]; g.cy+=xyz[p+1]; g.cz+=xyz[p+2];
        g.xmin=std::min(g.xmin, static_cast<double>(xyz[p])); g.xmax=std::max(g.xmax, static_cast<double>(xyz[p]));
        g.ymin=std::min(g.ymin, static_cast<double>(xyz[p+1])); g.ymax=std::max(g.ymax, static_cast<double>(xyz[p+1]));
        g.zmin=std::min(g.zmin, static_cast<double>(xyz[p+2])); g.zmax=std::max(g.zmax, static_cast<double>(xyz[p+2]));
        g.radial_min_sq=std::min(g.radial_min_sq,rsq);
        g.radial_max_sq=std::max(g.radial_max_sq,rsq);
    }
    std::vector<Region> occupied;occupied.reserve(std::min(bins,static_cast<size_t>(n)));
    for(Region& g:grid){if(g.members.empty())continue;
        const double count=static_cast<double>(g.members.size());
        g.cx/=count; g.cy/=count; g.cz/=count;
        g.radial_min=std::sqrt(g.radial_min_sq);
        g.radial_max=std::sqrt(g.radial_max_sq);
        for (int i: g.members) {const size_t p=static_cast<size_t>(i)*3;
            const double dx=static_cast<double>(xyz[p])-g.cx,dy=static_cast<double>(xyz[p+1])-g.cy,dz=static_cast<double>(xyz[p+2])-g.cz;
            g.radius=std::max(g.radius,std::sqrt(dx*dx+dy*dy+dz*dz));
        }
        g.radius+=1e-7*(1.0+g.radius);
        g.radial_min=std::max(0.0,g.radial_min-1e-7*(1.0+g.radial_min));
        g.radial_max+=1e-7*(1.0+g.radial_max);
        g.best_index=g.members.front();occupied.push_back(std::move(g));
    }
    return occupied;
}
static double lower_bound(const Region& g,const std::vector<float>& xyz,int newest,double selected_radius){
    const size_t p=static_cast<size_t>(newest)*3;
    const double x=static_cast<double>(xyz[p]),y=static_cast<double>(xyz[p+1]),z=static_cast<double>(xyz[p+2]);
    const double dx=x-g.cx,dy=y-g.cy,dz=z-g.cz;
    const double center_dist=std::sqrt(dx*dx+dy*dy+dz*dz);
    const double xyz_lb=std::max(0.0,center_dist-g.radius);
    const double radial_lb=std::max({0.0,g.radial_min-selected_radius,selected_radius-g.radial_max});
    // AABB distance is another safe lower bound: exact points are inside the box.
    const double bx=std::max({g.xmin-x, 0.0, x-g.xmax});
    const double by=std::max({g.ymin-y, 0.0, y-g.ymax});
    const double bz=std::max({g.zmin-z, 0.0, z-g.zmax});
    const double aabb_lb=std::sqrt(bx*bx+by*by+bz*bz);
    const double combined=std::max({xyz_lb,radial_lb,aabb_lb});
    return std::max(0.0,combined-1e-6*(1.0+combined));
}

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

int main(int argc,char** argv){
    if(argc!=4){std::fprintf(stderr,"Usage: %s <KITTI_XYZI.bin> <N> <M>\n",argv[0]);return 2;}
    try{const int n=std::stoi(argv[2]),m=std::stoi(argv[3]);
        const long long bin_count=1LL*AZ_BINS*EL_BINS*RAD_BINS;
        if(n<1||m<1||m>n||AZ_BINS<1||EL_BINS<1||RAD_BINS<1||bin_count>100000)return 2;
        std::vector<float> xyz;if(!load_kitti_xyz(argv[1],xyz))return 1;
        if(static_cast<size_t>(n)*3>xyz.size()){std::fprintf(stderr,"ERROR: fewer than N points\n");return 2;}
        m5_reset_stats(0,0);
        const auto start_prep=std::chrono::steady_clock::now();
        double origin[3];calculate_centroid(xyz,n,origin);
        std::vector<Region> regions=build_regions(xyz,n,origin);
        const auto end_prep=std::chrono::steady_clock::now();
        Counters counters;
        const std::vector<int> selected=spherical_fps(xyz,n,m,origin,regions,counters);
        const auto end_sample=std::chrono::steady_clock::now();m5_dump_stats(0,0);
        const double prep_ms=std::chrono::duration<double,std::milli>(end_prep-start_prep).count();
        const double sample_ms=std::chrono::duration<double,std::milli>(end_sample-end_prep).count();
        std::printf("PREPARATION_MS=%.6f\nSAMPLING_MS=%.6f\nTOTAL_MS=%.6f\n",prep_ms,sample_ms,prep_ms+sample_ms);
        const uint64_t all_evals=static_cast<uint64_t>(n)*(m-1);
        const double skip_pct=all_evals?100.0*(1.0-static_cast<double>(counters.evaluated_distances)/all_evals):0;
        std::printf("ALGORITHM=LowerBoundSphericalFPS\nDATASET=%s\nN=%d\nM=%d\n",argv[1],n,m);
        std::printf("AZ_BINS=%d\nEL_BINS=%d\nRAD_BINS=%d\n",AZ_BINS,EL_BINS,RAD_BINS);
        std::printf("OCCUPIED_REGIONS=%d\nREGION_TESTS=%llu\nREGIONS_SKIPPED=%llu\n",counters.occupied_regions,(unsigned long long)counters.checked_regions,(unsigned long long)counters.skipped_regions);
        std::printf("POINT_DISTANCE_EVALS=%llu\nSKIPPED_POINT_EVALS=%llu\n",(unsigned long long)counters.evaluated_distances,(unsigned long long)counters.skipped_points);
        std::printf("DISTANCE_EVAL_REDUCTION_PCT=%.5f\n",skip_pct);
        for(int i=0;i<m;++i)std::printf("FPS_INDEX[%d]=%d\n",i,selected[i]);
        std::printf("CORRECTNESS=NOT_CHECKED (performance-only run)\n");return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"ERROR: %s\n",e.what());return 2;}
}
