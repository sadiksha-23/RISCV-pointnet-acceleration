// RVV SphericalFPS: same F2 region geometry and pruning; vectorized point-distance update.
// RVV builds require -march=rv64gcv -mabi=lp64d and riscv_vector.h.
// Host builds use the scalar fallback for correctness smoke tests, not RVV performance.
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
#if defined(__riscv_vector)
#include <riscv_vector.h>
#define SPHERICAL_FPS_HAS_RVV 1
#else
#define SPHERICAL_FPS_HAS_RVV 0
#endif
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
    // Region-contiguous SoA storage makes distance updates vector-loadable.
    // Ordering matches members exactly; local winner tie-breaking remains scalar.
    std::vector<float> px,py,pz,near_sq;
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
        g.px.reserve(g.members.size());g.py.reserve(g.members.size());g.pz.reserve(g.members.size());
        g.near_sq.assign(g.members.size(),1e38f);
        for (int i:g.members) {const size_t p=static_cast<size_t>(i)*3;
            g.px.push_back(xyz[p]);g.py.push_back(xyz[p+1]);g.pz.push_back(xyz[p+2]);
        }
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
static std::vector<int> spherical_fps(const std::vector<float>& xyz,int n,int m,const double origin[3],std::vector<Region>& regions,Counters& c){
    std::vector<int> selected(static_cast<size_t>(m));selected[0]=0;
    c.occupied_regions=static_cast<int>(regions.size());
    for(int step=1;step<m;++step){const int newest=selected[step-1];const size_t p=static_cast<size_t>(newest)*3;
        const double ox=static_cast<double>(xyz[p])-origin[0],oy=static_cast<double>(xyz[p+1])-origin[1],oz=static_cast<double>(xyz[p+2])-origin[2];
        const double selected_radius=std::sqrt(ox*ox+oy*oy+oz*oz);
        for(Region& g:regions){++c.checked_regions;
            const double lb=lower_bound(g,xyz,newest,selected_radius);
            const double max_sq=static_cast<double>(g.max_nearest_sq);
            const double margin=1e-5*(1.0+std::abs(max_sq));
            if (lb*lb > max_sq+margin) {
                ++c.skipped_regions;c.skipped_points+=g.members.size();continue;
            }
            const float sx=xyz[p],sy=xyz[p+1],sz=xyz[p+2];
            const size_t count=g.members.size();
#if SPHERICAL_FPS_HAS_RVV
            size_t offset=0;
            while(offset<count){
                const size_t vl=__riscv_vsetvl_e32m1(count-offset);
                const vfloat32m1_t x=__riscv_vle32_v_f32m1(g.px.data()+offset,vl);
                const vfloat32m1_t y=__riscv_vle32_v_f32m1(g.py.data()+offset,vl);
                const vfloat32m1_t z=__riscv_vle32_v_f32m1(g.pz.data()+offset,vl);
                const vfloat32m1_t dx=__riscv_vfsub_vf_f32m1(x,sx,vl);
                const vfloat32m1_t dy=__riscv_vfsub_vf_f32m1(y,sy,vl);
                const vfloat32m1_t dz=__riscv_vfsub_vf_f32m1(z,sz,vl);
                const vfloat32m1_t xx=__riscv_vfmul_vv_f32m1(dx,dx,vl);
                const vfloat32m1_t yy=__riscv_vfmul_vv_f32m1(dy,dy,vl);
                const vfloat32m1_t zz=__riscv_vfmul_vv_f32m1(dz,dz,vl);
                const vfloat32m1_t xy=__riscv_vfadd_vv_f32m1(xx,yy,vl);
                const vfloat32m1_t d=__riscv_vfadd_vv_f32m1(xy,zz,vl);
                const vfloat32m1_t previous=__riscv_vle32_v_f32m1(g.near_sq.data()+offset,vl);
                const vfloat32m1_t updated=__riscv_vfmin_vv_f32m1(previous,d,vl);
                __riscv_vse32_v_f32m1(g.near_sq.data()+offset,updated,vl);
                offset+=vl;
            }
#else
            for(size_t k=0;k<count;++k){
                const float dx=g.px[k]-sx,dy=g.py[k]-sy,dz=g.pz[k]-sz;
                const float d=dx*dx+dy*dy+dz*dz;
                if(d<g.near_sq[k])g.near_sq[k]=d;
            }
#endif
            c.evaluated_distances+=static_cast<uint64_t>(count);
            float best=-1;int best_i=-1;
            for(size_t k=0;k<count;++k){
                const float d=g.near_sq[k]; const int i=g.members[k];
                if(d>best||(d==best&&(best_i<0||i<best_i))){best=d;best_i=i;}
            }
            g.max_nearest_sq=best;g.best_index=best_i;
        }
        float global_best=-1;int global_i=-1;
        for(const Region& g:regions)if(g.max_nearest_sq>global_best||(g.max_nearest_sq==global_best&&(global_i<0||g.best_index<global_i))){global_best=g.max_nearest_sq;global_i=g.best_index;}
        if(global_i<0)throw std::runtime_error("No FPS winner");selected[step]=global_i;
    }
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
        std::printf("ALGORITHM=LowerBoundSphericalFPS_RVV\nRVV_ACTIVE=%d\nDATASET=%s\nN=%d\nM=%d\n",SPHERICAL_FPS_HAS_RVV,argv[1],n,m);
        std::printf("AZ_BINS=%d\nEL_BINS=%d\nRAD_BINS=%d\n",AZ_BINS,EL_BINS,RAD_BINS);
        std::printf("OCCUPIED_REGIONS=%d\nREGION_TESTS=%llu\nREGIONS_SKIPPED=%llu\n",counters.occupied_regions,(unsigned long long)counters.checked_regions,(unsigned long long)counters.skipped_regions);
        std::printf("POINT_DISTANCE_EVALS=%llu\nSKIPPED_POINT_EVALS=%llu\n",(unsigned long long)counters.evaluated_distances,(unsigned long long)counters.skipped_points);
        std::printf("DISTANCE_EVAL_REDUCTION_PCT=%.5f\n",skip_pct);
        for(int i=0;i<m;++i)std::printf("FPS_INDEX[%d]=%d\n",i,selected[i]);
        std::printf("CORRECTNESS=NOT_CHECKED (performance-only run)\n");return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"ERROR: %s\n",e.what());return 2;}
}
