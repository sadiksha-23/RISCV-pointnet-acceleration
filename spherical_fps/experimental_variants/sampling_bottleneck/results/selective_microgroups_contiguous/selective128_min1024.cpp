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
        for(int i=0;i<AZ_BINS;++i) {
            const double angle=TWO_PI*static_cast<double>(i)/AZ_BINS;
            double xx=std::cos(angle),yy=std::sin(angle);
            if(std::abs(xx)<1e-14)xx=0;
            if(std::abs(yy)<1e-14)yy=0;
            rays.push_back({xx,yy});
        }
        return rays;
    }();
    Ray2 q{-x,-y};
    if (AZ_BINS != 24) { // Retain F2 for other build-time bin counts.
        int lo=0,hi=AZ_BINS;
        while(lo<hi){int mid=lo+(hi-lo)/2;if(polar_less(q,boundaries[mid]))hi=mid;else lo=mid+1;}
        return std::clamp(lo-1,0,AZ_BINS-1);
    }
    // Exact axes go to the sector that begins on that axis.
    // Handle signed zero consistently with F2's half-plane predicate.
    int quadrant;
    if (q.y > 0.0) quadrant=(q.x > 0.0 ? 0 : 1);
    else if (q.y < 0.0) quadrant=(q.x < 0.0 ? 2 : 3);
    else quadrant=(q.x >= 0.0 ? 0 : 2);
    const int base=6*quadrant;

    // Binary search within just six sector boundaries; same predicates as F2.
    int lo=base,hi=base+6;
    while(lo<hi){int mid=lo+(hi-lo)/2;if(polar_less(q,boundaries[mid]))hi=mid;else lo=mid+1;}
    return std::clamp(lo-1,0,AZ_BINS-1);
}
static inline bool f2_elevation_above(double horizontal_sq, double z, int mid) {
    static const std::vector<std::pair<double,double>> boundaries=[] {
        std::vector<std::pair<double,double>> edges;
        for(int k=0;k<=EL_BINS;++k) {
            const double theta=-PI/2.0+(PI*k)/EL_BINS;
            edges.emplace_back(std::sin(theta),std::cos(theta));
        }
        return edges;
    }();
    const auto& edge=boundaries[mid];
    const double lhs=z*edge.second;
    const double sin_theta=edge.first;
    bool above;
    if (sin_theta == 0.0) above=lhs >= 0.0;
    else if (sin_theta > 0.0) {
        above=lhs >= 0.0 && lhs*lhs >= horizontal_sq*sin_theta*sin_theta;
    } else {
        above=lhs >= 0.0 || lhs*lhs <= horizontal_sq*sin_theta*sin_theta;
    }
    return above;
}
static int direct_elevation_bin(double x,double y,double z) {
    const double horizontal_sq=x*x+y*y;
    if(horizontal_sq==0.0) return z>=0 ? EL_BINS-1 : 0;
    // The middle boundary is exactly 0 degrees: equivalent to z>=0 here.
    if(z>=0.0) {
        if(f2_elevation_above(horizontal_sq,z,6))
            return f2_elevation_above(horizontal_sq,z,7) ? 7 : 6;
        return f2_elevation_above(horizontal_sq,z,5) ? 5 : 4;
    }
    if(f2_elevation_above(horizontal_sq,z,2))
        return f2_elevation_above(horizontal_sq,z,3) ? 3 : 2;
    return f2_elevation_above(horizontal_sq,z,1) ? 1 : 0;
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
struct Counters {uint64_t micro_checks=0,micro_skips=0,micro_skipped_points=0;uint64_t checked_regions=0,skipped_regions=0,skipped_points=0,evaluated_distances=0;int occupied_regions=0;};
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

// Strictly conservative early-out: matches the original lower_bound()
// safety reduction and comparison margin, but tests the AABB alone.
// If it cannot prove a skip, the original three-bound calculation runs.
static inline bool aabb_can_skip_early(const Region& g,
                                      const std::vector<float>& xyz,
                                      int newest, float current_max_sq) {
    const size_t p=static_cast<size_t>(newest)*3;
    const double x=static_cast<double>(xyz[p]);
    const double y=static_cast<double>(xyz[p+1]);
    const double z=static_cast<double>(xyz[p+2]);
    const double bx=std::max({g.xmin-x,0.0,x-g.xmax});
    const double by=std::max({g.ymin-y,0.0,y-g.ymax});
    const double bz=std::max({g.zmin-z,0.0,z-g.zmax});
    const double raw_lb=std::sqrt(bx*bx+by*by+bz*bz);
    const double lb=std::max(0.0,raw_lb-1e-6*(1.0+raw_lb));
    const double max_sq=static_cast<double>(current_max_sq);
    const double margin=1e-5*(1.0+std::abs(max_sq));
    return lb*lb > max_sq+margin;
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
static constexpr bool PACKED=false;

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
        auto microregions=make_microregions(xyz,regions,128,1024);
        auto packed=PACKED ? pack_regions(xyz,regions,n) : PackedPoints{};
        const auto end_prep=std::chrono::steady_clock::now();
        m5_dump_stats(0,0);
        m5_reset_stats(0,0);
        Counters counters;
        const std::vector<int> selected=spherical_fps(xyz,n,m,origin,regions,microregions,packed,counters);
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
        std::printf("MICRO_CHECKS=%llu\nMICRO_SKIPS=%llu\nMICRO_SKIPPED_POINTS=%llu\n",(unsigned long long)counters.micro_checks,(unsigned long long)counters.micro_skips,(unsigned long long)counters.micro_skipped_points);
        std::printf("CORRECTNESS=NOT_CHECKED (performance-only run)\n");return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"ERROR: %s\n",e.what());return 2;}
}
