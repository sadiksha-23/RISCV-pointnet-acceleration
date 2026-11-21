// SphericalFPS: exact lower-bound FPS, scalar C++17, KITTI XYZ input.
//
// Isolated native optimization experiment. Based on the verified FPS algorithm.
// Original repository baseline remains untouched.
//
// Runtime arguments: <KITTI_XYZI.bin> <N> <M>
// Compile-time grid parameters (positive integers):
//   -DAZ_BINS=24 -DEL_BINS=8 -DRAD_BINS=6
//
// For gem5, build with -DENABLE_GEM5 and link the target's libm5.a.
// Performance-only implementation: no BasicFPS verification.

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

static constexpr double PI = 3.1415926535897932384626433832795;
static constexpr double TWO_PI = 6.283185307179586476925286766559;


// Experiment 1: azimuth/elevation bins via signed geometric boundary comparisons.
// Trig calls happen only once when sector boundaries are initialized.
struct Ray2 { double x, y; };
static bool polar_half(const Ray2& r) {
    return r.y > 0.0 || (r.y == 0.0 && r.x >= 0.0);
}
static bool polar_less(const Ray2& a, const Ray2& b) {
    const bool ha = polar_half(a), hb = polar_half(b);
    if (ha != hb) return ha;
    const double cross = a.x*b.y - a.y*b.x;
    return cross > 0.0;
}
static int direct_azimuth_bin(double x, double y) {
    if (x == 0.0 && y == 0.0) return 0;
    static const std::vector<Ray2> boundaries = [] {
        std::vector<Ray2> rays;
        rays.reserve(AZ_BINS);
        for (int i=0; i<AZ_BINS; ++i) {
            const double angle = TWO_PI * static_cast<double>(i) / AZ_BINS;
            double xx = std::cos(angle), yy = std::sin(angle);
            if (std::abs(xx) < 1e-14) xx=0;
            if (std::abs(yy) < 1e-14) yy=0;
            rays.push_back({xx,yy});
        }
        return rays;
    }();
    // Negating x,y rotates atan2's [-pi, pi] range to the [0, 2pi) ordering.
    Ray2 q{-x,-y};
    int lo=0, hi=AZ_BINS;
    while (lo<hi) {
        int mid = lo+(hi-lo)/2;
        if (polar_less(q,boundaries[mid])) hi=mid;
        else lo=mid+1;
    }
    return std::clamp(lo-1,0,AZ_BINS-1);
}
static int direct_elevation_bin(double x, double y, double z) {
    const double horizontal = std::sqrt(x*x+y*y);
    if (horizontal==0.0) return z>=0 ? EL_BINS-1 : 0;
    static const std::vector<std::pair<double,double>> boundaries = [] {
        std::vector<std::pair<double,double>> edges;
        for (int k=0; k<=EL_BINS; ++k) {
            const double theta=-PI/2.0 + (PI*k)/EL_BINS;
            edges.emplace_back(std::sin(theta),std::cos(theta));
        }
        return edges;
    }();
    int lo=0, hi=EL_BINS;
    while (lo+1<hi) {
        int mid=lo+(hi-lo)/2;
        const auto& edge=boundaries[mid];
        if (z*edge.second >= horizontal*edge.first) lo=mid;
        else hi=mid;
    }
    return lo;
}

struct Region {
    std::vector<int> members;
    double cx=0, cy=0, cz=0, radius=0;
    double radial_min=std::numeric_limits<double>::infinity();
    double radial_max=0;
    float max_nearest_sq=1e38f;
    int best_index=-1;
};

struct Counters {
    uint64_t checked_regions=0;
    uint64_t skipped_regions=0;
    uint64_t skipped_points=0;
    uint64_t evaluated_distances=0;
    int occupied_regions=0;
};

static bool load_kitti_xyz(const char* filename, std::vector<float>& xyz) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) {
        std::fprintf(stderr, "ERROR: cannot open KITTI file: %s\n", filename);
        return false;
    }
    const std::streamoff bytes=file.tellg();
    if (bytes<=0 || bytes % (4 * static_cast<int>(sizeof(float))) != 0) {
        std::fprintf(stderr, "ERROR: KITTI file is not valid XYZI float32 data\n");
        return false;
    }
    file.seekg(0, std::ios::beg);
    std::vector<float> raw(static_cast<size_t>(bytes)/sizeof(float));
    if (!file.read(reinterpret_cast<char*>(raw.data()),
                   static_cast<std::streamsize>(bytes))) {
        std::fprintf(stderr, "ERROR: cannot read KITTI file\n");
        return false;
    }
    const size_t count=raw.size()/4;
    xyz.resize(count*3);
    for (size_t i=0; i<count; ++i) {
        for (size_t d=0; d<3; ++d) {
            const float v=raw[4*i+d];
            if (!std::isfinite(v)) {
                std::fprintf(stderr, "ERROR: invalid coordinate at point %zu\n", i);
                return false;
            }
            xyz[3*i+d]=v;
        }
    }
    return true;
}

static inline float sqdist(const std::vector<float>& xyz, int a, int b) {
    const size_t i=static_cast<size_t>(a)*3;
    const size_t j=static_cast<size_t>(b)*3;
    const float dx=xyz[i]-xyz[j];
    const float dy=xyz[i+1]-xyz[j+1];
    const float dz=xyz[i+2]-xyz[j+2];
    return dx*dx+dy*dy+dz*dz;
}

static void calculate_centroid(const std::vector<float>& xyz, int n, double out[3]) {
    out[0]=out[1]=out[2]=0;
    for (int i=0; i<n; ++i)
        for (int d=0; d<3; ++d)
            out[d] += static_cast<double>(xyz[static_cast<size_t>(i)*3+d]);
    for (int d=0; d<3; ++d) out[d]/=static_cast<double>(n);
}

// Build centroid-centered spherical bins. Each occupied bin gets a certified
// XYZ bounding sphere and an interval of radial distances from the origin.
static std::vector<Region> build_regions(const std::vector<float>& xyz,
                                          int n, const double origin[3]) {
    const size_t bins=static_cast<size_t>(AZ_BINS)*EL_BINS*RAD_BINS;
    std::vector<Region> grid(bins);
    std::vector<double> radii(static_cast<size_t>(n));
    std::vector<int> az_bin(static_cast<size_t>(n));
    std::vector<int> el_bin(static_cast<size_t>(n));
    double global_rmax=1e-12;

    for (int i=0; i<n; ++i) {
        const size_t p=static_cast<size_t>(i)*3;
        const double x=static_cast<double>(xyz[p])-origin[0];
        const double y=static_cast<double>(xyz[p+1])-origin[1];
        const double z=static_cast<double>(xyz[p+2])-origin[2];
        const double r=std::sqrt(x*x+y*y+z*z);
        radii[i]=r;
        az_bin[i]=direct_azimuth_bin(x,y);
        el_bin[i]=direct_elevation_bin(x,y,z);
        if (r>global_rmax) global_rmax=r;
    }

    for (int i=0; i<n; ++i) {
        const int ai=az_bin[i];
        const int ei=el_bin[i];
        const int ri=std::clamp(static_cast<int>(radii[i]/global_rmax*RAD_BINS),0,RAD_BINS-1);
        const size_t key=(static_cast<size_t>(ri)*EL_BINS+ei)*AZ_BINS+ai;
        Region& g=grid[key];
        g.members.push_back(i);
        const size_t p=static_cast<size_t>(i)*3;
        g.cx+=xyz[p]; g.cy+=xyz[p+1]; g.cz+=xyz[p+2];
        g.radial_min=std::min(g.radial_min,radii[i]);
        g.radial_max=std::max(g.radial_max,radii[i]);
    }

    std::vector<Region> occupied;
    occupied.reserve(std::min(bins,static_cast<size_t>(n)));
    for (Region& g: grid) {
        if (g.members.empty()) continue;
        const double count=static_cast<double>(g.members.size());
        g.cx/=count; g.cy/=count; g.cz/=count;
        for (int i: g.members) {
            const size_t p=static_cast<size_t>(i)*3;
            const double dx=static_cast<double>(xyz[p])-g.cx;
            const double dy=static_cast<double>(xyz[p+1])-g.cy;
            const double dz=static_cast<double>(xyz[p+2])-g.cz;
            g.radius=std::max(g.radius,std::sqrt(dx*dx+dy*dy+dz*dz));
        }
        // Outward rounding provides conservative geometric enclosures.
        g.radius+=1e-7*(1.0+g.radius);
        g.radial_min=std::max(0.0,g.radial_min-1e-7*(1.0+g.radial_min));
        g.radial_max+=1e-7*(1.0+g.radial_max);
        g.best_index=g.members.front();
        occupied.push_back(std::move(g));
    }
    return occupied;
}

// Lower bound on distance(newest, every point in a region).
static double lower_bound(const Region& g, const std::vector<float>& xyz,
                          int newest, double selected_radius) {
    const size_t p=static_cast<size_t>(newest)*3;
    const double x=static_cast<double>(xyz[p]);
    const double y=static_cast<double>(xyz[p+1]);
    const double z=static_cast<double>(xyz[p+2]);
    const double dx=x-g.cx, dy=y-g.cy, dz=z-g.cz;
    const double center_dist=std::sqrt(dx*dx+dy*dy+dz*dz);
    const double xyz_lb=std::max(0.0,center_dist-g.radius);
    const double radial_lb=std::max({0.0,
                   g.radial_min-selected_radius,
                   selected_radius-g.radial_max});
    const double combined=std::max(xyz_lb,radial_lb);
    return std::max(0.0,combined-1e-6*(1.0+combined));
}

static std::vector<int> spherical_fps(const std::vector<float>& xyz, int n, int m,
                                      const double origin[3],
                                      std::vector<Region>& regions, Counters& c) {
    std::vector<float> nearest(static_cast<size_t>(n),1e38f);
    std::vector<int> selected(static_cast<size_t>(m));
    selected[0]=0;
    c.occupied_regions=static_cast<int>(regions.size());
    for (int step=1; step<m; ++step) {
        const int newest=selected[step-1];
        const size_t p=static_cast<size_t>(newest)*3;
        const double ox=static_cast<double>(xyz[p])-origin[0];
        const double oy=static_cast<double>(xyz[p+1])-origin[1];
        const double oz=static_cast<double>(xyz[p+2])-origin[2];
        const double selected_radius=std::sqrt(ox*ox+oy*oy+oz*oz);

        for (Region& g: regions) {
            ++c.checked_regions;
            const double lb=lower_bound(g,xyz,newest,selected_radius);
            const double max_sq=static_cast<double>(g.max_nearest_sq);
            const double margin=1e-5*(1.0+std::abs(max_sq));
            if (lb*lb > max_sq+margin) {
                ++c.skipped_regions;
                c.skipped_points+=g.members.size();
                continue;
            }

            float best=-1;
            int best_i=-1;
            for (int i: g.members) {
                const float d=sqdist(xyz,i,newest);
                ++c.evaluated_distances;
                if (d<nearest[i]) nearest[i]=d;
                if (nearest[i]>best ||
                    (nearest[i]==best && (best_i<0 || i<best_i))) {
                    best=nearest[i];
                    best_i=i;
                }
            }
            g.max_nearest_sq=best;
            g.best_index=best_i;
        }

        float global_best=-1;
        int global_i=-1;
        for (const Region& g: regions) {
            if (g.max_nearest_sq>global_best ||
                (g.max_nearest_sq==global_best &&
                 (global_i<0 || g.best_index<global_i))) {
                global_best=g.max_nearest_sq;
                global_i=g.best_index;
            }
        }
        if (global_i<0) throw std::runtime_error("No FPS winner");
        selected[step]=global_i;
    }
    return selected;
}

int main(int argc, char** argv) {
    if (argc!=4) {
        std::fprintf(stderr,"Usage: %s <KITTI_XYZI.bin> <N> <M>\n",argv[0]);
        return 2;
    }
    try {
        const int n=std::stoi(argv[2]);
        const int m=std::stoi(argv[3]);
        const long long bin_count=1LL*AZ_BINS*EL_BINS*RAD_BINS;
        if (n<1 || m<1 || m>n ||
            AZ_BINS<1 || EL_BINS<1 || RAD_BINS<1 ||
            bin_count>100000) {
            std::fprintf(stderr,"ERROR: invalid N, M, or grid configuration\n");
            return 2;
        }
        std::vector<float> xyz;
        if (!load_kitti_xyz(argv[1],xyz)) return 1;
        if (static_cast<size_t>(n)*3>xyz.size()) {
            std::fprintf(stderr,"ERROR: KITTI frame has fewer than N=%d points\n",n);
            return 2;
        }

        // gem5 interval: centroid + region preparation + sampling.
        // Loading/printing/reference verification are deliberately excluded.
        m5_reset_stats(0,0);
        const auto start_prep = std::chrono::steady_clock::now();
        double origin[3];
        calculate_centroid(xyz,n,origin);
        std::vector<Region> regions=build_regions(xyz,n,origin);
        const auto end_prep = std::chrono::steady_clock::now();
        Counters counters;
        const std::vector<int> selected=spherical_fps(xyz,n,m,origin,regions,counters);
        const auto end_sample = std::chrono::steady_clock::now();
        m5_dump_stats(0,0);
        const double prep_ms = std::chrono::duration<double, std::milli>(end_prep-start_prep).count();
        const double sample_ms = std::chrono::duration<double, std::milli>(end_sample-end_prep).count();
        std::printf("PREPARATION_MS=%.6f\nSAMPLING_MS=%.6f\nTOTAL_MS=%.6f\n",
                    prep_ms, sample_ms, prep_ms+sample_ms);

        const uint64_t all_evals=static_cast<uint64_t>(n)*(m-1);
        const double skip_pct=all_evals ?
            100.0*(1.0-static_cast<double>(counters.evaluated_distances)/all_evals) : 0.0;
        std::printf("ALGORITHM=LowerBoundSphericalFPS\n");
        std::printf("DATASET=%s\nN=%d\nM=%d\n",argv[1],n,m);
        std::printf("AZ_BINS=%d\nEL_BINS=%d\nRAD_BINS=%d\n",AZ_BINS,EL_BINS,RAD_BINS);
        std::printf("OCCUPIED_REGIONS=%d\n",counters.occupied_regions);
        std::printf("REGION_TESTS=%llu\nREGIONS_SKIPPED=%llu\n",
             static_cast<unsigned long long>(counters.checked_regions),
             static_cast<unsigned long long>(counters.skipped_regions));
        std::printf("POINT_DISTANCE_EVALS=%llu\nSKIPPED_POINT_EVALS=%llu\n",
             static_cast<unsigned long long>(counters.evaluated_distances),
             static_cast<unsigned long long>(counters.skipped_points));
        std::printf("DISTANCE_EVAL_REDUCTION_PCT=%.5f\n",skip_pct);
        for (int i=0;i<m;++i) std::printf("FPS_INDEX[%d]=%d\n",i,selected[i]);

        std::printf("CORRECTNESS=NOT_CHECKED (performance-only run)\n");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr,"ERROR: %s\n",e.what());
        return 2;
    }
}
