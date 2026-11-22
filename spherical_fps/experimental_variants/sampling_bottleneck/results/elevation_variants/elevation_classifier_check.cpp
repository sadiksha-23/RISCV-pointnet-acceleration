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
static int elevation_reference(double x,double y,double z) {
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

static inline bool sign_elevation_above(double horizontal_sq, double z, int mid) {
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

static int elevation_sign_first(double x,double y,double z) {
    const double horizontal_sq=x*x+y*y;
    if(horizontal_sq==0.0) return z>=0 ? EL_BINS-1 : 0;
    // The middle boundary is exactly 0 degrees: equivalent to z>=0 here.
    if(z>=0.0) {
        if(sign_elevation_above(horizontal_sq,z,6))
            return sign_elevation_above(horizontal_sq,z,7) ? 7 : 6;
        return sign_elevation_above(horizontal_sq,z,5) ? 5 : 4;
    }
    if(sign_elevation_above(horizontal_sq,z,2))
        return sign_elevation_above(horizontal_sq,z,3) ? 3 : 2;
    return sign_elevation_above(horizontal_sq,z,1) ? 1 : 0;
}

static inline bool tree_elevation_above(double horizontal_sq, double z, int mid) {
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

static int elevation_fixed_tree(double x,double y,double z) {
    const double horizontal_sq=x*x+y*y;
    if(horizontal_sq==0.0) return z>=0 ? EL_BINS-1 : 0;
    // Fully unrolled decision tree, same F2 comparison at every boundary.
    if(tree_elevation_above(horizontal_sq,z,4)) {
        if(tree_elevation_above(horizontal_sq,z,6)) {
            if(tree_elevation_above(horizontal_sq,z,7)) return 7;
            return 6;
        }
        if(tree_elevation_above(horizontal_sq,z,5)) return 5;
        return 4;
    }
    if(tree_elevation_above(horizontal_sq,z,2)) {
        if(tree_elevation_above(horizontal_sq,z,3)) return 3;
        return 2;
    }
    if(tree_elevation_above(horizontal_sq,z,1)) return 1;
    return 0;
}


#include <random>
int main() {
    uint64_t count=0, mismatches=0;
    auto check=[&](double x,double y,double z) {
        int a=elevation_reference(x,y,z), b=elevation_sign_first(x,y,z), c=elevation_fixed_tree(x,y,z);
        ++count;
        if(a!=b || a!=c) {
            if(mismatches<8) std::printf("MISMATCH x=%.17g y=%.17g z=%.17g ref=%d sign=%d tree=%d\n",x,y,z,a,b,c);
            ++mismatches;
        }
    };
    check(0,0,0);check(0,0,1);check(0,0,-1);
    check(1,0,0);check(-1,0,0);check(0,1,0);check(0,-1,0);
    for(int k=0;k<=8;++k) {
        double a=-PI/2.0+(PI*k)/8.0;
        for(double az : {0.0,0.1,0.6,1.4,3.1,4.8}) {
            double x=std::cos(az)*std::cos(a),y=std::sin(az)*std::cos(a),z=std::sin(a);
            check(x,y,z);
            check(x,y,std::nextafter(z,-std::numeric_limits<double>::infinity()));
            check(x,y,std::nextafter(z,std::numeric_limits<double>::infinity()));
            check(x*1000,y*1000,z*1000);
        }
    }
    std::mt19937_64 rng(20261121);
    std::uniform_real_distribution<double> dist(-1000,1000);
    for(int i=0;i<300000;++i) check(dist(rng),dist(rng),dist(rng));
    std::printf("CLASSIFIER_CASES=%llu\nCLASSIFIER_MISMATCHES=%llu\n",(unsigned long long)count,(unsigned long long)mismatches);
    return mismatches ? 1 : 0;
}
