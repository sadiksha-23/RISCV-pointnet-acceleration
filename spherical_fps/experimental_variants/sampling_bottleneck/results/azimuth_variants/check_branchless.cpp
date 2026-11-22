#define main original_fps_entry
#include "branchless.cpp"
#undef main
static int reference_azimuth_bin(double x,double y) {
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

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <limits>
int main() {
    uint64_t count=0,mismatch=0;
    auto check=[&](double x,double y) {
        const int a=reference_azimuth_bin(x,y), b=direct_azimuth_bin(x,y);
        if(a!=b) { if(mismatch<10) std::printf("MISMATCH x=%.17g y=%.17g reference=%d candidate=%d\n",x,y,a,b);++mismatch; }
        ++count;
    };
    check(0,0);check(1,0);check(-1,0);check(0,1);check(0,-1);
    for(int k=0;k<24;++k) {
        const double a=TWO_PI*static_cast<double>(k)/24.0;
        const double x=std::cos(a),y=std::sin(a);
        for(double scale : {1.0, 1e-8, 1e8}) {
            const double xx=x*scale,yy=y*scale;
            check(xx,yy);
            check(std::nextafter(xx,std::numeric_limits<double>::infinity()),yy);
            check(std::nextafter(xx,-std::numeric_limits<double>::infinity()),yy);
            check(xx,std::nextafter(yy,std::numeric_limits<double>::infinity()));
            check(xx,std::nextafter(yy,-std::numeric_limits<double>::infinity()));
        }
    }
    uint64_t state=0x8e5a3b42d191ec75ULL;
    for(int j=0;j<250000;++j) {
        state=state*2862933555777941757ULL+3037000493ULL;
        const double x=static_cast<double>((state>>12)&0xFFFFF)-524288.0;
        state=state*2862933555777941757ULL+3037000493ULL;
        const double y=static_cast<double>((state>>12)&0xFFFFF)-524288.0;
        check(x,y);
    }
    std::printf("CLASSIFIER_CASES=%llu\nCLASSIFIER_MISMATCHES=%llu\n",(unsigned long long)count,(unsigned long long)mismatch);
    return mismatch ? 1 : 0;
}
