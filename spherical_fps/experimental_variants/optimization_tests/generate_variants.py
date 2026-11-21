#!/usr/bin/env python3
"""Derive independent 05/06/07 experiments from the repository's unchanged 01 source."""
from pathlib import Path
root=Path(__file__).resolve().parents[3] if Path(__file__).resolve().parent.name=='optimization_tests' else None
# This file is intended to live in spherical_fps/experimental_variants/optimization_tests/
here=Path(__file__).resolve().parent
original=here/'spherical_fps_01_direct_sectors.cpp'
if not original.is_file(): raise SystemExit(f'Missing {original}; install the ZIP in optimization_tests/')
base=original.read_text()
def replace_once(s,old,new):
    n=s.count(old)
    if n!=1: raise RuntimeError(f'Expected exactly one match (found {n}) for: {old[:90]!r}')
    return s.replace(old,new,1)

# 05: compare signed squares instead of taking a horizontal square root.
s05=base
s05=replace_once(s05,'''    const double horizontal = std::sqrt(x*x+y*y);
    if (horizontal==0.0) return z>=0 ? EL_BINS-1 : 0;''','''    const double horizontal_sq = x*x+y*y;
    if (horizontal_sq==0.0) return z>=0 ? EL_BINS-1 : 0;''')
s05=replace_once(s05,'''        if (z*edge.second >= horizontal*edge.first) lo=mid;
        else hi=mid;''','''        // cos(theta)>0 for internal boundaries. Preserve signs before squaring.
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
        else hi=mid;''')

# 06: no per-point radial sqrt: assign bins from squared thresholds;
# retain conservative spherical interval distances via 2 sqrt calls per occupied region.
s06=base
s06=replace_once(s06,'''    float max_nearest_sq=1e38f;''','''    double radial_min_sq=std::numeric_limits<double>::infinity();
    double radial_max_sq=0.0;
    float max_nearest_sq=1e38f;''')
s06=replace_once(s06,'''    std::vector<double> radii(static_cast<size_t>(n));''','''    std::vector<double> radii_sq(static_cast<size_t>(n));''')
s06=replace_once(s06,'''    double global_rmax=1e-12;''','''    double global_rmax_sq=1e-24;''')
s06=replace_once(s06,'''        const double r=std::sqrt(x*x+y*y+z*z);
        radii[i]=r;''','''        const double r_sq=x*x+y*y+z*z;
        radii_sq[i]=r_sq;''')
s06=replace_once(s06,'''        if (r>global_rmax) global_rmax=r;''','''        if (r_sq>global_rmax_sq) global_rmax_sq=r_sq;''')
s06=replace_once(s06,'''    for (int i=0; i<n; ++i) {
        const int ai=az_bin[i];''','''    // Precompute squared radial boundaries once, not one square root per point.
    const double global_rmax=std::sqrt(global_rmax_sq);
    std::vector<double> radial_edges_sq(static_cast<size_t>(RAD_BINS)+1);
    for (int k=1;k<RAD_BINS;++k) {
        const double boundary=global_rmax*static_cast<double>(k)/RAD_BINS;
        radial_edges_sq[k]=boundary*boundary;
    }
    for (int i=0; i<n; ++i) {
        const int ai=az_bin[i];''')
s06=replace_once(s06,'''        const int ri=std::clamp(static_cast<int>(radii[i]/global_rmax*RAD_BINS),0,RAD_BINS-1);''','''        const double rsq=radii_sq[i];
        // upper_bound behavior: equality at a boundary goes into the upper bin.
        const int ri=static_cast<int>(std::upper_bound(radial_edges_sq.begin()+1,
            radial_edges_sq.begin()+RAD_BINS,rsq)- (radial_edges_sq.begin()+1));''')
s06=replace_once(s06,'''        g.radial_min=std::min(g.radial_min,radii[i]);
        g.radial_max=std::max(g.radial_max,radii[i]);''','''        g.radial_min_sq=std::min(g.radial_min_sq,rsq);
        g.radial_max_sq=std::max(g.radial_max_sq,rsq);''')
s06=replace_once(s06,'''        g.cx/=count; g.cy/=count; g.cz/=count;
        for (int i: g.members) {''','''        g.cx/=count; g.cy/=count; g.cz/=count;
        g.radial_min=std::sqrt(g.radial_min_sq);
        g.radial_max=std::sqrt(g.radial_max_sq);
        for (int i: g.members) {''')

# 07: max squared radius during scan, one sqrt per occupied region.
s07=base
s07=replace_once(s07,'''        for (int i: g.members) {
            const size_t p=static_cast<size_t>(i)*3;
            const double dx=static_cast<double>(xyz[p])-g.cx;
            const double dy=static_cast<double>(xyz[p+1])-g.cy;
            const double dz=static_cast<double>(xyz[p+2])-g.cz;
            g.radius=std::max(g.radius,std::sqrt(dx*dx+dy*dy+dz*dz));
        }
        // Outward rounding''','''        double max_radius_sq=0.0;
        for (int i: g.members) {
            const size_t p=static_cast<size_t>(i)*3;
            const double dx=static_cast<double>(xyz[p])-g.cx;
            const double dy=static_cast<double>(xyz[p+1])-g.cy;
            const double dz=static_cast<double>(xyz[p+2])-g.cz;
            max_radius_sq=std::max(max_radius_sq,dx*dx+dy*dy+dz*dz);
        }
        g.radius=std::sqrt(max_radius_sq);
        // Outward rounding''')

for name,s in [('05_elevation_squared',s05),('06_radial_squared',s06),('07_region_radius_squared',s07)]:
    out=here/f'spherical_fps_{name}.cpp'
    out.write_text('// GENERATED from unchanged optimization 01; see generate_variants.py\n'+s)
    print('Created',out)
print('No original repository files were modified.')
