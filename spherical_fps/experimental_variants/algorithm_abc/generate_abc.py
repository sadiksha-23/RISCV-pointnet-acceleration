#!/usr/bin/env python3
"""Generate independent A/B/C variants from the user's current 01 reference.
Does not modify original repository sources. Refuses unsupported source layouts.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]  # spherical_fps/experimental_variants/algorithm_abc -> repo
SOURCE = REPO / 'spherical_fps/experimental_variants/optimization_tests/spherical_fps_01_direct_sectors.cpp'


def replace_once(s: str, old: str, new: str, label: str) -> str:
    count = s.count(old)
    if count != 1:
        raise RuntimeError(f'{label}: expected exactly one match, found {count}. Baseline may have changed.')
    return s.replace(old, new, 1)


def make_A(s):
    s = replace_once(s,
        '    float max_nearest_sq=1e38f;\n',
        '    double xmin=std::numeric_limits<double>::infinity(), ymin=std::numeric_limits<double>::infinity(), zmin=std::numeric_limits<double>::infinity();\n'
        '    double xmax=-std::numeric_limits<double>::infinity(), ymax=-std::numeric_limits<double>::infinity(), zmax=-std::numeric_limits<double>::infinity();\n'
        '    float max_nearest_sq=1e38f;\n', 'A: bounding box fields')
    s = replace_once(s,
        '        g.cx+=xyz[p]; g.cy+=xyz[p+1]; g.cz+=xyz[p+2];\n',
        '        g.cx+=xyz[p]; g.cy+=xyz[p+1]; g.cz+=xyz[p+2];\n'
        '        g.xmin=std::min(g.xmin, static_cast<double>(xyz[p])); g.xmax=std::max(g.xmax, static_cast<double>(xyz[p]));\n'
        '        g.ymin=std::min(g.ymin, static_cast<double>(xyz[p+1])); g.ymax=std::max(g.ymax, static_cast<double>(xyz[p+1]));\n'
        '        g.zmin=std::min(g.zmin, static_cast<double>(xyz[p+2])); g.zmax=std::max(g.zmax, static_cast<double>(xyz[p+2]));\n',
        'A: construct bounding boxes')
    s = replace_once(s,
        '    const double combined=std::max(xyz_lb,radial_lb);\n',
        '    // AABB distance is another safe lower bound: exact points are inside the box.\n'
        '    const double bx=std::max({g.xmin-x, 0.0, x-g.xmax});\n'
        '    const double by=std::max({g.ymin-y, 0.0, y-g.ymax});\n'
        '    const double bz=std::max({g.zmin-z, 0.0, z-g.zmax});\n'
        '    const double aabb_lb=std::sqrt(bx*bx+by*by+bz*bz);\n'
        '    const double combined=std::max({xyz_lb,radial_lb,aabb_lb});\n',
        'A: combine bounds')
    return s


def make_B(s):
    marker = '    return occupied;\n}\n\n// Lower bound on distance(newest, every point in a region).'
    code = r'''    // B: independently split oversized occupied regions along the widest XYZ axis.
    // Refinement affects only region geometry, never the FPS selection rule.
    constexpr size_t SPLIT_THRESHOLD = 512;
    constexpr int MAX_SPLIT_DEPTH = 5;
    std::vector<Region> pending;
    std::vector<int> depths;
    pending.reserve(occupied.size());
    depths.reserve(occupied.size());
    for (Region& g : occupied) {
        pending.push_back(std::move(g));
        depths.push_back(0);
    }
    std::vector<Region> refined;
    refined.reserve(pending.size() * 2);
    while (!pending.empty()) {
        Region g = std::move(pending.back()); pending.pop_back();
        int depth = depths.back(); depths.pop_back();
        if (g.members.size() <= SPLIT_THRESHOLD || depth >= MAX_SPLIT_DEPTH) {
            refined.push_back(std::move(g));
            continue;
        }
        double lo[3] = {1e300,1e300,1e300};
        double hi[3] = {-1e300,-1e300,-1e300};
        for (int idx : g.members) for (int d=0;d<3;++d) {
            double v=xyz[static_cast<size_t>(idx)*3+d];
            lo[d]=std::min(lo[d],v); hi[d]=std::max(hi[d],v);
        }
        int axis=0;
        for (int d=1;d<3;++d) if (hi[d]-lo[d]>hi[axis]-lo[axis]) axis=d;
        // Stable sort keeps deterministic ordering in the presence of ties.
        std::stable_sort(g.members.begin(),g.members.end(),[&](int a,int b){
            double va=xyz[static_cast<size_t>(a)*3+axis];
            double vb=xyz[static_cast<size_t>(b)*3+axis];
            return va<vb || (va==vb && a<b);
        });
        const size_t mid=g.members.size()/2;
        for (int half=0;half<2;++half) {
            Region child;
            const size_t begin=half==0 ? 0 : mid;
            const size_t end=half==0 ? mid : g.members.size();
            child.members.assign(g.members.begin()+begin,g.members.begin()+end);
            for (int idx : child.members) {
                const size_t p=static_cast<size_t>(idx)*3;
                child.cx+=xyz[p]; child.cy+=xyz[p+1]; child.cz+=xyz[p+2];
                const double dx=static_cast<double>(xyz[p])-origin[0];
                const double dy=static_cast<double>(xyz[p+1])-origin[1];
                const double dz=static_cast<double>(xyz[p+2])-origin[2];
                const double r=std::sqrt(dx*dx+dy*dy+dz*dz);
                child.radial_min=std::min(child.radial_min,r);
                child.radial_max=std::max(child.radial_max,r);
            }
            const double inv=1.0/static_cast<double>(child.members.size());
            child.cx*=inv; child.cy*=inv; child.cz*=inv;
            for (int idx : child.members) {
                const size_t p=static_cast<size_t>(idx)*3;
                const double dx=static_cast<double>(xyz[p])-child.cx;
                const double dy=static_cast<double>(xyz[p+1])-child.cy;
                const double dz=static_cast<double>(xyz[p+2])-child.cz;
                child.radius=std::max(child.radius,std::sqrt(dx*dx+dy*dy+dz*dz));
            }
            child.radius+=1e-7*(1.0+child.radius);
            child.radial_min=std::max(0.0,child.radial_min-1e-7*(1.0+child.radial_min));
            child.radial_max+=1e-7*(1.0+child.radial_max);
            child.best_index=child.members.front();
            pending.push_back(std::move(child));
            depths.push_back(depth+1);
        }
    }
    return refined;
}

// Lower bound on distance(newest, every point in a region).'''
    return replace_once(s,marker,code,'B: refined region return')


def make_C(s):
    original = '''            const double lb=lower_bound(g,xyz,newest,selected_radius);
            const double max_sq=static_cast<double>(g.max_nearest_sq);
            const double margin=1e-5*(1.0+std::abs(max_sq));
            if (lb*lb > max_sq+margin) {'''
    updated = '''            const double max_sq=static_cast<double>(g.max_nearest_sq);
            const double margin=1e-5*(1.0+std::abs(max_sq));
            // C: cheap radial-only safe rejection before expensive sqrt(center distance).
            // Uses the same conservative numerical margin as lower_bound().
            const double radial_lb=std::max({0.0,
                    g.radial_min-selected_radius,
                    selected_radius-g.radial_max});
            const double radial_safe=std::max(0.0,radial_lb-1e-6*(1.0+radial_lb));
            if (radial_safe*radial_safe > max_sq+margin) {
                ++c.skipped_regions;
                c.skipped_points+=g.members.size();
                continue;
            }
            const double lb=lower_bound(g,xyz,newest,selected_radius);
            if (lb*lb > max_sq+margin) {'''
    return replace_once(s,original,updated,'C: short circuit radial bound')


def main():
    if not SOURCE.is_file():
        raise SystemExit(f'Missing 01 baseline: {SOURCE}\nRun from the intended repository layout.')
    original=SOURCE.read_text()
    assert 'static std::vector<Region> build_regions(' in original
    for name,fn in [('A_aabb_bound',make_A),('B_adaptive_regions',make_B),('C_radial_short_circuit',make_C)]:
        target=HERE / f'spherical_fps_{name}.cpp'
        target.write_text('// Generated experiment from optimization 01; originals remain untouched.\n'+fn(original))
        print('Generated:',target)
    print('No original repository files were changed.')

if __name__=='__main__':main()
