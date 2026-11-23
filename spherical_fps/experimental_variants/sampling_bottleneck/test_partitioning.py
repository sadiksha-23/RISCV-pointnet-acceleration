#!/usr/bin/env python3
"""Compare F2 spherical binning with Cartesian binning using the SAME FPS loop.

Run from any directory:
    python spherical_fps/experimental_variants/partition_comparison/test_partitioning.py

Requires a native g++ compiler and datasets/kitti/0000000000.bin.
No edits are made to the original F2 implementation.
"""
from __future__ import annotations

import csv
import re
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'spherical_fps/ref_implementation/spherical_fps.cpp'
DATASET = ROOT / 'datasets/kitti/0000000000.bin'
RESULTS = Path(__file__).resolve().parent / 'results'

N = 100000
M_VALUES = [128, 256, 512, 1024]
REPEATS = 7
COMPILER = 'g++'
FLAGS = ['-O3', '-std=c++17', '-ffp-contract=off']

# Both variants use exactly 24*8*6 = 12*12*8 = 1152 possible bins.
# Cartesian axes use x=12, y=12, z=8: regular equal-width Cartesian bins.
CART_X, CART_Y, CART_Z = 12, 12, 8

# Only replace build_regions(). The lower_bound(), spherical_fps() and main() from
# original F2 are preserved byte-for-byte in the Cartesian generated program.
CARTESIAN_BUILD_REGIONS = r'''
static std::vector<Region> build_regions(const std::vector<float>& xyz,int n,const double origin[3]) {
    constexpr int NX = 12, NY = 12, NZ = 8;
    static_assert(NX*NY*NZ == AZ_BINS*EL_BINS*RAD_BINS,
        "Cartesian and spherical variants must have equal candidate bin counts");
    std::vector<Region> grid(static_cast<size_t>(NX)*NY*NZ);
    double lo[3] = {std::numeric_limits<double>::infinity(),
                    std::numeric_limits<double>::infinity(),
                    std::numeric_limits<double>::infinity()};
    double hi[3] = {-std::numeric_limits<double>::infinity(),
                    -std::numeric_limits<double>::infinity(),
                    -std::numeric_limits<double>::infinity()};
    for (int i=0;i<n;++i) {
        const size_t p=static_cast<size_t>(i)*3;
        for(int d=0;d<3;++d) {
            const double v=static_cast<double>(xyz[p+d]);
            lo[d]=std::min(lo[d],v);
            hi[d]=std::max(hi[d],v);
        }
    }
    for (int i=0;i<n;++i) {
        const size_t p=static_cast<size_t>(i)*3;
        const double x=static_cast<double>(xyz[p]);
        const double y=static_cast<double>(xyz[p+1]);
        const double z=static_cast<double>(xyz[p+2]);
        const auto bin = [](double v,double lower,double upper,int count) {
            if(!(upper>lower)) return 0;
            const double scaled=(v-lower)*static_cast<double>(count)/(upper-lower);
            return std::clamp(static_cast<int>(scaled),0,count-1);
        };
        const int xi=bin(x,lo[0],hi[0],NX);
        const int yi=bin(y,lo[1],hi[1],NY);
        const int zi=bin(z,lo[2],hi[2],NZ);
        Region& g=grid[(static_cast<size_t>(zi)*NY+yi)*NX+xi];
        g.members.push_back(i);
        g.cx+=x;g.cy+=y;g.cz+=z;
        g.xmin=std::min(g.xmin,x);g.xmax=std::max(g.xmax,x);
        g.ymin=std::min(g.ymin,y);g.ymax=std::max(g.ymax,y);
        g.zmin=std::min(g.zmin,z);g.zmax=std::max(g.zmax,z);
        const double ox=x-origin[0],oy=y-origin[1],oz=z-origin[2];
        const double rsq=ox*ox+oy*oy+oz*oz;
        g.radial_min_sq=std::min(g.radial_min_sq,rsq);
        g.radial_max_sq=std::max(g.radial_max_sq,rsq);
    }
    std::vector<Region> occupied;
    occupied.reserve(std::min(grid.size(),static_cast<size_t>(n)));
    for (Region& g:grid) {
        if(g.members.empty()) continue;
        const double count=static_cast<double>(g.members.size());
        g.cx/=count;g.cy/=count;g.cz/=count;
        g.radial_min=std::sqrt(g.radial_min_sq);
        g.radial_max=std::sqrt(g.radial_max_sq);
        for(int i:g.members) {
            const size_t p=static_cast<size_t>(i)*3;
            const double dx=static_cast<double>(xyz[p])-g.cx;
            const double dy=static_cast<double>(xyz[p+1])-g.cy;
            const double dz=static_cast<double>(xyz[p+2])-g.cz;
            g.radius=std::max(g.radius,std::sqrt(dx*dx+dy*dy+dz*dz));
        }
        g.radius+=1e-7*(1.0+g.radius);
        g.radial_min=std::max(0.0,g.radial_min-1e-7*(1.0+g.radial_min));
        g.radial_max+=1e-7*(1.0+g.radial_max);
        g.best_index=g.members.front();
        occupied.push_back(std::move(g));
    }
    return occupied;
}
'''


def run_command(args: list[str], *, capture: bool = False) -> str:
    p = subprocess.run(args, cwd=ROOT, text=True, stdout=subprocess.PIPE if capture else None,
                       stderr=subprocess.PIPE if capture else None, check=False)
    if p.returncode:
        detail = (p.stderr or '') + '\n' + (p.stdout or '')
        raise RuntimeError(f'Command failed ({p.returncode}): {args}\n{detail}')
    return p.stdout if capture else ''


def parse_output(output: str) -> dict:
    entries = dict(re.findall(r'^([A-Z][A-Z0-9_]*)=([^\r\n]+)', output, flags=re.M))
    indices = [int(x) for x in re.findall(r'^FPS_INDEX\[\d+\]=(\d+)', output, flags=re.M)]
    needed = ['PREPARATION_MS','SAMPLING_MS','TOTAL_MS','OCCUPIED_REGIONS',
              'REGION_TESTS','REGIONS_SKIPPED','POINT_DISTANCE_EVALS']
    missing = [k for k in needed if k not in entries]
    if missing or not indices:
        raise ValueError(f'Missing output fields: {missing}; FPS indices found={len(indices)}')
    return {
        'indices': indices,
        **{k: float(entries[k]) if k.endswith('_MS') else int(entries[k]) for k in needed},
    }


def median(values: list[float]) -> float:
    return statistics.median(values)


def main() -> None:
    if not SOURCE.is_file():
        raise SystemExit(f'F2 source not found: {SOURCE}')
    if not DATASET.is_file():
        raise SystemExit(f'KITTI frame not found: {DATASET}')
    RESULTS.mkdir(parents=True, exist_ok=True)
    original = SOURCE.read_text(encoding='utf-8')
    start = original.find('static std::vector<Region> build_regions(')
    end = original.find('static double lower_bound(', start)
    if start < 0 or end <= start:
        raise SystemExit('Unexpected F2 source layout; refusing to generate variant.')
    if '#define AZ_BINS 24' not in original or '#define EL_BINS 8' not in original or '#define RAD_BINS 6' not in original:
        raise SystemExit('The bin defaults have changed; review CART_X/Y/Z before testing.')

    changed = original[:start] + CARTESIAN_BUILD_REGIONS + '\n' + original[end:]
    # In particular, all FPS update logic and geometric lower bounds are identical.
    if not changed.endswith(original[end:]):
        raise RuntimeError('Internal safety check failed: sampling code was modified')
    cart_source = RESULTS / 'spherical_fps_cartesian_partition.cpp'
    cart_source.write_text(changed, encoding='utf-8')

    exe_suffix = '.exe' if __import__('os').name == 'nt' else ''
    executables = {
        'spherical': RESULTS / ('f2_spherical_native'+exe_suffix),
        'cartesian': RESULTS / ('f2_cartesian_native'+exe_suffix),
    }
    for name, src in [('spherical', SOURCE), ('cartesian', cart_source)]:
        cmd=[COMPILER, *FLAGS, str(src), '-o', str(executables[name])]
        print('Compiling:', ' '.join(cmd), flush=True)
        run_command(cmd)

    rows: list[dict] = []
    summaries: list[dict] = []
    for m in M_VALUES:
        collected = {'spherical': [], 'cartesian': []}
        for repeat in range(REPEATS):
            # Alternating order reduces bias from warmup/thermal/OS background load.
            order=['spherical','cartesian'] if repeat%2==0 else ['cartesian','spherical']
            current = {}
            for name in order:
                output=run_command([str(executables[name]),str(DATASET),str(N),str(m)],capture=True)
                parsed=parse_output(output)
                if len(parsed['indices'])!=m:
                    raise RuntimeError(f'{name} returned wrong selection count for M={m}')
                current[name]=parsed
                (RESULTS/f'{name}_N{N}_M{m}_run{repeat+1}.txt').write_text(output,encoding='utf-8')
                row={'partition':name,'N':N,'M':m,'run':repeat+1,
                     **{k:v for k,v in parsed.items() if k!='indices'}}
                rows.append(row)
                collected[name].append(parsed)
            if current['spherical']['indices']!=current['cartesian']['indices']:
                a=current['spherical']['indices'];b=current['cartesian']['indices']
                first=next(i for i,(x,y) in enumerate(zip(a,b)) if x!=y)
                raise RuntimeError(f'FPS INDEX MISMATCH at N={N}, M={m}, run={repeat+1}, index={first}: spherical={a[first]}, cartesian={b[first]}')
            print(f'N={N} M={m} run={repeat+1}: indices MATCH; '
                  f'spherical total={current["spherical"]["TOTAL_MS"]:.3f}ms; '
                  f'cartesian total={current["cartesian"]["TOTAL_MS"]:.3f}ms',flush=True)

        sums={}
        for name in ['spherical','cartesian']:
            readings=collected[name]
            sums[name]={k: median([v[k] for v in readings]) for k in
                        ['PREPARATION_MS','SAMPLING_MS','TOTAL_MS','OCCUPIED_REGIONS',
                         'REGION_TESTS','REGIONS_SKIPPED','POINT_DISTANCE_EVALS']}
        percent=100*(sums['spherical']['TOTAL_MS']-sums['cartesian']['TOTAL_MS'])/sums['spherical']['TOTAL_MS']
        print(f'\nM={m} median native timings (ms):')
        print(f'  spherical: prep={sums["spherical"]["PREPARATION_MS"]:.3f} '
              f'sampling={sums["spherical"]["SAMPLING_MS"]:.3f} total={sums["spherical"]["TOTAL_MS"]:.3f}')
        print(f'  cartesian: prep={sums["cartesian"]["PREPARATION_MS"]:.3f} '
              f'sampling={sums["cartesian"]["SAMPLING_MS"]:.3f} total={sums["cartesian"]["TOTAL_MS"]:.3f}')
        print(f'  cartesian vs spherical total: {percent:+.2f}% improvement (positive means Cartesian faster)')
        print(f'  occupied regions: spherical={sums["spherical"]["OCCUPIED_REGIONS"]:.0f}, '
              f'cartesian={sums["cartesian"]["OCCUPIED_REGIONS"]:.0f}')
        print(f'  evaluated distances: spherical={sums["spherical"]["POINT_DISTANCE_EVALS"]:.0f}, '
              f'cartesian={sums["cartesian"]["POINT_DISTANCE_EVALS"]:.0f}\n')
        for name in ['spherical','cartesian']:
            summaries.append({'partition':name,'N':N,'M':m,**sums[name]})

    with (RESULTS/'runs.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (RESULTS/'summary.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    print('All selected indices MATCH across both grouping methods.')
    print('Results:', RESULTS)
    print('NOTE: Native CPU results only; not a gem5 performance comparison.')
    print('NOTE: This tests one equal-width Cartesian grid (12x12x8) versus F2 spherical (24x8x6).')
    print('      Both have 1152 possible bins, but may have different occupied region counts.')
    print('      This isolates grouping inside the F2 framework, not the full RadiusFPS implementation.')


if __name__=='__main__':
    main()
